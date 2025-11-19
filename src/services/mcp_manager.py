from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import traceback
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


@dataclass
class MCPServerConfig:
    command: str
    args: List[str] = field(default_factory=list)
    description: str = ""
    env: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MCPServerConfig":
        if "command" not in payload:
            raise ValueError("MCP server definition requires a 'command'")
        return cls(
            command=str(payload["command"]),
            args=[str(arg) for arg in payload.get("args", [])],
            description=str(payload.get("description", "") or ""),
            env={str(k): str(v) for k, v in (payload.get("env") or {}).items()},
        )


@dataclass
class ToolInfo:
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str

    def format_for_llm(self) -> str:
        args_desc = []
        properties = self.input_schema.get("properties", {})
        required = set(self.input_schema.get("required", []))
        for param_name, param_info in properties.items():
            info = param_info if isinstance(param_info, dict) else {}
            desc = info.get("description", "No description")
            suffix = " (required)" if param_name in required else ""
            args_desc.append(f"- {param_name}: {desc}{suffix}")

        return (
            f"Tool: {self.name}\n"
            f"Description: {self.description or 'No description provided'}\n"
            f"Arguments:\n{chr(10).join(args_desc) if args_desc else 'None'}"
        )


class MCPServer:
    def __init__(self, name: str, config: MCPServerConfig):
        self.name = name
        self.config = config
        self._session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()

    async def start(self) -> None:
        command = self._resolve_command(self.config.command)
        env = os.environ.copy()
        env.update(self.config.env)

        server_params = StdioServerParameters(
            command=command,
            args=self.config.args,
            env=env,
        )
        transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = transport
        session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._session = session
        logging.info("Initialized MCP server '%s'", self.name)

    async def stop(self) -> None:
        await self._exit_stack.aclose()
        self._session = None

    async def list_tools(self) -> List[ToolInfo]:
        session = self._require_session()
        response = await session.list_tools()
        tools: List[ToolInfo] = []
        for tool in response.tools:
            schema = (
                getattr(tool, "input_schema", None)
                or getattr(tool, "inputSchema", None)
                or {}
            )
            tools.append(
                ToolInfo(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=schema,
                    server_name=self.name,
                )
            )
        return tools

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        session = self._require_session()
        return await session.call_tool(tool_name, arguments or {})

    def _require_session(self) -> ClientSession:
        if not self._session:
            raise RuntimeError(f"MCP server '{self.name}' not initialized")
        return self._session

    def _resolve_command(self, command: str) -> str:
        resolved = shutil.which(command)
        return resolved or command


class MCPManager:
    """Loads MCP server definitions and proxies tool calls to them."""

    def __init__(self, config_path: str | Path = "mcp_servers.json"):
        self.config_path = Path(config_path)
        self.servers: Dict[str, MCPServer] = {}
        self.tools: Dict[str, ToolInfo] = {}
        self._lock = asyncio.Lock()
        self._started = False

    async def start(self) -> None:
        async with self._lock:
            if self._started:
                return

            configs = self._load_server_configs()
            if not configs:
                logging.info(
                    "No MCP servers configured (expected at %s). Skipping MCP startup.",
                    self.config_path,
                )
                self._started = True
                return

            started = 0
            for name, cfg in configs.items():
                server = MCPServer(name, cfg)
                try:
                    await server.start()
                    tools = await server.list_tools()
                except Exception:
                    logging.exception("Failed to start MCP server '%s'", name)
                    await server.stop()
                    continue

                self.servers[name] = server
                for tool in tools:
                    self.tools[tool.name] = tool
                started += 1

            if started == 0:
                raise RuntimeError("No MCP servers could be started")

            logging.info(
                "MCP Manager loaded %d server(s) and %d tool(s)",
                len(self.servers),
                len(self.tools),
            )
            self._started = True

    async def stop(self) -> None:
        async with self._lock:
            if not self._started:
                return

            for server in self.servers.values():
                try:
                    await server.stop()
                except Exception:
                    logging.exception(
                        "Error shutting down MCP server '%s'", server.name
                    )

            self.servers.clear()
            self.tools.clear()
            self._started = False

    async def execute(self, name: str, args: Dict[str, Any]) -> str:
        if not self._started:
            return "MCP manager is not initialized"

        tool = self.tools.get(name)
        if not tool:
            return f"Unknown tool '{name}'"

        server = self.servers.get(tool.server_name)
        if not server:
            return f"Server for tool '{name}' is unavailable"

        try:
            result = await server.execute_tool(name, args)
            return self._stringify_result(result)
        except Exception:
            logging.exception("Error executing tool '%s'", name)
            return traceback.format_exc()

    def list_tools(self) -> List[ToolInfo]:
        return list(self.tools.values())

    def format_tools_for_llm(self) -> str:
        tools = sorted(self.tools.values(), key=lambda t: t.name.lower())
        if not tools:
            return "No external tools are configured."
        return "\n\n".join(tool.format_for_llm() for tool in tools)

    def _load_server_configs(self) -> Dict[str, MCPServerConfig]:
        if not self.config_path.exists():
            logging.warning("MCP server config '%s' not found", self.config_path)
            return {}

        raw = self.config_path.read_text()
        expanded = self._expand_env(raw)
        try:
            data = json.loads(expanded)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in MCP config '{self.config_path}': {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"MCP config '{self.config_path}' must be a JSON object mapping names to definitions"
            )

        configs: Dict[str, MCPServerConfig] = {}
        for name, payload in data.items():
            if not isinstance(payload, dict):
                logging.warning("Ignoring invalid MCP server entry for '%s'", name)
                continue
            try:
                configs[name] = MCPServerConfig.from_dict(payload)
            except Exception as exc:
                logging.error("Invalid MCP server definition for '%s': %s", name, exc)

        return configs

    def _expand_env(self, raw: str) -> str:
        def replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            value = os.getenv(var_name)
            if value is None:
                raise RuntimeError(
                    f"Environment variable '{var_name}' is required for MCP config but is not set"
                )
            return value

        return ENV_PATTERN.sub(replace, raw)

    def _stringify_result(self, result: Any) -> str:
        if hasattr(result, "content"):
            parts = []
            for item in getattr(result, "content", []):
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
                else:
                    parts.append(str(item))
            if parts:
                return "\n".join(parts)

        if hasattr(result, "model_dump_json"):
            try:
                return result.model_dump_json(indent=2)
            except TypeError:
                return json.dumps(result.model_dump(), indent=2)

        return str(result)
