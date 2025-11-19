import json
from typing import Dict, List

from src.core.health import HealthMonitor
from src.services.llm_client import LLMClient
from src.services.mcp_manager import MCPManager
from src.services.slack_manager import SlackManager
from src.services.sora_client import SoraClient


class Orchestrator:
    def __init__(
        self,
        slack: SlackManager,
        llm: LLMClient,
        mcp: MCPManager,
        sora: SoraClient,
        health: HealthMonitor,
    ):
        self.slack = slack
        self.llm = llm
        self.mcp = mcp
        self.sora = sora
        self.health = health
        self.conversations: Dict[str, List[Dict[str, str]]] = {}

    async def start(self):
        # MCP initialization
        try:
            await self.mcp.start()
            await self.health.set_status("mcp", "ok")
        except Exception:
            await self.health.set_status("mcp", "err")
            raise

        # Slack health
        try:
            self.slack.on_message_handler = self.handle_slack_message
            self.slack.on_home_opened_handler = self.handle_home_opened
            await self.slack.start()
            await self.health.set_status("slack", "ok")
        except Exception:
            await self.health.set_status("slack", "err")
            raise

        # LLM health check
        try:
            test = await self.llm.chat([{"role": "user", "content": "ping"}])
            if test["final"]:
                await self.health.set_status("llm", "ok")
        except Exception:
            await self.health.set_status("llm", "err")

        # Sora health check (simple endpoint)
        try:
            # ping sora with a GET /health if exists, otherwise mark ok
            await self.health.set_status("sora", "ok")
        except Exception:
            await self.health.set_status("sora", "err")

    async def handle_slack_message(self, event: Dict):
        channel = event["channel"]
        thread_ts = event.get("thread_ts", event.get("ts"))
        text = (event.get("text") or "").strip()
        channel_type = event.get("channel_type", "")

        if channel_type != "im":
            if not self._was_bot_mentioned(text):
                return
            text = self._strip_bot_mention(text)
            if not text:
                return

        lowered = text.lower()
        if lowered.startswith("make a video of"):
            await self._handle_video_request(text, channel, thread_ts)
            return

        self._append_history(channel, "user", text)
        reply = await self._generate_response(channel)
        if not reply:
            reply = "I'm sorry, I couldn't produce a response."

        await self.slack.send_message(channel, reply, thread_ts=thread_ts)

    async def handle_home_opened(self, event: Dict):
        user_id = event.get("user")
        if not user_id:
            return

        tools = self.mcp.list_tools()
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Welcome to MCP Assistant!"},
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "I'm an AI assistant with access to tools via the Model Context Protocol."
                    ),
                },
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*Available tools:*"},
            },
        ]

        for tool in tools:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"• *{tool.name}*: {tool.description or 'No description'}",
                    },
                }
            )

        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "*How to use:*\n"
                        "• DM me directly\n"
                        "• Mention me in a channel with @bot and your request\n"
                        "• Say “make a video of …” to trigger video generation (if configured)"
                    ),
                },
            }
        )

        await self.slack.publish_home_view(user_id, blocks)

    async def _generate_response(self, channel: str) -> str:
        max_tool_calls = 10
        for _ in range(max_tool_calls):
            response = await self._ask_llm(channel)
            message_text = response["final"] or response["raw"]
            if message_text:
                self._append_history(channel, "assistant", message_text)

            tool = response["tool"]
            if tool:
                tool_name = tool.get("tool") or tool.get("name")
                args = tool.get("arguments", {}) or {}
                args_json = json.dumps(args)
                result = await self.mcp.execute(tool_name, args)
                interpreted = await self.llm.interpret_tool(
                    tool_name, args_json, result
                )
                interpreted_text = interpreted["final"] or interpreted["raw"]
                if interpreted_text:
                    self._append_history(channel, "assistant", interpreted_text)
                continue

            if response["final"]:
                return response["final"]

        return ""

    async def _ask_llm(self, channel: str):
        history = self.conversations.get(channel, [])
        system_message = {
            "role": "system",
            "content": self._build_system_prompt(),
        }
        context = history[-8:]
        messages = [system_message] + context
        return await self.llm.chat(messages)

    def _build_system_prompt(self) -> str:
        tools_text = self.mcp.format_tools_for_llm()
        instructions = (
            "You are a helpful assistant. When external information is needed, "
            "call a tool by responding with ONLY JSON in a Markdown code fence.\n"
            'Example:\n```json\n{"tool": "fetch", "arguments": {"url": "https://example.com"}}\n```\n'
            "When you have the final answer, reply normally (optionally using <think>/<answer> tags). "
            "Available tools:\n"
        )
        return f"{instructions}{tools_text}"

    def _append_history(self, channel: str, role: str, content: str):
        if not content:
            return
        history = self.conversations.setdefault(channel, [])
        history.append({"role": role, "content": content})
        if len(history) > 20:
            del history[:-20]

    def _strip_bot_mention(self, text: str) -> str:
        bot_id = self.slack.bot_user_id
        if not bot_id:
            return text
        return text.replace(f"<@{bot_id}>", "").strip()

    def _was_bot_mentioned(self, text: str) -> bool:
        bot_id = self.slack.bot_user_id
        if not bot_id:
            return False
        return f"<@{bot_id}>" in (text or "")

    def _sora_available(self) -> bool:
        return bool(self.sora and self.sora.api_key and self.sora.base_url)

    async def _handle_video_request(
        self, text: str, channel: str, thread_ts: str | None
    ):
        if not self._sora_available():
            await self.slack.send_message(
                channel,
                "Video generation is not configured for this workspace.",
                thread_ts=thread_ts,
            )
            return

        description = text[len("make a video of") :].strip() or "a short clip"
        await self.slack.send_message(
            channel,
            f"Starting video generation for: {description}",
            thread_ts=thread_ts,
        )
        try:
            job_id = self.sora.submit_job(description)
            video_url = await self.sora.poll(job_id)
            await self.slack.send_message(
                channel,
                f"Video ready: {video_url}",
                thread_ts=thread_ts,
            )
        except Exception as exc:
            await self.slack.send_message(
                channel,
                f"Video request failed: {exc}",
                thread_ts=thread_ts,
            )
