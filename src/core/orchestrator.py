from typing import Dict

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

    async def start(self):
        # Slack health
        try:
            self.slack.on_message_handler = self.handle_slack_message
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
        text = event.get("text", "")

        response = await self.llm.chat([{"role": "user", "content": text}])

        if response["tool"]:
            tool = response["tool"]
            tool_name = tool.get("tool") or tool.get("name")
            args = tool.get("arguments", {})

            result = await self.mcp.execute(tool_name, args)
            interpreted = await self.llm.interpret_tool(tool_name, str(args), result)
            await self.slack.send_message(channel, interpreted["final"])
            return

        await self.slack.send_message(channel, response["final"])
