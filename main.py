import asyncio
import logging

from aiohttp import web

from src.core.config import load_config
from src.core.health import HealthMonitor
from src.core.health_server import create_health_app
from src.core.orchestrator import Orchestrator
from src.services.llm_client import LLMClient
from src.services.mcp_manager import MCPManager
from src.services.slack_manager import SlackManager
from src.services.sora_client import SoraClient


async def main():
    logging.basicConfig(level=logging.INFO)

    cfg = load_config()
    health = HealthMonitor()

    llm = LLMClient(cfg.openai_api_key, cfg.openai_base_url, cfg.openai_model)
    slack = SlackManager(cfg.slack_bot_token, cfg.slack_app_token)
    mcp = MCPManager()
    sora = SoraClient(cfg.sora_api_key, cfg.sora_base_url)

    orchestrator = Orchestrator(slack, llm, mcp, sora, health)

    # aiohttp health server
    app = create_health_app(health)
    runner = web.AppRunner(app)
    await runner.setup()
    health_site = web.TCPSite(runner, "0.0.0.0", 8080)

    await asyncio.gather(
        orchestrator.start(),
        health_site.start(),
    )


if __name__ == "__main__":
    asyncio.run(main())
