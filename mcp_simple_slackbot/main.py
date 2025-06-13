"""
MCP Bot - Central Hub

This is the main entry point for the MCP bot system. It orchestrates the
initialization and coordination of all components including configuration,
MCP servers, LLM client, and platform-specific bots (Slack, Discord, etc.).
"""

import asyncio
import logging
import sys

from config import Configuration
from llm import LLMClient
from mcp_manager import MCPManager, Server
from slack_manager import SlackBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class MCPBotHub:
    """Central hub that manages all bot components."""

    def __init__(self):
        self.config = Configuration()
        self.mcp_manager = None
        self.llm_client = None
        self.slack_bot = None
        # Future: self.discord_bot = None

    async def initialize(self):
        """Initialize all components."""
        logging.info("Initializing MCP Bot Hub...")

        # Validate configuration
        self._validate_config()

        # Initialize LLM client
        self.llm_client = LLMClient(
            api_key=str(self.config.openai_api_key),
            model=self.config.llm_model,
            base_url=self.config.openai_base_url
        )
        logging.info("LLM client initialized")

        # Initialize MCP servers
        await self._initialize_mcp_servers()

        # Initialize platform-specific bots
        await self._initialize_bots()

        logging.info("MCP Bot Hub initialization complete")

    def _validate_config(self):
        """Validate all required configuration."""
        try:
            self.config.validate_llm_config()
            self.config.validate_slack_config()
            self.config.validate_video_config()  # This just warns if missing
        except ValueError as e:
            logging.error(f"Configuration validation failed: {e}")
            sys.exit(1)

    async def _initialize_mcp_servers(self):
        """Initialize MCP servers and manager."""
        try:
            server_config = self.config.load_config("servers_config.json")
            servers = [
                Server(name, srv_config)
                for name, srv_config in server_config["mcpServers"].items()
            ]

            self.mcp_manager = MCPManager(servers)
            await self.mcp_manager.initialize_all_servers()

            tools = self.mcp_manager.get_all_tools()
            logging.info(f"MCP Manager initialized with {len(tools)} total tools")

        except Exception as e:
            logging.error(f"Failed to initialize MCP servers: {e}")
            raise

    async def _initialize_bots(self):
        """Initialize platform-specific bots."""
        # Initialize Slack bot
        if self.config.slack_bot_token and self.config.slack_app_token and self.mcp_manager and self.llm_client:
            self.slack_bot = SlackBot(
                slack_bot_token=self.config.slack_bot_token,
                slack_app_token=self.config.slack_app_token,
                mcp_manager=self.mcp_manager,
                llm_client=self.llm_client,
                has_video_config=self.config.has_video_config()
            )
            logging.info("Slack bot initialized")
        else:
            logging.warning("Slack bot not initialized - missing tokens")

        # Future: Initialize Discord bot
        # if self.config.discord_token:
        #     self.discord_bot = DiscordBot(...)
        #     logging.info("Discord bot initialized")

    async def start(self):
        """Start all bots."""
        logging.info("Starting all bots...")

        # Start Slack bot
        if self.slack_bot:
            await self.slack_bot.start()
            logging.info("Slack bot started")

        # Future: Start Discord bot
        # if self.discord_bot:
        #     await self.discord_bot.start()
        #     logging.info("Discord bot started")

        # Keep the main process alive
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logging.info("Received shutdown signal...")
            await self.cleanup()

    async def cleanup(self):
        """Clean up all resources."""
        logging.info("Cleaning up resources...")

        # Cleanup Slack bot
        if self.slack_bot:
            await self.slack_bot.cleanup()
            logging.info("Slack bot cleaned up")

        # Future: Cleanup Discord bot
        # if self.discord_bot:
        #     await self.discord_bot.cleanup()
        #     logging.info("Discord bot cleaned up")

        # Cleanup MCP servers
        if self.mcp_manager:
            await self.mcp_manager.cleanup_all_servers()
            logging.info("MCP servers cleaned up")

        logging.info("Cleanup complete")


async def main():
    """Main entry point."""
    hub = MCPBotHub()

    try:
        await hub.initialize()
        await hub.start()
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        await hub.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
