import json
import logging
import os
import re
from typing import Any, Dict

from dotenv import load_dotenv


class Configuration:
    """Manages configuration and environment variables for the MCP bot."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.slack_bot_token = os.getenv("SLACK_BOT_TOKEN")
        self.slack_app_token = os.getenv("SLACK_APP_TOKEN")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4-turbo")
        
        # Azure OpenAI for video generation (Sora)
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        # Read the raw file content
        with open(file_path, "r") as f:
            content = f.read()

        # Find all placeholders like ${VAR_NAME}
        placeholder_pattern = r"\$\{(\w+)\}"

        for match in re.finditer(placeholder_pattern, content):
            var_name = match.group(1)
            env_value = os.getenv(var_name)
            if env_value is None:
                raise RuntimeError(f"Environment variable '{var_name}' not set")
            # Replace all occurrences of this placeholder
            content = content.replace(match.group(0), env_value)

        # Parse JSON
        return json.loads(content)

    def validate_slack_config(self) -> None:
        """Validate that required Slack configuration is present."""
        if not self.slack_bot_token:
            raise ValueError("SLACK_BOT_TOKEN must be set in environment variables")
        if not self.slack_app_token:
            raise ValueError("SLACK_APP_TOKEN must be set in environment variables")

    def validate_llm_config(self) -> None:
        """Validate that required LLM configuration is present."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment variables")

    def validate_video_config(self) -> None:
        """Validate that Azure OpenAI configuration for video generation is present."""
        if not self.azure_openai_endpoint:
            logging.warning("AZURE_OPENAI_ENDPOINT not set - video generation will be disabled")
        if not self.azure_openai_api_key:
            logging.warning("AZURE_OPENAI_API_KEY not set - video generation will be disabled")

    def has_video_config(self) -> bool:
        """Check if video generation configuration is available."""
        return bool(self.azure_openai_endpoint and self.azure_openai_api_key)