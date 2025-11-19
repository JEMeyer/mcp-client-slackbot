import logging
import os

from pydantic import BaseModel, ValidationError


class AppConfig(BaseModel):
    openai_api_key: str
    openai_base_url: str
    openai_model: str

    slack_bot_token: str
    slack_app_token: str

    sora_api_key: str
    sora_base_url: str

    debug: bool = False


def load_config() -> AppConfig:
    """Load configuration from environment variables."""
    try:
        return AppConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
            openai_model=os.getenv("OPENAI_MODEL", ""),
            slack_bot_token=os.getenv("SLACK_BOT_TOKEN", ""),
            slack_app_token=os.getenv("SLACK_APP_TOKEN", ""),
            sora_api_key=os.getenv("SORA_API_KEY", ""),
            sora_base_url=os.getenv("SORA_BASE_URL", ""),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )
    except ValidationError:
        logging.error("Configuration error", exc_info=True)
        raise
