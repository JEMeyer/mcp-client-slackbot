from typing import Awaitable, Callable, Optional

from slack_sdk.socket_mode.aiohttp import SocketModeClient
from slack_sdk.web.async_client import AsyncWebClient


class SlackManager:
    def __init__(self, bot_token: str, app_token: str):
        self.client = AsyncWebClient(token=bot_token)
        self.socket = SocketModeClient(
            app_token=app_token,
            web_client=self.client,
        )
        self.on_message_handler: Callable[[dict], Awaitable[None]] = self._noop_handler
        self.on_home_opened_handler: Callable[[dict], Awaitable[None]] = (
            self._noop_handler
        )
        self.bot_user_id: Optional[str] = None

    async def _noop_handler(self, event: dict) -> None:  # default no-op handler
        return None

    async def start(self):
        self.socket.socket_mode_request_listeners.append(self._handle_event)
        await self._ensure_bot_user_id()
        await self.socket.connect()

    async def _ensure_bot_user_id(self):
        if self.bot_user_id:
            return
        auth = await self.client.auth_test()
        self.bot_user_id = auth.get("user_id")

    async def _handle_event(self, client, req):
        if req.type != "events_api":
            return

        event = req.payload.get("event", {})
        event_type = event.get("type")

        if event_type == "message" and "bot_id" not in event:
            await self.on_message_handler(event)
        elif event_type == "app_home_opened":
            await self.on_home_opened_handler(event)

        await client.ack(req)

    async def send_message(self, channel: str, text: str, thread_ts: str | None = None):
        payload = {"channel": channel, "text": text}
        if thread_ts:
            payload["thread_ts"] = thread_ts
        await self.client.chat_postMessage(**payload)

    async def publish_home_view(self, user_id: str, blocks: list[dict]):
        await self.client.views_publish(
            user_id=user_id, view={"type": "home", "blocks": blocks}
        )
