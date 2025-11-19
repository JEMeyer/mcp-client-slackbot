from typing import Awaitable, Callable

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

    async def _noop_handler(self, event: dict) -> None:  # default no-op handler
        return None

    async def start(self):
        self.socket.socket_mode_request_listeners.append(self._handle_event)
        await self.socket.connect()

    async def _handle_event(self, client, req):
        if req.type != "events_api":
            return

        event = req.payload.get("event", {})
        if event.get("type") == "message" and "bot_id" not in event:
            await self.on_message_handler(event)

        await client.ack(req)

    async def send_message(self, channel: str, text: str):
        await self.client.chat_postMessage(channel=channel, text=text)
