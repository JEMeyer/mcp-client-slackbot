import traceback
from typing import Any, Awaitable, Callable, Dict


class MCPManager:
    """Tool registry + safe execution layer."""

    def __init__(self):
        self.tools: Dict[str, Callable[..., Awaitable[Any]]] = {}

    def register(self, name: str, fn: Callable[..., Awaitable[Any]]):
        self.tools[name] = fn

    async def execute(self, name: str, args: Dict[str, Any]) -> str:
        if name not in self.tools:
            return f"Unknown tool '{name}'"

        try:
            result = await self.tools[name](**args)
            return str(result)
        except Exception:
            return traceback.format_exc()
