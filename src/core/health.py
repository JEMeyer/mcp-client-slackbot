import asyncio
from typing import Dict


class HealthMonitor:
    """
    Tracks component health states.
    """

    def __init__(self):
        self.components: Dict[str, str] = {
            "slack": "unknown",
            "llm": "unknown",
            "mcp": "ok",
            "sora": "unknown",
        }
        self.lock = asyncio.Lock()

    async def set_status(self, component: str, status: str):
        async with self.lock:
            self.components[component] = status

    async def is_healthy(self) -> bool:
        async with self.lock:
            return all(v == "ok" for v in self.components.values())

    async def report(self) -> Dict[str, str]:
        async with self.lock:
            return dict(self.components)
