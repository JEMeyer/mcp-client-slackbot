import asyncio
import logging
from typing import List

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.core.normalization import normalize_output


class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_retries = 2

    async def chat(self, messages: List[ChatCompletionMessageParam]):
        """Return normalized, safe output."""
        raw = await self._raw_chat(messages)
        normalized = normalize_output(raw)
        return normalized

    async def _raw_chat(self, messages):
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                logging.error(f"LLM error: {e}")

                if attempt == self.max_retries:
                    raise

                await asyncio.sleep(2**attempt)

    async def interpret_tool(self, tool_name: str, args: str, result: str):
        messages = [
            {
                "role": "system",
                "content": ("You are a helpful interpreter of tool output."),
            },
            {
                "role": "user",
                "content": (
                    f"I used tool '{tool_name}' with args: {args}\n"
                    f"Tool returned:\n{result}\n"
                    "Interpret this clearly."
                ),
            },
        ]
        return await self.chat(messages)
