import asyncio
import logging
from typing import Dict, List

import httpx


class LLMClient:
    """Client for communicating with LLM APIs."""

    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        """Initialize the LLM client.

        Args:
            api_key: API key for the LLM provider
            model: Model identifier to use
            base_url: Base URL for the LLM provider's API
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = 30.0  # 30 second timeout
        self.max_retries = 2

    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the LLM API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            The LLM's response as a string
        """
        return await self._get_openai_response(messages)

    async def _get_openai_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the OpenAI API."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
        }

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)

                    if response.status_code == 200:
                        response_data = response.json()
                        return response_data["choices"][0]["message"]["content"]
                    else:
                        if attempt == self.max_retries:
                            return (
                                f"Error from API: {response.status_code} - "
                                f"{response.text}"
                            )
                        await asyncio.sleep(2**attempt)  # Exponential backoff
            except Exception as e:
                if attempt == self.max_retries:
                    return f"Failed to get response: {str(e)}"
                await asyncio.sleep(2**attempt)  # Exponential backoff
        return ""

    async def interpret_tool_result(self, tool_name: str, arguments: str, tool_result: str) -> str:
        """Get an interpretation of a tool result from the LLM.

        Args:
            tool_name: Name of the tool that was executed
            arguments: Arguments that were passed to the tool
            tool_result: Result returned by the tool

        Returns:
            LLM's interpretation of the tool result
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful record keeper. When you receive "
                    "a result from a tool as reported by the user, "
                    "interpret these results in a clear, helpful way, "
                    "which may mean no modification to the result."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"I used the tool {tool_name} with arguments "
                    f"{arguments} and got this result:\n\n"
                    f"{tool_result}\n\n"
                    f"Please interpret this result for me."
                ),
            },
        ]

        try:
            return await self.get_response(messages)
        except Exception as e:
            logging.error(f"Error getting tool result interpretation: {e}", exc_info=True)
            raise
