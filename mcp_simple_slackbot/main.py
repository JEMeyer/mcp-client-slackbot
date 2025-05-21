import asyncio
import json
import logging
import os
import re
import shutil
from contextlib import AsyncExitStack
from typing import Any, Dict, List

import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Configuration:
    """Manages configuration and environment variables for the MCP Slackbot."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.slack_bot_token = os.getenv("SLACK_BOT_TOKEN")
        self.slack_app_token = os.getenv("SLACK_APP_TOKEN")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4-turbo")

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

class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]}
            if self.config.get("env")
            else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)
                return result
            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")

class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: Dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: Dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""

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

class SlackMCPBot:
    """Manages the Slack bot integration with MCP servers."""

    def __init__(
        self,
        slack_bot_token: str,
        slack_app_token: str,
        servers: List[Server],
        llm_client: LLMClient,
    ) -> None:
        self.app = AsyncApp(token=slack_bot_token)
        # Create a socket mode handler with the app token
        self.socket_mode_handler = AsyncSocketModeHandler(self.app, slack_app_token)

        self.client = AsyncWebClient(token=slack_bot_token)
        self.servers = servers
        self.llm_client = llm_client
        self.conversations = {}  # Store conversation context per channel
        self.tools = []

        # Set up event handlers
        self.app.event("app_mention")(self.handle_mention)
        self.app.message()(self.handle_message)
        self.app.event("app_home_opened")(self.handle_home_opened)

    async def initialize_servers(self) -> None:
        """Initialize all MCP servers and discover tools."""
        for server in self.servers:
            try:
                await server.initialize()
                server_tools = await server.list_tools()
                self.tools.extend(server_tools)
                logging.info(
                    f"Initialized server {server.name} with {len(server_tools)} tools"
                )
            except Exception as e:
                logging.error(f"Failed to initialize server {server.name}: {e}")

    async def initialize_bot_info(self) -> None:
        """Get the bot's ID and other info."""
        try:
            auth_info = await self.client.auth_test()
            self.bot_id = auth_info["user_id"]
            logging.info(f"Bot initialized with ID: {self.bot_id}")
        except Exception as e:
            logging.error(f"Failed to get bot info: {e}")
            self.bot_id = None

    async def handle_mention(self, event, say):
        """Handle mentions of the bot in channels."""
        await self._process_message(event, say)

    async def handle_message(self, message, say):
        """Handle direct messages to the bot."""
        # Only process direct messages
        if message.get("channel_type") == "im" and not message.get("subtype"):
            await self._process_message(message, say)

    async def handle_home_opened(self, event, client):
        """Handle when a user opens the App Home tab."""
        user_id = event["user"]

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Welcome to MCP Assistant!"},
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "I'm an AI assistant with access to tools and resources "
                        "through the Model Context Protocol."
                    ),
                },
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*Available Tools:*"},
            },
        ]

        # Add tools
        for tool in self.tools:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"• *{tool.name}*: {tool.description}",
                    },
                }
            )

        # Add usage section
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "*How to Use:*\n• Send me a direct message\n"
                        "• Mention me in a channel with @MCP Assistant"
                    ),
                },
            }
        )

        try:
            await client.views_publish(
                user_id=user_id, view={"type": "home", "blocks": blocks}
            )
        except Exception as e:
            logging.error(f"Error publishing home view: {e}")

    def add_response_to_conversation(self, channel: str, response: str) -> dict:
        if channel not in self.conversations:
            self.conversations[channel] = []

        # Append the response as a message dict and return
        responseDict = {"role": "assistant", "content": response}
        self.conversations[channel]["messages"].append(responseDict)
        return responseDict

    async def _process_message(self, event, say):
        """Process incoming messages and generate responses."""
        channel = event["channel"]
        user_id = event.get("user")

        # Skip messages from the bot itself
        if user_id == getattr(self, "bot_id", None):
            return

        # Get text and remove bot mention if present
        text = event.get("text", "")
        if hasattr(self, "bot_id") and self.bot_id:
            text = text.replace(f"<@{self.bot_id}>", "").strip()

        thread_ts = event.get("thread_ts", event.get("ts"))

        # Get or create conversation context
        if channel not in self.conversations:
            self.conversations[channel] = {"messages": []}

        try:
            # Create system message with tool descriptions
            tools_text = "\n".join([tool.format_for_llm() for tool in self.tools])
            system_message = {
                "role": "system",
                "content": (
                    f"""You are a helpful assistant with access to the following tools:

{tools_text}

When you need to use a tool, you MUST format your response exactly like this:
[TOOL] tool_name
{{"param1": "value1", "param2": "value2"}}

Make sure to include both the tool name AND the JSON arguments.
Never leave out the JSON arguments.

After receiving tool results, interpret them for the user in a helpful way, which may include additional tool calls if necessary.
You can continue to use tools as many times as needed to fulfill the request, without asking for permission to continue. You are in a loop that will stop after 10 tool calls maximum.
"""
                ),
            }

            # Add user message to history
            self.conversations[channel]["messages"].append(
                {"role": "user", "content": text}
            )

            # Set up messages for LLM
            messages = [system_message]

            # Add conversation history (last 5 messages)
            if "messages" in self.conversations[channel]:
                messages.extend(self.conversations[channel]["messages"][-5:])

            response = await self.llm_client._get_openai_response(messages)
            messages += self.add_response_to_conversation(channel, response)

            tool_call_count = 0
            while "[TOOL]" in response and tool_call_count < 10:
                tool_call_count += 1
                response = await self._process_tool_call(response, channel)
                # After tool call, we need to consult the LLM again.
                messages += self.add_response_to_conversation(channel, response)
                # Get LLM response with tool result in history
                response = await self.llm_client._get_openai_response(messages)
                messages += self.add_response_to_conversation(channel, response)

            # Send the response to the user
            await say(text=response, channel=channel, thread_ts=thread_ts)

        except Exception as e:
            error_message = f"I'm sorry, I encountered an error: {str(e)}"
            logging.error(f"Error processing message: {e}", exc_info=True)
            await say(text=error_message, channel=channel, thread_ts=thread_ts)

    async def _process_tool_call(self, response: str, channel: str) -> str:
        """Process a tool call from the LLM response."""
        try:
            # Extract tool name and arguments
            tool_parts = response.split("[TOOL]")[1].strip().split("\n", 1)
            tool_name = tool_parts[0].strip()

            # Handle incomplete tool calls
            if len(tool_parts) < 2:
                return (
                    f"I tried to use the tool '{tool_name}', but the request "
                    f"was incomplete. Here's my response without the tool:"
                    f"\n\n{response.split('[TOOL]')[0]}"
                )

            # Parse JSON arguments
            try:
                args_text = tool_parts[1].strip()
                arguments = json.loads(args_text)
            except json.JSONDecodeError:
                return (
                    f"I tried to use the tool '{tool_name}', but the arguments "
                    f"were not properly formatted. Here's my response without "
                    f"the tool:\n\n{response.split('[TOOL]')[0]}"
                )

            # Find the appropriate server for this tool
            for server in self.servers:
                server_tools = [tool.name for tool in await server.list_tools()]
                if tool_name in server_tools:
                    # Execute the tool
                    tool_result = await server.execute_tool(tool_name, arguments)

                    try:
                        # Get interpretation from LLM
                        messages = [
                            {
                                "role": "system",
                                "content": (
                                    "You are a helpful record keeper. When you receive "
                                   " a result from a tool as reported by the user, "
                                    "interpret these results in a clear, helpful way, "
                                    "which may mean no modification to the result."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"I used the tool {tool_name} with arguments "
                                    f"{args_text} and got this result:\n\n"
                                    f"{tool_result}\n\n"
                                    f"Please interpret this result for me."
                                ),
                            },
                        ]

                        interpretation = await self.llm_client._get_openai_response(messages)
                        return interpretation
                    except Exception as e:
                        logging.error(
                            f"Error getting tool result interpretation: {e}",
                            exc_info=True,
                        )
                        # Fallback to basic formatting
                        if isinstance(tool_result, dict):
                            result_text = json.dumps(tool_result, indent=2)
                        else:
                            result_text = str(tool_result)
                        return (
                            f"I used the {tool_name} tool and got these results:"
                            f"\n\n```\n{result_text}\n```"
                        )

            # No server had the tool
            return (
                f"I tried to use the tool '{tool_name}', but it's not available. "
                f"Here's my response without the tool:\n\n{response.split('[TOOL]')[0]}"
            )

        except Exception as e:
            logging.error(f"Error executing tool: {e}", exc_info=True)
            return (
                f"I tried to use a tool, but encountered an error: {str(e)}\n\n"
                f"Here's my response without the tool:\n\n{response.split('[TOOL]')[0]}"
            )

    async def start(self) -> None:
        """Start the Slack bot."""
        await self.initialize_servers()
        await self.initialize_bot_info()
        # Start the socket mode handler
        logging.info("Starting Slack bot...")
        asyncio.create_task(self.socket_mode_handler.start_async())
        logging.info("Slack bot started and waiting for messages")

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self, "socket_mode_handler"):
                await self.socket_mode_handler.close_async()
            logging.info("Slack socket mode handler closed")
        except Exception as e:
            logging.error(f"Error closing socket mode handler: {e}")

        # Clean up servers
        for server in self.servers:
            try:
                await server.cleanup()
                logging.info(f"Server {server.name} cleaned up")
            except Exception as e:
                logging.error(f"Error during cleanup of server {server.name}: {e}")


async def main() -> None:
    """Initialize and run the Slack bot."""
    config = Configuration()

    if not config.slack_bot_token or not config.slack_app_token or not config.openai_api_key:
        raise ValueError(
            "SLACK_BOT_TOKEN, SLACK_APP_TOKEN, and OPENAI_API_KEY must be set in environment variables"
        )

    server_config = config.load_config("servers_config.json")
    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]

    llm_client = LLMClient(config.openai_api_key, config.llm_model, config.openai_base_url)

    slack_bot = SlackMCPBot(
        config.slack_bot_token, config.slack_app_token, servers, llm_client
    )

    try:
        await slack_bot.start()
        # Keep the main task alive until interrupted
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        await slack_bot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
