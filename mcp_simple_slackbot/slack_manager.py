import asyncio
import json
import logging
from typing import Dict

from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from llm import LLMClient
from mcp_manager import MCPManager
from sora import generate_video


class SlackBot:
    """Manages the Slack bot integration with MCP servers."""

    def __init__(
        self,
        slack_bot_token: str,
        slack_app_token: str,
        mcp_manager: MCPManager,
        llm_client: LLMClient,
        has_video_config: bool = False,
    ) -> None:
        self.app = AsyncApp(token=slack_bot_token)
        # Create a socket mode handler with the app token
        self.socket_mode_handler = AsyncSocketModeHandler(self.app, slack_app_token)

        self.client = AsyncWebClient(token=slack_bot_token)
        self.mcp_manager = mcp_manager
        self.llm_client = llm_client
        self.has_video_config = has_video_config
        self.conversations = {}  # Store conversation context per channel
        self.bot_id = None

        # Set up event handlers
        self.app.event("app_mention")(self.handle_mention)
        self.app.message()(self.handle_message)
        self.app.event("app_home_opened")(self.handle_home_opened)

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
        tools = self.mcp_manager.get_all_tools()

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
        for tool in tools:
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

    def add_response_to_conversation(self, channel: str, response: str) -> Dict[str, str]:
        """Add a response to the conversation history."""
        if channel not in self.conversations:
            self.conversations[channel] = {"messages": []}

        # Append the response as a message dict and return
        response_dict = {"role": "assistant", "content": response}
        self.conversations[channel]["messages"].append(response_dict)
        return response_dict

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

        # Check if the message is asking to make a video
        if text.lower().startswith('make a video of') and self.has_video_config:
            await self._handle_video_request(text, channel, thread_ts, say)
            return

        # Get or create conversation context
        if channel not in self.conversations:
            self.conversations[channel] = {"messages": []}

        try:
            # Create system message with tool descriptions
            tools_text = self.mcp_manager.format_tools_for_llm()
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

            response = await self.llm_client.get_response(messages)
            self.add_response_to_conversation(channel, response)

            tool_call_count = 0
            while "[TOOL]" in response and tool_call_count < 10:
                tool_call_count += 1
                response = await self._process_tool_call(response, channel)
                # After tool call, we need to consult the LLM again.
                self.add_response_to_conversation(channel, response)
                # Get LLM response with tool result in history
                messages = [system_message] + self.conversations[channel]["messages"][-5:]
                response = await self.llm_client.get_response(messages)
                self.add_response_to_conversation(channel, response)

            # Send the response to the user
            await say(text=response, channel=channel, thread_ts=thread_ts)

        except Exception as e:
            error_message = f"I'm sorry, I encountered an error: {str(e)}"
            logging.error(f"Error processing message: {e}", exc_info=True)
            await say(text=error_message, channel=channel, thread_ts=thread_ts)

    async def _handle_video_request(self, text: str, channel: str, thread_ts: str, say):
        """Handle video generation requests."""
        try:
            # Extract the video description
            video_description = text[len('make a video of'):].strip()

            # Call sora.py script
            video = generate_video(video_description)

            # Use the video returned from generate_video
            if video:
                try:
                    # Upload the video to Slack
                    await self.client.files_upload_v2(
                        channel=channel,
                        file=video,
                        title=f"Generated video: {video_description}",
                        thread_ts=thread_ts
                    )
                    return
                except Exception as e:
                    await say(text=f"Error uploading video: {str(e)}", channel=channel, thread_ts=thread_ts)
                    return
            else:
                await say(text="Video generation failed - no video returned", channel=channel, thread_ts=thread_ts)
                return

        except Exception as e:
            await say(text=f"Error processing video request: {str(e)}", channel=channel, thread_ts=thread_ts)
            return

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

            # Execute the tool
            tool_result = None
            try:
                tool_result = await self.mcp_manager.execute_tool(tool_name, arguments)

                # Get interpretation from LLM
                interpretation = await self.llm_client.interpret_tool_result(
                    tool_name, args_text, str(tool_result)
                )
                return interpretation

            except ValueError:
                # Tool not found
                return (
                    f"I tried to use the tool '{tool_name}', but it's not available. "
                    f"Here's my response without the tool:\n\n{response.split('[TOOL]')[0]}"
                )
            except Exception as e:
                logging.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
                # Fallback to basic formatting
                if isinstance(tool_result, dict):
                    result_text = json.dumps(tool_result, indent=2)
                else:
                    result_text = str(tool_result)
                return (
                    f"I used the {tool_name} tool and got these results:"
                    f"\n\n```\n{result_text}\n```"
                )

        except Exception as e:
            logging.error(f"Error processing tool call: {e}", exc_info=True)
            return (
                f"I tried to use a tool, but encountered an error: {str(e)}\n\n"
                f"Here's my response without the tool:\n\n{response.split('[TOOL]')[0]}"
            )

    async def start(self) -> None:
        """Start the Slack bot."""
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
