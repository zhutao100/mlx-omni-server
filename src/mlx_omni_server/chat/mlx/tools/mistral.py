import json
import uuid
from typing import List, Optional

from mlx_lm.tokenizer_utils import TokenizerWrapper

from mlx_omni_server.chat.mlx.tools.tool_parser import GenericToolParser

from ...schema import ChatMessage, FunctionCall, Role, Tool, ToolCall
from .chat_tokenizer import ChatTokenizer


class MistralChatTokenizer(ChatTokenizer):
    """Tools handler for Llama models."""

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(tokenizer)
        self.tool_parser = GenericToolParser(tool_call_start_token="[TOOL_CALLS]", tool_call_end_token="")

    def decode_stream(self, delta_text: str, tools: list[Tool] | None = None) -> Optional[ChatMessage]:
        return ChatMessage(role=Role.ASSISTANT, content=delta_text)

    def decode(self, text: str, tools: list[Tool] | None = None) -> Optional[ChatMessage]:
        """Parse tool calls from model output.

        The model outputs function calls in the format:
        [TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"location": "Boston, MA"}},
                     {"name": "get_current_weather", "arguments": {"location": "Boston, MA"}}]

        Args:
            text: The model output text containing tool calls

        Returns:
            ChatMessage: A message containing the parsed tool calls
        """
        # Look for JSON patterns in the text
        tool_calls = []

        if text.startswith(self.tool_parser.tool_call_start_token):
            try:
                # Extract the JSON array from between square brackets after [TOOL_CALLS]
                json_str = text[len(self.tool_parser.tool_call_start_token):].strip()
                if json_str.startswith("[") and json_str.endswith("]"):
                    json_str = json_str.strip("[]").strip()
                    # Try to parse as array first
                    try:
                        tool_data = json.loads(f"[{json_str}]")
                    except json.JSONDecodeError:
                        # If array parsing fails, try single object
                        tool_data = json.loads(json_str)

                    # Handle both single object and array of objects
                    if isinstance(tool_data, dict):
                        tool_data = [tool_data]
                    elif not isinstance(tool_data, list):
                        raise ValueError(
                            "Invalid tool call format: expected dict or list"
                        )

                    for call in tool_data:
                        if not isinstance(call, dict) or "name" not in call:
                            continue

                        # Get arguments and ensure they're a JSON string
                        args = call.get("arguments", call.get("parameters", {}))
                        if isinstance(args, str):
                            # Already a JSON string
                            arguments = args
                        else:
                            # Convert dict to JSON string
                            arguments = json.dumps(args)

                        tool_calls.append(
                            ToolCall(
                                id=f"call_{uuid.uuid4().hex[:8]}",
                                function=FunctionCall(
                                    name=call["name"],
                                    arguments=arguments,
                                ),
                            )
                        )
                else:
                    # Invalid format, return original text
                    return ChatMessage(
                        role=Role.ASSISTANT,
                        content=text,
                        tool_calls=None,
                    )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Error parsing tool call: {e}")
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content=text,
                    tool_calls=None,
                )

        return ChatMessage(
            role=Role.ASSISTANT,
            content=None if tool_calls else text,
            tool_calls=tool_calls if tool_calls else None,
        )
