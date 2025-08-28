from abc import ABC, abstractmethod
import json
import logging
from typing import Dict

from mlx_lm.tokenizer_utils import TokenizerWrapper

from mlx_omni_server.chat.mlx.tools.tool_parser import BaseToolParser

from ...schema import ChatMessage, Role, Tool, ToolChoice, ToolChoiceType


class ChatTokenizer(ABC):
    """Base class for tools handlers."""

    start_tool_calls: str
    end_tool_calls: str

    def __init__(self, tokenizer: TokenizerWrapper):
        self.tokenizer = tokenizer

    def _ensure_dict_arguments(self, tools: list[Dict]) -> list[Dict]:
        """Ensure that all tool arguments are in dict format rather than JSON strings.

        This prevents unsafe JSON parsing in Jinja2 templates.
        """
        if not tools:
            return tools

        processed_tools = []
        for tool in tools:
            processed_tool = tool.copy()
            if 'function' in processed_tool and 'arguments' in processed_tool['function']:
                args = processed_tool['function']['arguments']
                # If arguments is a JSON string, parse it to dict
                if isinstance(args, str):
                    try:
                        processed_tool['function']['arguments'] = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        # If parsing fails, leave as is but log warning
                        logging.warning(f"Failed to parse tool arguments as JSON: {args[:100]}...")
                # If arguments is already a dict, leave as is
            processed_tools.append(processed_tool)
        return processed_tools

    def encode(
        self,
        messages: list[ChatMessage],
        tools: list[Tool] | None = None,
        tool_choice: ToolChoiceType | None = None,
        **kwargs,
    ) -> str:
        """Encode tools and conversation into a prompt string.

        This is a common implementation that uses the tokenizer's chat template.
        Subclasses can override this if they need different behavior.
        """
        schema_tools = None
        if tools:
            # Convert tools to schema format and ensure arguments are dicts
            schema_tools = [tool.model_dump(exclude_none=True) for tool in tools]
            schema_tools = self._ensure_dict_arguments(schema_tools)

        should_prefill = messages[-1].role == Role.ASSISTANT

        conversation = []
        for message in messages:
            msg_dict = message.model_dump(exclude_none=True)
            if isinstance(msg_dict.get("content"), list):
                msg_dict["content"] = "\n\n".join(
                    item["text"]
                    for item in msg_dict["content"]
                    if item.get("type") == "text"
                )

            # Process tool calls in assistant messages to ensure arguments are dicts
            if msg_dict.get("role") == "assistant" and "tool_calls" in msg_dict:
                for tool_call in msg_dict["tool_calls"]:
                    if "function" in tool_call and "arguments" in tool_call["function"]:
                        args = tool_call["function"]["arguments"]
                        if isinstance(args, str):
                            try:
                                tool_call["function"]["arguments"] = json.loads(args)
                            except (json.JSONDecodeError, TypeError):
                                logging.warning(f"Failed to parse tool call arguments as JSON: {args[:100]}...")

            conversation.append(msg_dict)

        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise TypeError(
                f"self.tokenizer.apply_chat_template is not callable (got type: {type(apply_chat_template)}). "
                "Please check the TokenizerWrapper implementation and ensure it provides a callable 'apply_chat_template' method."
            )
        if should_prefill:
            prompt = apply_chat_template(
                conversation=conversation,
                tools=schema_tools,
                tokenize=False,
                continue_final_message=True,
                **kwargs,
            )
        else:
            prompt = apply_chat_template(
                conversation=conversation,
                tools=schema_tools,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )

        if not isinstance(prompt, str):
            prompt = str(prompt)
        if tools:
            if (
                isinstance(tool_choice, ToolChoice)
                and tool_choice == ToolChoice.REQUIRED
            ):
                prompt += self.start_tool_calls

        return prompt

    @abstractmethod
    def decode_stream(self, delta_text: str, tools: list[Tool] | None = None) -> ChatMessage | None:
        """Parse tool calls from model output."""
        pass

    @abstractmethod
    def decode(self, text: str, tools: list[Tool] | None = None) -> ChatMessage | None:
        """Parse tool calls from model output."""
        pass

    def parse_buffer(self, tools: list[Tool] | None = None) -> ChatMessage | None:
        """Parse any buffered text into a ChatMessage."""
        # This method can be overridden by subclasses if they need to handle buffering
        return None


class ToolParsingChatTokenizer(ChatTokenizer):
    """Tools handler for ToolParsing models with XML tool parsing support."""

    tool_parser: BaseToolParser

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(tokenizer)
        self.start_tool_calls = ""
        self.end_tool_calls = ""
        self.pre_fill_tools_prompt = ""
        self.buffer = ""
        self.left_bracket_pos = -1  # Position of the first '<' in the buffer

    def decode_stream(
        self, delta_text: str, tools: list[Tool] | None = None
    ) -> ChatMessage | None:
        """Parse tool calls from model output in streaming mode."""
        self.buffer += delta_text

        skip_delta = False
        # Simple approach: stop streaming as soon as we see < character in buffer
        if self.left_bracket_pos < 0:
            self.left_bracket_pos = self.buffer.find("<")
            if self.left_bracket_pos >= 0:
                # Calculate what part of this segment comes before the <
                text_before_segment = (
                    self.buffer[: -len(delta_text)]
                    if len(delta_text) <= len(self.buffer)
                    else ""
                )

                if self.left_bracket_pos >= len(text_before_segment):
                    # The < is in this segment
                    chars_before_bracket = self.left_bracket_pos - len(
                        text_before_segment
                    )
                    delta_text = delta_text[:chars_before_bracket]
                else:
                    # The < was in previous segments, don't send anything
                    delta_text = ""
                    skip_delta = True
            else:
                # No < found yet, send the segment
                pass
        else:
            # Already detected <, don't send anything more
            delta_text = ""
            skip_delta = True

        if not skip_delta:
            return ChatMessage(
                role=Role.ASSISTANT,
                content=delta_text,
            )

    def parse_buffer(self, tools: list[Tool] | None = None) -> ChatMessage | None:
        """Process the buffer to extract complete tool calls."""
        if self.left_bracket_pos >= 0:
            # We have seen a '<', so we should parse the buffer for tool calls
            text = self.buffer[self.left_bracket_pos:]
            self.buffer = ""
            self.left_bracket_pos = -1
            return self.decode(text, tools)
        else:
            logging.warning("No matched tool calls in buffer, sending as content.")
            return None
        

    def decode(self, text: str, tools: list[Tool] | None = None) -> ChatMessage | None:
        """Parse tool calls from model output in non-streaming mode."""
        # Use the ToolParsing tool parser to extract tool calls from XML format
        content, tool_calls = self.tool_parser.extract_tool_calls(text, tools)

        return ChatMessage(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        )
