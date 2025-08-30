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
        self.potential_tool_start_pos = -1  # Position of potential tool call start

    def _check_tool_start_token(self, text: str, start_pos: int) -> bool:
        """Check if text at start_pos matches the tool call start token."""
        if not hasattr(self.tool_parser, 'tool_call_start_token') or not self.tool_parser.tool_call_start_token:
            return False

        start_token = self.tool_parser.tool_call_start_token
        if start_pos + len(start_token) > len(text):
            return False

        return text[start_pos:start_pos + len(start_token)] == start_token

    def _find_next_potential_tool_start(self, text: str, search_start: int) -> int:
        """Find the next position that could be the start of a tool call."""
        if not hasattr(self.tool_parser, 'tool_call_start_token') or not self.tool_parser.tool_call_start_token:
            return -1

        start_token = self.tool_parser.tool_call_start_token
        if not start_token:
            return -1

        # Look for the first character of the start token
        first_char = start_token[0]
        pos = text.find(first_char, search_start)

        while pos >= 0:
            # Check if this position actually matches the full start token
            if self._check_tool_start_token(text, pos):
                return pos
            # Continue searching
            pos = text.find(first_char, pos + 1)

        return -1

    def _extract_content_and_buffer_partials(self, text: str) -> tuple[str, str]:
        """
        From a given text, extracts the content to be returned and the part to be buffered.
        The part to be buffered is a suffix of the text that is a prefix of the tool start token.
        """
        if not hasattr(self.tool_parser, 'tool_call_start_token') or not self.tool_parser.tool_call_start_token:
            return text, ""

        tool_token = self.tool_parser.tool_call_start_token
        max_partial_length = len(tool_token) - 1

        if not text:
            return "", ""

        for i in range(1, min(len(text), max_partial_length) + 1):
            suffix = text[-i:]
            if tool_token.startswith(suffix):
                return text[:-i], suffix

        return text, ""

    def _process_buffer_for_tool_start(self) -> ChatMessage | None:
        """
        Processes the buffer to find a tool start.
        If found, returns preceding content and updates buffer state.
        If not found, returns content that is not a partial match and buffers the rest.
        """
        tool_start_pos = self._find_next_potential_tool_start(self.buffer, 0)

        if tool_start_pos != -1:
            # Found a tool start.
            content_before = self.buffer[:tool_start_pos]
            self.buffer = self.buffer[tool_start_pos:]
            self.potential_tool_start_pos = 0
            if content_before:
                return ChatMessage(role=Role.ASSISTANT, content=content_before)
            return None
        else:
            # No full tool start found. Buffer partial matches.
            content_to_return, buffer_to_keep = self._extract_content_and_buffer_partials(
                self.buffer
            )
            self.buffer = buffer_to_keep
            if content_to_return:
                return ChatMessage(role=Role.ASSISTANT, content=content_to_return)
            return None

    def decode_stream(
        self,
        delta_text: str,
        tools: list[Tool] | None = None,
    ) -> ChatMessage | None:
        """Parse tool calls from model output in streaming mode."""
        if not delta_text and not self.buffer:
            return None

        self.buffer += delta_text

        if self.potential_tool_start_pos < 0:
            return self._process_buffer_for_tool_start()
        else:  # We are in a potential tool call.
            if self._check_tool_start_token(self.buffer, self.potential_tool_start_pos):
                # Confirmed tool start. Continue buffering.
                return None
            else:
                # False positive. The potential start was not a real one.
                # Release the first character and re-evaluate the buffer.
                content_to_release = self.buffer[0]
                self.buffer = self.buffer[1:]
                self.potential_tool_start_pos = -1

                # Re-process the modified buffer to find the next tool start
                next_message = self._process_buffer_for_tool_start()

                if next_message and next_message.content:
                    # Combine the released character with content from the next message
                    return ChatMessage(
                        role=Role.ASSISTANT,
                        content=content_to_release + next_message.content,
                    )
                else:
                    # Only the released character is available as content
                    return ChatMessage(role=Role.ASSISTANT, content=content_to_release)

    def parse_buffer(self, tools: list[Tool] | None = None) -> ChatMessage | None:
        """Process the buffer to extract complete tool calls."""
        try:
            if not self.buffer.strip():
                return None

            if self.potential_tool_start_pos >= 0 and self._check_tool_start_token(
                self.buffer, self.potential_tool_start_pos
            ):
                # We have a confirmed tool call.
                text_to_parse = self.buffer[self.potential_tool_start_pos:]
                content_before_tool = self.buffer[:self.potential_tool_start_pos]

                tool_result = self.decode(text_to_parse, tools)

                if tool_result and tool_result.tool_calls:
                    # Prioritize tool calls, content before is ignored as per original logic
                    if tool_result.content == "":
                        tool_result.content = None
                    return tool_result

                # No tool calls were found, so combine all content
                full_content = content_before_tool + (
                    tool_result.content if tool_result else ""
                )
                if full_content.strip():
                    return ChatMessage(role=Role.ASSISTANT, content=full_content)

                return None
            else:
                # No tool call or a false positive, treat buffer as content
                if self.buffer.strip():
                    return ChatMessage(role=Role.ASSISTANT, content=self.buffer)
                return None
        finally:
            # Always reset the buffer and state after parsing.
            self.buffer = ""
            self.potential_tool_start_pos = -1

    def decode(self, text: str, tools: list[Tool] | None = None) -> ChatMessage | None:
        """Parse tool calls from model output in non-streaming mode."""
        # Use the ToolParsing tool parser to extract tool calls from XML format
        content, tool_calls = self.tool_parser.extract_tool_calls(text, tools)

        return ChatMessage(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        )
