import json
import uuid

from mlx_lm.tokenizer_utils import TokenizerWrapper
from rich.markup import escape

from ...utils.logger import logger
from ..schema import (
    ChatMessage,
    FunctionCall,
    Role,
    SpecificToolChoice,
    Tool,
    ToolCall,
    ToolChoiceType,
)
from .chat_tokenizer import ChatTokenizer
from .tool_parser import GenericToolParser


class JsonToolPrefillChatTokenizer(ChatTokenizer):
    """ChatTokenizer that supports JSON tool-call prefill for SpecificToolChoice."""

    def __init__(
        self,
        tokenizer: TokenizerWrapper,
        *,
        tool_call_start_token: str,
        tool_call_end_token: str,
    ) -> None:
        super().__init__(tokenizer)
        self.strict_mode = False
        self.pre_fill_tools_prompt = ""
        self.tool_parser = GenericToolParser(
            tool_call_start_token=tool_call_start_token,
            tool_call_end_token=tool_call_end_token,
        )

    def encode(
        self,
        messages: list[ChatMessage],
        tools: list[Tool] | None = None,
        tool_choice: ToolChoiceType | None = None,
        **kwargs,
    ) -> str:
        self.pre_fill_tools_prompt = ""
        prompt = super().encode(messages, tools, tool_choice, **kwargs)

        if tools and isinstance(tool_choice, SpecificToolChoice):
            function_name = tool_choice.function["name"]
            self.pre_fill_tools_prompt = (
                f"{self.tool_parser.tool_call_start_token}"
                f'{{"name": "{function_name}", "arguments":'
            )

        return prompt + self.pre_fill_tools_prompt

    def decode_stream(self, delta_text: str, tools: list[Tool] | None = None) -> ChatMessage | None:
        return ChatMessage(role=Role.ASSISTANT, content=delta_text)

    def _parse_strict_tools(self, text: str) -> list[ToolCall] | None:
        logger.debug(f"_parse_strict_tools: {escape(text)}")

        stripped = text.lstrip()
        leading = len(text) - len(stripped)
        if not stripped.startswith(self.tool_parser.tool_call_start_token):
            return None

        start = leading + len(self.tool_parser.tool_call_start_token)
        if self.tool_parser.tool_call_end_token:
            end_in_stripped = stripped.find(self.tool_parser.tool_call_end_token)
            if end_in_stripped < 0:
                return None
            end = leading + end_in_stripped
            json_str = text[start:end].strip()
        else:
            json_str = text[start:].strip()

        try:
            tool_data = json.loads(json_str)
            if not isinstance(tool_data, dict) or "name" not in tool_data:
                return None

            args = tool_data.get("arguments", tool_data.get("parameters", {}))
            arguments = args if isinstance(args, str) else json.dumps(args)

            return [
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    function=FunctionCall(
                        name=tool_data["name"],
                        arguments=arguments,
                    ),
                )
            ]
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error("Error parsing tool call: %s", exc)
            return None

    def decode(self, text: str, tools: list[Tool] | None = None) -> ChatMessage:
        response = self.pre_fill_tools_prompt + text
        self.pre_fill_tools_prompt = ""

        if self.strict_mode:
            tool_calls = self._parse_strict_tools(response)
        else:
            _, tool_calls = self.tool_parser.extract_tool_calls(response)

        return ChatMessage(
            role=Role.ASSISTANT,
            content=None if tool_calls else text,
            tool_calls=tool_calls,
        )
