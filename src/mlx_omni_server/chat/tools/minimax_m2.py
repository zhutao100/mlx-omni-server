import json
import re
import uuid
from typing import Tuple

import regex
from mlx_lm.tokenizer_utils import TokenizerWrapper
from rich.markup import escape

from ...utils.logger import logger
from ..schema import FunctionCall, Tool, ToolCall, ToolType
from .chat_tokenizer import ToolParsingChatTokenizer
from .tool_parser import GenericToolParser, _scan_and_parse_tool_calls


class MinimaxM2ToolParser(GenericToolParser):
    """Tool parser for Minimax M2 tool-call format (<minimax:tool_call> / <invoke>)."""

    def __init__(
        self,
        tool_call_start_token: str = "<minimax:tool_call>",
        tool_call_end_token: str = "</minimax:tool_call>",
    ):
        super().__init__(
            tool_call_start_token=tool_call_start_token,
            tool_call_end_token=tool_call_end_token,
        )
        self.strict = False
        self.invoke_end_token: str = "</invoke>"
        self.parameter_prefix: str = "<parameter"
        self.parameter_end_token: str = "</parameter>"

        self.invoke_start_pattern = re.compile(r"<invoke\s+name=(['\"])([^'\"]+)\1\s*>", re.DOTALL)
        self.param_start_pattern = re.compile(
            r"<parameter\s+name=(['\"])([^'\"]+)\1\s*>", re.DOTALL
        )

    def update_tool_start_pattern(self, tools: list[Tool] | None):
        """Update the potential tool start pattern based on available tools."""
        self.tool_start_max_prefix_len = len(self.tool_call_start_token)
        pattern = rf"(?:(?<=^)|(?<=\n))[ \t]*({re.escape(self.tool_call_start_token)})"
        self.tool_start_pattern = regex.compile(pattern, re.DOTALL)

    def _find_param_end(self, text: str, start_pos: int, param_name: str) -> int:
        """Find the end of a parameter value, skipping fake/nested </parameter> tokens."""
        search_pos = start_pos
        while True:
            cand = text.find(self.parameter_end_token, search_pos)
            if cand == -1:
                if self.strict:
                    raise ValueError(
                        f"Missing {self.parameter_end_token} for parameter '{param_name}'"
                    )
                invoke_end_pos = text.find(self.invoke_end_token, start_pos)
                if invoke_end_pos != -1:
                    return invoke_end_pos
                wrapper_end_pos = text.find(self.tool_call_end_token, start_pos)
                return wrapper_end_pos if wrapper_end_pos != -1 else len(text)

            after = text[cand + len(self.parameter_end_token) :].lstrip()
            if (
                after.startswith(self.parameter_prefix)
                or after.startswith(self.invoke_end_token)
                or after.startswith(self.tool_call_end_token)
                or after == ""
            ):
                return cand

            search_pos = cand + len(self.parameter_end_token)

    def parse_tool_call_block(self, text: str, tools: list[Tool] | None) -> ToolCall | None:
        """Parse a single <minimax:tool_call>...</minimax:tool_call> block into ToolCall."""
        logger.debug(escape("Parsing tool call block: %s"), text)

        stripped = text.lstrip()
        if stripped.startswith(self.tool_call_start_token):
            inner = stripped[len(self.tool_call_start_token) :]
        else:
            if self.strict:
                raise ValueError(f"Missing {self.tool_call_start_token} in block")
            inner = text

        # Remove trailing wrapper end token if present.
        end_idx = inner.rfind(self.tool_call_end_token)
        if end_idx != -1:
            inner = inner[:end_idx]

        invoke_match = self.invoke_start_pattern.search(inner)
        if not invoke_match:
            if self.strict:
                raise ValueError('Missing <invoke name="..."> in block')
            return None

        func_name = invoke_match.group(2).strip()
        if not func_name:
            if self.strict:
                raise ValueError("Empty tool name in <invoke>")
            return None

        if tools and func_name not in {t.function.name for t in tools}:
            logger.warning(f"Tool '{func_name}' is not defined in the tools list.")
            return None

        invoke_body_start = invoke_match.end()
        invoke_end_idx = inner.find(self.invoke_end_token, invoke_body_start)
        if invoke_end_idx == -1:
            if self.strict:
                raise ValueError(f"Missing {self.invoke_end_token} for tool '{func_name}'")
            invoke_end_idx = len(inner)

        second_invoke = self.invoke_start_pattern.search(inner, invoke_match.end())
        if second_invoke and second_invoke.start() < invoke_end_idx:
            if self.strict:
                raise ValueError("Multiple <invoke> blocks found in a single tool-call wrapper")
            return None

        invoke_body = inner[invoke_body_start:invoke_end_idx]

        args: dict[str, object] = {}
        pos = 0
        param_config = self._get_arguments_config(func_name, tools)

        while (m := self.param_start_pattern.search(invoke_body, pos)) is not None:
            name = m.group(2).strip()
            value_start = m.end()
            value_end = self._find_param_end(invoke_body, value_start, name)
            raw_value = invoke_body[value_start:value_end].strip()
            args[name] = self._convert_param_value(raw_value, name, param_config, func_name)
            pos = value_end + len(self.parameter_end_token)

        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:24]}",
            type=ToolType.FUNCTION,
            function=FunctionCall(
                name=func_name,
                arguments=json.dumps(args, ensure_ascii=False),
            ),
        )

    def extract_tool_calls(
        self, model_output: str, tools: list[Tool] | None = None
    ) -> Tuple[str, list[ToolCall] | None]:
        """Extract tool calls from model output."""

        def find_start(text: str, pos: int) -> tuple[int, int] | None:
            start_idx = text.find(self.tool_call_start_token, pos)
            if start_idx == -1:
                return None
            return start_idx, start_idx + len(self.tool_call_start_token)

        def find_end(text: str, start_idx: int, match_end_idx: int) -> int:
            end_idx = text.find(self.tool_call_end_token, start_idx)
            if end_idx != -1:
                return end_idx + len(self.tool_call_end_token)

            next_start_idx = match_end_idx + 1
            next_block = text.find(self.tool_call_start_token, next_start_idx)
            return next_block if next_block != -1 else len(text)

        rest_text, results = _scan_and_parse_tool_calls(
            model_output,
            tools=tools,
            strict=self.strict,
            find_start=find_start,
            find_end=find_end,
            parse_block=self.parse_tool_call_block,
        )
        logger.debug(escape("Extracted tool calls %s"), results)
        logger.debug(escape("Remaining text: %s"), rest_text)
        return rest_text, results


class MinimaxM2ChatTokenizer(ToolParsingChatTokenizer):
    """Tools handler for Minimax M2 models with XML tool parsing support."""

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(tokenizer)
        self.tool_parser = MinimaxM2ToolParser()
