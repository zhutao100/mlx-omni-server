import json
import re
import uuid
from typing import Pattern, Tuple

import regex
from mlx_lm.tokenizer_utils import TokenizerWrapper
from rich.markup import escape

from ...utils.logger import logger
from ..schema import (
    FunctionCall,
    Tool,
    ToolCall,
    ToolType,
)
from .chat_tokenizer import ToolParsingChatTokenizer
from .tool_parser import GenericToolParser, _scan_and_parse_tool_calls


class Qwen3ToolParser(GenericToolParser):
    """Tool parser for Qwen3's XML format that converts to OpenAI JSON format."""

    def __init__(self, strict: bool = False, tool_call_start_token="<tool_call>", tool_call_end_token="</tool_call>"):
        super().__init__(tool_call_start_token=tool_call_start_token, tool_call_end_token=tool_call_end_token)
        self.strict = strict
        # Sentinel tokens for streaming mode
        self.function_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"

        # Compiled regex patterns for efficiency
        self.func_name_pattern = re.compile(rf"{self.function_prefix}([^>]+)>", re.DOTALL)
        self.param_start_pattern = re.compile(rf"{self.parameter_prefix}([^>]+)>", re.DOTALL)
        self.param_end_pattern = re.compile(rf"{self.parameter_end_token}", re.DOTALL)
        self.tool_call_start_pattern = re.compile(rf"{self.tool_call_start_token}", re.DOTALL)
        self.tool_call_or_function_start_pattern = re.compile(
            rf"{self.tool_call_start_token}|{self.function_prefix}[^>]+>", re.DOTALL
        )

    def update_tool_start_pattern(self, tools: list[Tool] | None):
        """Update the potential tool start pattern based on available tools and model specific patterns.

        This method creates a regex pattern that can match either the standard tool call start token
        or the function prefix, allowing for more flexible tool call detection in Qwen3 models.
        """
        self.tool_start_max_prefix_len = max(
            len(self.tool_call_start_token),
            len(self.function_prefix),
        )
        # Match at the beginning of string or after a newline, allowing indentation (spaces/tabs).
        pattern = rf"(?:(?<=^)|(?<=\n))[ \t]*({re.escape(self.tool_call_start_token)}|{re.escape(self.function_prefix)})"
        self.tool_start_pattern = regex.compile(pattern, re.DOTALL)

    def _find_param_end(self, text: str, start_pos: int, param_name: str) -> int:
        """
        Find the true end of a parameter, skipping fake/nested </parameter> tokens.
        Heuristic: a </parameter> is valid only if followed by:
          - another <parameter=...>, or
          - </function>, or
          - </tool_call>, or
          - end-of-text.
        Otherwise, treat it as literal text.
        """
        search_pos = start_pos
        while True:
            match = self.param_end_pattern.search(text, search_pos)
            if not match:
                if self.strict:
                    raise ValueError(f"Missing {self.parameter_end_token} for '{param_name}'")
                # fallback: parameter ends before </function> or end of block
                func_end_pos = text.find(self.function_end_token, start_pos)
                return func_end_pos if func_end_pos != -1 else len(text)

            after = text[match.end():].lstrip()
            if (after.startswith(self.parameter_prefix)
                or after.startswith(self.function_end_token)
                or after.startswith(self.tool_call_end_token)
                    or after == ""):
                return match.start()

            # otherwise treat as literal and keep searching
            search_pos = match.end()

    def _find_block_end(
        self,
        model_output: str,
        start_idx: int,
        next_search_start_idx: int,
        next_block_pattern: Pattern
    ) -> int:
        """
        Heuristically find the end of a <tool_call> block.
        - If </tool_call> appears before the next <tool_call>/<function>, assume valid end.
        - If missing, end before next block start or at end-of-text.
        - If ambiguous (overlap), consume until end-of-text as fallback.
        """
        next_block_match = next_block_pattern.search(model_output, next_search_start_idx)
        close_idx = model_output.find(self.tool_call_end_token, start_idx)

        if close_idx == -1:
            return (next_search_start_idx + next_block_match.start()
                    if next_block_match else len(model_output))

        potential_end = close_idx + len(self.tool_call_end_token)
        if not next_block_match:
            return potential_end

        next_block_start = next_search_start_idx + next_block_match.start()
        return potential_end if potential_end < next_block_start else len(model_output)

    def parse_tool_call_block(self, text: str, tools: list[Tool] | None) -> ToolCall | None:
        """
        Parse a single tool_call or function block into ToolCall.
        Falls back gracefully if malformed.
        """
        # Normalize: ensure <tool_call> wrapper
        if text.lstrip().startswith(self.function_prefix):
            if self.strict:
                raise ValueError(f"Missing {self.tool_call_start_token} in block")
            text = self.tool_call_start_token + text
            if not text.rstrip().endswith(self.function_end_token):
                if self.strict:
                    raise ValueError(f"Missing {self.function_end_token} in block")
                text += self.function_end_token

        func_name_pattern = self.func_name_pattern
        if tools:
            tool_names = [tool.function.name for tool in tools]
            if tool_names:
                func_name_pattern = re.compile(rf"{self.function_prefix}({'|'.join(re.escape(name) for name in tool_names)})>", re.DOTALL)

        func_match = func_name_pattern.search(text)
        if not func_match:
            if self.strict:
                raise ValueError(f"Missing {self.function_prefix}...> in block")
            return None

        func_name = func_match.group(1)

        params, pos = {}, func_match.end()
        while (m := self.param_start_pattern.search(text, pos)):
            name, start = m.group(1), m.end()
            chosen_close = self._find_param_end(text, start, name)
            raw_value = text[start:chosen_close].strip()

            param_cfg = self._get_arguments_config(func_name, tools)
            params[name] = self._convert_param_value(raw_value, name, param_cfg, func_name)

            pos = chosen_close + len(self.parameter_end_token)

        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:24]}",
            type=ToolType.FUNCTION,
            function=FunctionCall(
                name=func_name,
                arguments=json.dumps(params, ensure_ascii=False),
            ),
        )

    def extract_tool_calls(
        self, model_output: str, tools: list[Tool] | None = None
    ) -> Tuple[str, list[ToolCall] | None]:
        """Extract tool calls from model output.

        This method finds and parses all tool call blocks from the model's output.
        It can identify tool calls enclosed in `<tool_call>...</tool_call>` tags
        as well as tool calls starting directly with a known tool name.
        """

        tool_name_pattern = None
        if tools:
            tool_names = [tool.function.name for tool in tools]
            # Create a regex pattern to match any of the tool names
            tool_name_pattern = r"\s*<function=(" + "|".join(re.escape(name) for name in tool_names) + r")"

        def find_start(text: str, pos: int) -> tuple[int, int] | None:
            m = re.search(rf"{self.tool_call_start_token}", text[pos:])
            if not m and tool_name_pattern:
                m = re.search(tool_name_pattern, text[pos:], re.DOTALL)
            if not m:
                return None

            start_idx = pos + m.start()
            match_end_idx = pos + m.end()
            return start_idx, match_end_idx

        def find_end(text: str, start_idx: int, match_end_idx: int) -> int:
            # Find end </tool_call>
            end_idx = text.find(self.tool_call_end_token, start_idx)
            if end_idx != -1:
                return end_idx + len(self.tool_call_end_token)

            # If missing, recover until next tool_call or EOF
            next_start_idx = match_end_idx + 1
            next_block = re.search(rf"{self.tool_call_start_token}", text[next_start_idx:])
            next_tool_call = None
            if tool_name_pattern:
                # Also look for the next tool call by name
                next_tool_call = re.search(tool_name_pattern, text[next_start_idx:], re.DOTALL)

            # Choose the closest match
            candidates: list[int] = []
            if next_block:
                candidates.append(next_start_idx + next_block.start())
            if next_tool_call:
                candidates.append(next_start_idx + next_tool_call.start())

            return min(candidates) if candidates else len(text)

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


class Qwen3ChatTokenizer(ToolParsingChatTokenizer):
    """Tools handler for Qwen3 models with XML tool parsing support."""

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(tokenizer)
        self.tool_parser = Qwen3ToolParser()
