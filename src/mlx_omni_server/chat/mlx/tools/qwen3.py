import json
import re
from rich.markup import escape
import uuid
from typing import Tuple, Pattern

from mlx_lm.tokenizer_utils import TokenizerWrapper
import regex

from ....utils.logger import logger
from ...schema import (
    FunctionCall,
    Tool,
    ToolCall,
    ToolType,
)
from .chat_tokenizer import ToolParsingChatTokenizer
from .tool_parser import GenericToolParser


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
        # Use (?<=^|\n) to match at the beginning of string or after a newline, instead of \s* which matches any whitespace
        pattern = rf"(?<=^|\n)({self.tool_call_start_token}|{self.function_prefix})"
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

        func_match = self.func_name_pattern.search(text)
        if not func_match:
            if self.strict:
                raise ValueError(f"Missing {self.function_prefix}...> in block")
            return None

        func_name = func_match.group(1)
        if tools and func_name not in {t.function.name for t in tools}:
            logger.warning(f"Parsed tool call `{func_name}` not in registered tool list")
            return None

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
        """
        Parse all tool_call blocks from model output.
        Returns (remaining_text, [ToolCall...]).
        """
        results, rest_parts, pos = [], [], 0

        while pos < len(model_output):
            m = self.tool_call_or_function_start_pattern.search(model_output, pos)
            if not m:
                rest_parts.append(model_output[pos:])
                break

            start_idx, next_search_start = m.start(), m.end()
            rest_parts.append(model_output[pos:start_idx])

            next_block_pattern = (
                self.tool_call_start_pattern
                if m.group(0).startswith(self.tool_call_start_token)
                else self.tool_call_or_function_start_pattern
            )
            end_idx = self._find_block_end(model_output, start_idx, next_search_start, next_block_pattern)
            block = model_output[start_idx:end_idx]

            try:
                parsed = self.parse_tool_call_block(block, tools)
            except ValueError:
                if self.strict:
                    raise
                parsed = None

            (results if parsed else rest_parts).append(parsed or block)
            pos = end_idx

        rest_text = "".join(rest_parts)
        logger.debug("Extracted tool calls %s", results)
        logger.debug("Remaining text: %s", rest_text)
        return rest_text, results


class Qwen3ChatTokenizer(ToolParsingChatTokenizer):
    """Tools handler for Qwen3 models with XML tool parsing support."""

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(tokenizer)
        self.tool_parser = Qwen3ToolParser()
