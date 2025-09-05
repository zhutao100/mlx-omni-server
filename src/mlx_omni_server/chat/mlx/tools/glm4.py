import ast
import logging
import json
import re
from rich.markup import escape
from typing import Tuple
import uuid

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


class Glm4ToolParser(GenericToolParser):
    """Tool parser for Glm4's XML format that converts to OpenAI JSON format."""

    def __init__(self, tool_call_start_token="<tool_call>", tool_call_end_token="</tool_call>"):
        super().__init__(tool_call_start_token=tool_call_start_token, tool_call_end_token=tool_call_end_token)
        self.strict = False
        # Sentinel tokens for streaming mode
        self.arg_start_token: str = "<arg_key>"
        self.arg_end_token: str = "</arg_key>"
        self.value_start_token: str = "<arg_value>"
        self.value_end_token: str = "</arg_value>"

    def update_tool_start_pattern(self, tools: list[Tool] | None):
        """Update the potential tool start pattern based on available tools."""
        if tools:
            tool_names = [tool.function.name for tool in tools]
            # Create a regex pattern to match any of the tool names at the start of a line or after a newline
            # Use (?<=^|\n) to match at the beginning of string or after a newline, instead of \s* which matches any whitespace
            tool_names_group = "|".join(re.escape(name) for name in tool_names)
            pattern = rf"(?<=^|\n)({self.tool_call_start_token}|(?:{tool_names_group})(?=[<\n]))"
            self.tool_start_pattern = regex.compile(pattern, re.DOTALL)
        else:
            self.tool_start_pattern = None

    def parse_tool_call_block(self, text: str, tools: list[Tool] | None) -> ToolCall | None:
        """
        Parse a <tool_call>... block in the format:
          <tool_call>function_name
          <arg_key>k</arg_key>
          <arg_value>v</arg_value>
          ...
          </tool_call>

        Or the alternate format:
          <tool_call>
          <function=function_name>
          <arg_key>k</arg_key>
          <arg_value>v</arg_value>
          ...
          </tool_call>

        It handles the standard format with a `<tool_call>` tag. If not in strict
        mode, it can also parse blocks that start directly with a valid tool name.
        """
        func_name = None
        pos = 0

        # Standard format: <tool_call>function_name
        m_std = re.match(rf"\s*(?:{self.tool_call_start_token}\s*)*([^\s<]+)", text, re.DOTALL)
        if m_std:
            func_name = m_std.group(1).strip()
            pos = m_std.end()

        # Alternate format: <function=function_name>
        if not func_name:
            m_alt1 = re.search(r"<function=([^\s>]+)>", text)
            if m_alt1:
                func_name = m_alt1.group(1).strip()
                pos = m_alt1.end()

        # Alternate format: just function_name (if not strict)
        if not func_name and not self.strict and tools:
            tool_names = {tool.function.name for tool in tools}
            pattern = r"\s*(" + "|".join(re.escape(name) for name in tool_names) + r")"
            m_alt2 = re.match(pattern, text, re.DOTALL)
            if m_alt2:
                func_name = m_alt2.group(1).strip()
                pos = m_alt2.end()

        if not func_name:
            if self.strict:
                raise ValueError("Missing or malformed function name")
            return None

        # Centralized check for tool name validity
        if tools and func_name not in {t.function.name for t in tools}:
            logger.warning(f"Tool '{func_name}' is not defined in the tools list.")
            return None

        args = {}
        while True:
            key_match = re.search(rf"{self.arg_start_token}(.*?){self.arg_end_token}", text[pos:], re.DOTALL)
            if not key_match:
                break
            key = key_match.group(1).strip()
            key_end = pos + key_match.end()

            # Find matching <arg_value>... </arg_value>
            val_match = re.search(rf"{self.value_start_token}", text[key_end:])
            if not val_match:
                if self.strict:
                    raise ValueError(f"Missing {self.value_start_token} for key {key}")
                break
            val_start = key_end + val_match.end()

            # Heuristic for the closing </arg_value>
            search_pos = val_start
            chosen_close = None
            while True:
                cand = text.find(rf"{self.value_end_token}", search_pos)
                if cand == -1:
                    if self.strict:
                        raise ValueError(f"Missing {self.value_end_token} for key {key}")
                    chosen_close = len(text)
                    break

                after = text[cand + len(self.value_end_token):].lstrip()
                if after.startswith(self.arg_start_token) or after.strip().startswith(self.tool_call_end_token) or after == "":
                    chosen_close = cand
                    break

                # Otherwise treat as literal inside content
                search_pos = cand + len(self.value_end_token)

            raw_value = text[val_start:chosen_close]
            param_config = self._get_arguments_config(func_name, tools)
            args[key] = self._convert_param_value(
                raw_value, key, param_config, func_name
            )
            pos = chosen_close + len(self.value_end_token)

        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:24]}",
            type=ToolType.FUNCTION,
            function=FunctionCall(
                name=func_name, arguments=json.dumps(args, ensure_ascii=False)
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

        results = []
        rest_parts = []
        pos = 0
        n = len(model_output)

        tool_name_pattern = None
        if tools:
            tool_names = [tool.function.name for tool in tools]
            # Create a regex pattern to match any of the tool names
            tool_name_pattern = r"\s*(" + "|".join(re.escape(name) for name in tool_names) + r")"

        while pos < n:
            # Find next <tool_call>
            m = re.search(rf"{self.tool_call_start_token}", model_output[pos:])
            if not m and tool_name_pattern:
                m = re.search(tool_name_pattern, model_output[pos:], re.DOTALL)
            if not m:
                rest_parts.append(model_output[pos:])
                break

            start_idx = pos + m.start()
            rest_parts.append(model_output[pos:start_idx])  # text before block

            # Find end </tool_call>
            end_idx = model_output.find(self.tool_call_end_token, start_idx)
            if end_idx == -1:
                # If missing, recover until next tool_call or EOF
                next_start_idx = pos + m.end() + 1
                next_block = re.search(rf"{self.tool_call_start_token}", model_output[next_start_idx:])
                next_tool_call = None
                if tool_name_pattern:
                    # Also look for the next tool call by name
                    next_tool_call = re.search(tool_name_pattern, model_output[next_start_idx:], re.DOTALL)

                # Choose the closest match
                candidates = []
                if next_block:
                    candidates.append(next_start_idx + next_block.start())
                if next_tool_call:
                    candidates.append(next_start_idx + next_tool_call.start())

                if candidates:
                    block_end = min(candidates)
                else:
                    block_end = n
            else:
                block_end = end_idx + len(self.tool_call_end_token)

            block = model_output[start_idx:block_end]

            try:
                parsed = self.parse_tool_call_block(block, tools=tools)
            except ValueError:
                if self.strict:
                    raise
                parsed = None

            if parsed:
                results.append(parsed)
            else:
                rest_parts.append(block)

            pos = block_end

        rest_text = "".join(rest_parts)
        logger.debug(escape("Extracted tool calls %s"), results)
        logger.debug(escape("Remaining text: %s"), "".join(rest_parts))
        return rest_text, results


class Glm4ChatTokenizer(ToolParsingChatTokenizer):
    """Tools handler for Glm4 models with XML tool parsing support."""

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(tokenizer)
        self.tool_parser = Glm4ToolParser()
