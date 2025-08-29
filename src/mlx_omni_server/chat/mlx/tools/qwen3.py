import json
import re
from rich.markup import escape
import uuid
from typing import Tuple

from mlx_lm.tokenizer_utils import TokenizerWrapper

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

    def __init__(self):
        self.strict = False
        # Sentinel tokens for streaming mode
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.function_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"

    def parse_tool_call_block(self, text: str, tools: list[Tool] | None, strict: bool = False) -> ToolCall | None:
        """Parse a tool call or tolerant a function block."""
        if text.lstrip().startswith(self.function_prefix):
            if strict:
                raise ValueError(f"Missing {self.tool_call_start_token} in block")
            text = self.tool_call_start_token + text
            if not text.rstrip().endswith(self.function_end_token):
                if strict:
                    raise ValueError(f"Missing {self.tool_call_end_token} in block")
                text = text + self.tool_call_end_token

        func_match = re.search(rf"{self.function_prefix}([^>]+)>", text)
        if not func_match:
            if strict:
                raise ValueError(f"Missing {self.function_prefix}...> in block")
            return None
        func_name = func_match.group(1)

        params = {}
        pos = func_match.end()
        while True:
            m = re.search(rf"{self.parameter_prefix}([^>]+)>", text[pos:])
            if not m:
                break
            name = m.group(1)
            start = pos + m.end()

            # Candidate scan for real closing
            search_pos = start
            chosen_close = None
            while True:
                cand = text.find(self.parameter_end_token, search_pos)
                if cand == -1:
                    if strict:
                        raise ValueError(f"Missing {self.parameter_end_token} for '{name}'")
                    chosen_close = len(text)
                    break

                after = text[cand + len(self.parameter_end_token):].lstrip()
                if (after.startswith(self.parameter_prefix)
                    or after.startswith(self.function_end_token)
                    or after.startswith(self.tool_call_end_token)
                        or after == ""):
                    chosen_close = cand
                    break

                # Otherwise treat as literal inside content
                search_pos = cand + len(self.parameter_end_token)

            raw_value = text[start:chosen_close]
            params[name] = raw_value

            # Remove prefix and trailing \n
            if raw_value.startswith("\n"):
                raw_value = raw_value[1:]
            if raw_value.endswith("\n"):
                raw_value = raw_value[:-1]

            param_config = self._get_arguments_config(func_name, tools)
            params[name] = self._convert_param_value(
                raw_value, name, param_config, func_name
            )

            pos = chosen_close + len(self.parameter_end_token)

        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:24]}",
            type=ToolType.FUNCTION,
            function=FunctionCall(
                name=func_name,
                arguments=json.dumps(params, ensure_ascii=False)
            ),
        )

    def extract_tool_calls(
        self, model_output: str, tools: list[Tool] | None = None
    ) -> Tuple[str, list[ToolCall] | None]:
        """Parse all tool_call blocks, recover across multiple malformed ones."""
        results = []
        rest_parts = []
        pos = 0
        n = len(model_output)

        while pos < n:
            # Find next candidate block opener
            m = re.search(rf"{self.tool_call_start_token}|{self.function_prefix}[^>]+>", model_output[pos:])
            if not m:
                rest_parts.append(model_output[pos:])
                break

            start_idx = pos + m.start()
            rest_parts.append(model_output[pos:start_idx])  # text before block

            # Try to find matching </tool_call>
            end_idx = model_output.find(self.tool_call_end_token, start_idx)
            next_block = re.search(
                rf"{self.tool_call_start_token}|{self.function_prefix}[^>]+>", model_output[start_idx + 1:])

            if end_idx == -1:
                if next_block:
                    # End block just before the next block
                    block_end = start_idx + 1 + next_block.start()
                else:
                    block_end = n
            else:
                block_end = end_idx + len(self.tool_call_end_token)

            block = model_output[start_idx:block_end]

            try:
                parsed = self.parse_tool_call_block(block, tools=tools, strict=self.strict)
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


class Qwen3ChatTokenizer(ToolParsingChatTokenizer):
    """Tools handler for Qwen3 models with XML tool parsing support."""

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(tokenizer)
        self.tool_parser = Qwen3ToolParser()
