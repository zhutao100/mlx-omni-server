import ast
import json
import re
from typing import Tuple
import uuid

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


class Glm4ToolParser(GenericToolParser):
    """Tool parser for Glm4's XML format that converts to OpenAI JSON format."""

    def __init__(self):
        self.strict = False
        # Sentinel tokens for streaming mode
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.arg_start_token: str = "<arg_key>"
        self.arg_end_token: str = "</arg_key>"
        self.value_start_token: str = "<arg_value>"
        self.value_end_token: str = "</arg_value>"

    def parse_tool_call_block(self, text: str, strict: bool = False) -> ToolCall | None:
        """
        Parse a <tool_call>... block in the format:
          <tool_call>function_name
          <arg_key>k</arg_key>
          <arg_value>v</arg_value>
          ...
          </tool_call>
        """
        # Tolerate leading junk / whitespace
        m = re.match(rf"\s*{self.tool_call_start_token}\s*([^\s<]+)", text, re.DOTALL)
        if not m:
            if strict:
                raise ValueError(f"Missing or malformed {self.tool_call_start_token}")
            return None

        func_name = m.group(1).strip()
        pos = m.end()
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
                if strict:
                    raise ValueError(f"Missing {self.value_start_token} for key {key}")
                break
            val_start = key_end + val_match.end()

            # Heuristic for the closing </arg_value>
            search_pos = val_start
            chosen_close = None
            while True:
                cand = text.find(rf"{self.value_end_token}", search_pos)
                if cand == -1:
                    if strict:
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
            args[key] = raw_value.strip()
            pos = chosen_close + len(self.value_end_token)

        return ToolCall(id=f"call_{uuid.uuid4().hex[:24]}",
                        type=ToolType.FUNCTION,
                        function=FunctionCall(
            name=func_name, arguments=json.dumps(args, ensure_ascii=False)))

    def extract_tool_calls(
        self, model_output: str, tools: list[Tool] | None = None
    ) -> Tuple[str, list[ToolCall] | None]:
        """Extract tool calls from model output in XML format."""

        results = []
        rest_parts = []
        pos = 0
        n = len(model_output)

        while pos < n:
            # Find next <tool_call>
            m = re.search(rf"{self.tool_call_start_token}", model_output[pos:])
            if not m:
                rest_parts.append(model_output[pos:])
                break

            start_idx = pos + m.start()
            rest_parts.append(model_output[pos:start_idx])  # text before block

            # Find end </tool_call>
            end_idx = model_output.find(self.tool_call_end_token, start_idx)
            if end_idx == -1:
                # If missing, recover until next <tool_call> or EOF
                next_block = re.search(rf"{self.tool_call_start_token}", model_output[start_idx + 1:])
                if next_block:
                    block_end = start_idx + 1 + next_block.start()
                else:
                    block_end = n
            else:
                block_end = end_idx + len(self.tool_call_end_token)

            block = model_output[start_idx:block_end]

            try:
                parsed = self.parse_tool_call_block(block, strict=self.strict)
            except ValueError:
                if self.strict:
                    raise
                parsed = None

            if parsed:
                results.append(parsed)
            else:
                rest_parts.append(block)

            pos = block_end

        rest_text = self._normalize_text("".join(rest_parts))
        logger.debug("Extracted tool calls %s", results)
        logger.debug("Remaining text: %s", "".join(rest_parts))
        return rest_text, results


class Glm4ChatTokenizer(ToolParsingChatTokenizer):
    """Tools handler for Glm4 models with XML tool parsing support."""

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(tokenizer)
        self.tool_parser = Glm4ToolParser()
