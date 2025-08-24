import ast
import json
import re
import logging
from typing import Dict, Any, Tuple
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
        # Sentinel tokens for streaming mode
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        # Track tool calls for finish_reason handling (like vLLM)
        self.prev_tool_call_arr = []

        # XML parsing patterns (matching vLLM exactly)
        self.func_call_regex = re.compile(r"<tool_call>.*?</tool_call>",
                                          re.DOTALL)
        self.func_detail_regex = re.compile(
            r"<tool_call>([^\n]*)\n(.*)</tool_call>", re.DOTALL)
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
            re.DOTALL)

    def extract_tool_calls(
        self, model_output: str, tools: list[Tool] | None = None
    ) -> Tuple[str, list[ToolCall] | None]:
        """Extract tool calls from model output and return in OpenAI format."""

        def _is_string_type(
                tool_name: str, arg_name: str,
                tools: list[Tool] | None) -> bool:
            if not tools:
                return False
            for tool in tools:
                if tool.function.name == tool_name:
                    if not tool.function.parameters or not tool.function.parameters.properties:
                        return False
                    properties = tool.function.parameters.properties
                    arg_type = properties.get(arg_name, {}).get("type", None)
                    return arg_type == "string"
            logger.warning("No tool named '%s'.", tool_name)
            return False

        def _deserialize(value: str) -> Any:
            try:
                return json.loads(value)
            except Exception:
                pass

            try:
                return ast.literal_eval(value)
            except Exception:
                pass
            return value

        logger.debug("model_output: %s", model_output)
        matched_tool_calls = self.func_call_regex.findall(model_output)

        try:
            tool_calls = []
            for match in matched_tool_calls:
                tc_detail = self.func_detail_regex.search(match)
                if tc_detail is None:
                    logger.warning("Failed to parse tool call details from match: %s", match)
                    continue
                tc_name = tc_detail.group(1)
                tc_args = tc_detail.group(2)
                pairs = self.func_arg_regex.findall(tc_args)
                arg_dct = {}
                for key, value in pairs:
                    arg_key = key.strip()
                    arg_val = value.strip()
                    if not _is_string_type(tc_name, arg_key, tools):
                        arg_val = _deserialize(arg_val)
                    logger.debug("arg_key = %s, arg_val = %s", arg_key,
                                 arg_val)
                    arg_dct[arg_key] = arg_val
                tool_calls.append(
                    ToolCall(id=f"call_{uuid.uuid4().hex[:24]}",
                             type=ToolType.FUNCTION,
                             function=FunctionCall(
                                 name=tc_name, arguments=json.dumps(arg_dct))))
        except Exception:
            logger.exception("Failed to extract tool call spec")
            return model_output, None
        else:
            content = model_output[:model_output.
                                   find(self.tool_call_start_token)]
            return content, tool_calls if tool_calls else None


class Glm4ChatTokenizer(ToolParsingChatTokenizer):
    """Tools handler for Glm4 models with XML tool parsing support."""

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(tokenizer)
        self.tool_parser = Glm4ToolParser()
