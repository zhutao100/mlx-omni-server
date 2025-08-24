import re
import uuid
import logging
from typing import Dict, Any, Tuple

from mlx_lm.tokenizer_utils import TokenizerWrapper

from ....utils.logger import logger
from ...schema import (
    Tool,
    ToolCall,
)
from .chat_tokenizer import ToolParsingChatTokenizer
from .tool_parser import GenericToolParser


class Qwen3ToolParser(GenericToolParser):
    """Tool parser for Qwen3's XML format that converts to OpenAI JSON format."""

    def __init__(self):
        # Sentinel tokens for streaming mode
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"
        # Track tool calls for finish_reason handling (like vLLM)
        self.prev_tool_call_arr = []

        # XML parsing patterns (matching vLLM exactly)
        self.tool_call_regex = re.compile(
            rf"{self.tool_call_start_token}(.*?){self.tool_call_end_token}|{self.tool_call_start_token}(.*?)$", re.DOTALL
        )
        self.tool_call_function_regex = re.compile(
            rf"{self.tool_call_prefix}(.*?){self.function_end_token}|{self.tool_call_prefix}(.*?)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            rf"{self.parameter_prefix}(.*?){self.parameter_end_token}|{self.parameter_prefix}(.*?)$", re.DOTALL
        )

    def _parse_xml_function_call(self, function_call_str: str, tools: list[Tool] | None) -> Dict[str, Any] | None:
        """Parse XML function call format to OpenAI JSON format."""
        try:
            # Handle incomplete XML gracefully
            if ">" not in function_call_str:
                logging.warning(f"Incomplete XML function call: {function_call_str[:100]}...")
                return None

            # Extract function name
            end_index = function_call_str.index(">")
            function_name = function_call_str[:end_index]
            param_config = self._get_arguments_config(function_name, tools)
            parameters = function_call_str[end_index + 1:]

            param_dict = {}

            # Handle incomplete parameters more gracefully
            parameter_matches = self.tool_call_parameter_regex.findall(parameters)
            for match in parameter_matches:
                try:
                    match_text = match[0] if match[0] else match[1]
                    if ">" not in match_text:
                        logging.warning(f"Incomplete parameter in XML: {match_text[:50]}...")
                        continue

                    idx = match_text.index(">")
                    param_name = match_text[:idx]
                    param_value = str(match_text[idx + 1:])

                    # Remove prefix and trailing \n
                    if param_value.startswith("\n"):
                        param_value = param_value[1:]
                    if param_value.endswith("\n"):
                        param_value = param_value[:-1]

                    param_dict[param_name] = self._convert_param_value(
                        param_value, param_name, param_config, function_name
                    )
                except Exception as param_e:
                    logging.warning(f"Error parsing parameter {match}: {param_e}")
                    continue

            return {
                "type": "function",
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "function": {
                    "name": function_name,
                    "arguments": param_dict
                }
            }
        except Exception as e:
            logging.error(f"Error parsing XML function call '{function_call_str[:100]}...': {e}")
            return None

    def _get_function_calls(self, model_output: str) -> list[str]:
        """Extract function calls from model output (matching vLLM implementation)."""
        # Find all tool calls
        matched_ranges = self.tool_call_regex.findall(model_output)
        raw_tool_calls = [
            match[0] if match[0] else match[1] for match in matched_ranges
        ]

        # Back-off strategy if no tool_call tags found (like vLLM)
        if len(raw_tool_calls) == 0:
            raw_tool_calls = [model_output]

        raw_function_calls = []
        for tool_call in raw_tool_calls:
            raw_function_calls.extend(self.tool_call_function_regex.findall(tool_call))

        function_calls = [
            match[0] if match[0] else match[1] for match in raw_function_calls
        ]
        return function_calls

    def extract_tool_calls(
        self, model_output: str, tools: list[Tool] | None = None
    ) -> Tuple[str, list[ToolCall] | None]:
        """Extract tool calls from model output and return in OpenAI format."""
        # Quick check to avoid unnecessary processing (like vLLM)
        if self.tool_call_prefix not in model_output:
            return model_output, None
        try:
            function_call_str_list: list[str] = self._get_function_calls(model_output)
            if not function_call_str_list:
                return model_output, None

            tool_calls: list[ToolCall] = []
            self.prev_tool_call_arr.clear()  # Clear previous calls
            for function_call_str in function_call_str_list:
                parsed_call = self._parse_xml_function_call(function_call_str, tools)
                if parsed_call:
                    tool_calls.append(self._create_tool_call_from_data(parsed_call))
                    # Populate prev_tool_call_arr for serving layer to set finish_reason (like vLLM)
                    self.prev_tool_call_arr.append(
                        {
                            "name": parsed_call["function"]["name"],
                            "arguments": parsed_call["function"]["arguments"],
                        }
                    )

            # Extract content before tool calls (like vLLM - no rstrip)
            content_index = model_output.find(self.tool_call_start_token)
            content_index = (
                content_index
                if content_index >= 0
                else model_output.find(self.tool_call_prefix)
            )
            content = model_output[:content_index] if content_index > 0 else ""

            return content, tool_calls
        except Exception as e:
            logging.error(f"Error in extracting tool call from response: {e}")
            return model_output, None


class Qwen3ChatTokenizer(ToolParsingChatTokenizer):
    """Tools handler for Qwen3 models with XML tool parsing support."""

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(tokenizer)
        self.tool_parser = Qwen3ToolParser()
