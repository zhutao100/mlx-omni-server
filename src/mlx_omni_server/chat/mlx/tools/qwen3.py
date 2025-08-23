import json
import re
import uuid
import logging
from typing import Dict, Any, Tuple

from mlx_lm.tokenizer_utils import TokenizerWrapper

from ....utils.logger import logger
from ...schema import (
    ChatMessage,
    FunctionCall,
    Role,
    Tool,
    ToolCall,
    ToolChoiceType,
    ToolType,
)
from .chat_tokenizer import ChatTokenizer

# Sentinel tokens
_TOOL_CALL_PREFIX = "<function="
_TOOL_CALL_POSTFIX = "</function>"
_TOOL_CALL_START_TOKEN = "<tool_call>"
_TOOL_CALL_END_TOKEN = "</tool_call>"


class Qwen3ToolParser:
    """Tool parser for Qwen3's XML format that converts to OpenAI JSON format."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        # Track tool calls for finish_reason handling (like vLLM)
        self.prev_tool_call_arr = []

        # XML parsing patterns (matching vLLM exactly)
        self.tool_call_regex = re.compile(
            rf"{_TOOL_CALL_START_TOKEN}(.*?){_TOOL_CALL_END_TOKEN}|{_TOOL_CALL_START_TOKEN}(.*?)$", re.DOTALL
        )
        self.tool_call_function_regex = re.compile(
            rf"{_TOOL_CALL_PREFIX}(.*?){_TOOL_CALL_POSTFIX}|{_TOOL_CALL_PREFIX}(.*?)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)</parameter>|<parameter=(.*?)$", re.DOTALL
        )

    def _convert_param_value(self, param_value: str, param_name: str, param_config: dict, func_name: str) -> Any:
        """Convert parameter value based on its expected type."""
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                logging.warning(
                    f"Parsed parameter '{param_name}' is not defined in the tool "
                    f"parameters for tool '{func_name}', directly returning the string value."
                )
            return param_value

        if (
            isinstance(param_config[param_name], dict)
            and "type" in param_config[param_name]
        ):
            param_type = str(param_config[param_name]["type"]).strip().lower()
        else:
            param_type = "string"

        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                int_param_value = int(param_value)
                return int_param_value
            except:
                logging.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not an integer in tool "
                    f"'{func_name}', degenerating to string."
                )
                return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                numeric_param_value = float(param_value)
                numeric_param_value = numeric_param_value if numeric_param_value - \
                    int(numeric_param_value) != 0 else int(numeric_param_value)
                return numeric_param_value
            except:
                logging.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a float in tool "
                    f"'{func_name}', degenerating to string."
                )
                return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value not in ["true", "false"]:
                logging.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a boolean (`true` of `false`) in tool '{func_name}', degenerating to false."
                )
            return param_value == "true"
        else:
            if param_type == "object" or param_type.startswith("dict"):
                try:
                    param_value = json.loads(param_value)
                    return param_value
                except:
                    logging.warning(
                        f"Parsed value '{param_value}' of parameter '{param_name}' is not a valid JSON object in tool "
                        f"'{func_name}', will try other methods to parse it."
                    )
            try:
                param_value = eval(param_value)
            except:
                logging.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' cannot be converted via Python `eval()` in tool '{func_name}', degenerating to string."
                )
            return param_value

    def _get_arguments_config(self, func_name: str, tools: list[Tool] | None) -> dict:
        """Get parameter configuration for a function from tools list."""
        if tools is None:
            return {}

        tools_names = []
        for tool in tools:
            if tool.type == ToolType.FUNCTION and tool.function:
                tools_names.append(tool.function.name)
                if tool.function.name == func_name:
                    params = tool.function.parameters
                    if params and params.properties:
                        return params.properties
                    else:
                        return {}
        logging.warning(f"Tool '{func_name}' is not defined in the tools list {tools_names}.")
        return {}

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

    def _create_tool_call_from_data(self, tool_call_data: Dict[str, Any]) -> ToolCall:
        """Create a ToolCall object from parsed tool call data."""
        args = tool_call_data["function"]["arguments"]
        return ToolCall(
            id=tool_call_data["id"],
            type=tool_call_data["type"],
            function=FunctionCall(
                name=tool_call_data["function"]["name"],
                arguments=args if isinstance(args, str) else json.dumps(args),
            ),
        )

    def extract_tool_calls(
        self, model_output: str, tools: list[Tool] | None = None
    ) -> Tuple[str, list[ToolCall] | None]:
        """Extract tool calls from model output and return in OpenAI format."""
        # Quick check to avoid unnecessary processing (like vLLM)
        if _TOOL_CALL_PREFIX not in model_output:
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
            content_index = model_output.find(_TOOL_CALL_START_TOKEN)
            content_index = (
                content_index
                if content_index >= 0
                else model_output.find(_TOOL_CALL_PREFIX)
            )
            content = model_output[:content_index] if content_index > 0 else ""

            return content, tool_calls
        except Exception as e:
            logging.error(f"Error in extracting tool call from response: {e}")
            return model_output, None


class Qwen3ChatTokenizer(ChatTokenizer):
    """Tools handler for Qwen3 models with XML tool parsing support."""

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(tokenizer)
        self.start_tool_calls = ""
        self.end_tool_calls = ""
        self.tool_parser = Qwen3ToolParser(tokenizer)
        self.pre_fill_tools_prompt = ""
        self.buffer = ""
        self.left_bracket_pos = -1  # Position of the first '<' in the buffer

    def encode(
        self,
        messages: list[ChatMessage],
        tools: list[Tool] | None = None,
        tool_choice: ToolChoiceType | None = None,
        **kwargs,
    ) -> str:
        """Encode tools and conversation into a prompt string."""
        # Use the parent class's encode method which uses the tokenizer's chat template
        prompt = super().encode(messages, tools, tool_choice, **kwargs)

        # For Qwen3, we don't need to prefill tools as the tokenizer handles it
        return prompt

    def decode_stream(
        self, delta_text: str, tools: list[Tool] | None = None
    ) -> ChatMessage | None:
        """Parse tool calls from model output in streaming mode."""
        self.buffer += delta_text

        skip_delta = False
        # Simple approach: stop streaming as soon as we see < character in buffer
        if self.left_bracket_pos < 0:
            self.left_bracket_pos = self.buffer.find("<")
            if self.left_bracket_pos >= 0:
                # Calculate what part of this segment comes before the <
                text_before_segment = (
                    self.buffer[: -len(delta_text)]
                    if len(delta_text) <= len(self.buffer)
                    else ""
                )

                if self.left_bracket_pos >= len(text_before_segment):
                    # The < is in this segment
                    chars_before_bracket = self.left_bracket_pos - len(
                        text_before_segment
                    )
                    delta_text = delta_text[:chars_before_bracket]
                else:
                    # The < was in previous segments, don't send anything
                    delta_text = ""
                    skip_delta = True
            else:
                # No < found yet, send the segment
                pass
        else:
            # Already detected <, don't send anything more
            delta_text = ""
            skip_delta = True

        if not skip_delta:
            return ChatMessage(
                role=Role.ASSISTANT,
                content=delta_text,
            )

    def parse_buffer(self, tools: list[Tool] | None = None) -> ChatMessage | None:
        """Process the buffer to extract complete tool calls."""
        if self.left_bracket_pos < 0:
            return None  # No left bracket found, nothing to parse

        self.buffer = self.buffer[self.left_bracket_pos:]
        tool_calls = []
        # Check for complete tool calls in the buffer
        while _TOOL_CALL_END_TOKEN in self.buffer:
            match = self.tool_parser.tool_call_function_regex.search(self.buffer)
            if not match:
                break

            tool_call_str = match.group(1)
            # Parse the complete tool call
            parsed_call = self.tool_parser._parse_xml_function_call(tool_call_str, tools)
            if parsed_call:
                tool_calls.append(self.tool_parser._create_tool_call_from_data(parsed_call))

            # Remove the parsed tool call from the buffer
            self.buffer = (
                self.buffer[match.end():].lstrip().removeprefix(_TOOL_CALL_END_TOKEN)
            )
            self.left_bracket_pos = self.buffer.find("<")

        content = self.buffer if self.buffer else None
        self.buffer = ""  # Clear the buffer after parsing
        self.left_bracket_pos = -1  # Reset left bracket position
        if tool_calls:
            return ChatMessage(
                role=Role.ASSISTANT,
                content=content,
                tool_calls=tool_calls,
            )

        else:
            logger.warning("No matched tool calls in buffer, sending as content.")
            if content:
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content=content,
                    tool_calls=None,
                )

    def decode(self, text: str, tools: list[Tool] | None = None) -> ChatMessage | None:
        """Parse tool calls from model output in non-streaming mode."""
        # Use the Qwen3 tool parser to extract tool calls from XML format
        content, tool_calls = self.tool_parser.extract_tool_calls(text, tools)

        return ChatMessage(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        )
