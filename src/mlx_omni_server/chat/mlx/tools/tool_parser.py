from abc import ABC, abstractmethod
import json
import re
from typing import Any, Dict, Tuple
import uuid

from ....utils.logger import logger
from ...schema import FunctionCall, Tool, ToolCall, ToolType


class BaseToolParser(ABC):
    """Base class for tool parsers."""

    tool_call_start_token: str
    tool_call_end_token: str

    @abstractmethod
    def extract_tool_calls(
        self, model_output: str, tools: list[Tool] | None = None
    ) -> Tuple[str, list[ToolCall] | None]:
        """
        Extract tool calls from model output.
        Returns the cleaned text and a list of ToolCall objects if any are found.
        """
        pass


class GenericToolParser(BaseToolParser):
    """Base class for tool parsers."""

    def __init__(self, tool_call_start_token: str, tool_call_end_token: str):
        self.tool_call_start_token = tool_call_start_token
        self.tool_call_end_token = tool_call_end_token

    def _extract_tools(self, text: str) -> list[dict[str, Any]] | None:
        results = []

        pattern = (
            r'"name"\s*:\s*"([^"]+)"'  # Match name
            r"(?:"  # Start non-capturing group for optional arguments/parameters
            r"[^}]*?"  # Allow any characters in between
            r'(?:"arguments"|"parameters")'  # Match arguments or parameters
            r"\s*:\s*"  # Match colon and whitespace
            r"("  # Start capturing parameter value
            r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"  # Match nested objects
            r"|\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]"  # Match arrays
            r"|null"  # Match null
            r'|"[^"]*"'  # Match strings
            r")"  # End capturing
            r")?"  # Make the entire arguments/parameters section optional
        )

        matches = re.finditer(pattern, text, re.DOTALL)

        matches_list = list(matches)
        for i, match in enumerate(matches_list):
            name, args_str = match.groups()
            results.append(
                {
                    "type": "function",
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "function": {
                        "name": name,
                        "arguments": args_str if args_str else "{}",
                    }
                }
            )

        return results

    def _convert_param_value(self, param_value: str, param_name: str, param_config: dict, func_name: str) -> Any:
        """Convert parameter value based on its expected type."""
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                logger.warning(
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
                logger.warning(
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
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a float in tool "
                    f"'{func_name}', degenerating to string."
                )
                return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value not in ["true", "false"]:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a boolean (`true` of `false`) in tool '{func_name}', degenerating to false."
                )
            return param_value == "true"
        else:
            if param_type == "object" or param_type.startswith("dict"):
                try:
                    param_value = json.loads(param_value)
                    return param_value
                except:
                    logger.warning(
                        f"Parsed value '{param_value}' of parameter '{param_name}' is not a valid JSON object in tool "
                        f"'{func_name}', will try other methods to parse it."
                    )
            try:
                param_value = eval(param_value)
            except:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' cannot be converted via Python `eval()` in tool '{func_name}', degenerating to string."
                )
            return param_value

    def _get_arguments_config(self, func_name: str, tools: list[Tool] | None) -> dict[str, Any]:
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
        logger.warning(f"Tool '{func_name}' is not defined in the tools list {tools_names}.")
        return {}

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

    def _normalize_text(self, s: str) -> str:
        """Normalize whitespace: strip edges, collapse multiple blank lines."""
        lines = [ln.rstrip() for ln in s.splitlines()]
        cleaned = "\n".join(ln for ln in lines if ln.strip() != "")
        return cleaned.strip()

    def extract_tool_calls(
        self, model_output: str, tools: list[Tool] | None = None
    ) -> Tuple[str, list[ToolCall] | None]:
        """
        Extract tool calls from model output.
        Returns the cleaned text and a list of ToolCall objects if any are found.
        """

        try:
            tool_calls = self._extract_tools(model_output)
            if tool_calls:
                results = []
                for call in tool_calls:
                    tool_call = self._create_tool_call_from_data(call)
                    results.append(tool_call)
                # TODO: Clean the model_output to remove tool call snippets and return the cleaned text
                return model_output, results

            return model_output, None
        except Exception as e:
            logger.error(f"Error during regex matching: {str(e)}")
            return model_output, None
