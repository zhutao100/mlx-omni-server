import ast
import json
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import regex

from ...utils.logger import logger
from ..schema import FunctionCall, Tool, ToolCall, ToolType


class BaseToolParser(ABC):
    """Base class for tool parsers."""

    tool_call_start_token: str
    tool_call_end_token: str
    tool_start_pattern: regex.Pattern | None

    @abstractmethod
    def extract_tool_calls(
        self, model_output: str, tools: list[Tool] | None = None
    ) -> Tuple[str, list[ToolCall] | None]:
        """
        Extract tool calls from model output.
        Returns the cleaned text and a list of ToolCall objects if any are found.
        """
        pass

    @abstractmethod
    def update_tool_start_pattern(self, tools: list[Tool] | None):
        """Update the potential tool start pattern based on available tools and model specific patterns."""
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
            except (TypeError, ValueError):
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not an integer in tool "
                    f"'{func_name}', degenerating to string."
                )
                return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                numeric_param_value = float(param_value)
            except (TypeError, ValueError):
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a float in tool "
                    f"'{func_name}', degenerating to string."
                )
                return param_value
            if numeric_param_value.is_integer():
                return int(numeric_param_value)
            return numeric_param_value
        elif param_type in ["boolean", "bool", "binary"]:
            normalized = param_value.strip().lower()
            if normalized not in ["true", "false"]:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a boolean (`true` of `false`) in tool '{func_name}', degenerating to false."
                )
            return normalized == "true"
        elif param_type == "object" or param_type.startswith("dict"):
            try:
                return json.loads(param_value)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a valid JSON value in tool "
                    f"'{func_name}', will try to parse it as a Python literal."
                )
            try:
                parsed = ast.literal_eval(param_value)
            except (SyntaxError, ValueError):
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' cannot be parsed as a JSON value or Python literal in tool '{func_name}', degenerating to string."
                )
                return param_value
            try:
                json.dumps(parsed)
            except TypeError:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' produced a non-JSON-serializable Python literal in tool '{func_name}', degenerating to string."
                )
                return param_value
            return parsed
        elif param_type == "array" or param_type.startswith("list"):
            try:
                return json.loads(param_value)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a valid JSON array in tool "
                    f"'{func_name}', will try to parse it as a Python literal."
                )
            try:
                parsed = ast.literal_eval(param_value)
            except (SyntaxError, ValueError):
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' cannot be parsed as a JSON array or Python literal in tool '{func_name}', degenerating to string."
                )
                return param_value
            if not isinstance(parsed, (list, tuple, set)):
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a list-like Python literal in tool '{func_name}', degenerating to string."
                )
                return param_value
            parsed_list = list(parsed)
            try:
                json.dumps(parsed_list)
            except TypeError:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' produced a non-JSON-serializable array literal in tool '{func_name}', degenerating to string."
                )
                return param_value
            return parsed_list
        else:
            # Unknown parameter type: accept JSON literals (safe) but do not execute/interpret arbitrary code.
            try:
                return json.loads(param_value)
            except (json.JSONDecodeError, TypeError):
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

    def update_tool_start_pattern(self, tools: list[Tool] | None):
        """Update the potential tool start pattern based on available tools and model specific patterns.

        Subclasses can override this with model specific patterns.
        """
        if not self.tool_call_start_token:
            self.tool_start_pattern = None
            self.tool_start_max_prefix_len = 0
            return

        # Default: literal start token search. Subclasses may override this with model-specific patterns.
        self.tool_start_pattern = regex.compile(rf"({re.escape(self.tool_call_start_token)})")
        self.tool_start_max_prefix_len = len(self.tool_call_start_token)

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
