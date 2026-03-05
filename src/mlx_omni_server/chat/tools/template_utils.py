from __future__ import annotations

import json
from typing import Any

from ...utils.logger import logger


def _normalize_function_arguments_in_place(container: dict[str, Any]) -> None:
    function = container.get("function")
    if not isinstance(function, dict):
        return

    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        return

    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Failed to parse tool arguments as JSON: %s", exc)
        return

    if isinstance(parsed, dict):
        function["arguments"] = parsed


def normalize_tool_calls_for_template(messages: list[dict[str, Any]]) -> None:
    """Ensure assistant tool call arguments are dicts (not JSON strings).

    Our Jinja chat templates treat `tool_call.arguments` as a mapping. Parsing here avoids
    doing JSON parsing in templates and matches the LM path behavior.
    """
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                _normalize_function_arguments_in_place(tool_call)


def normalize_tools_for_template(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize any accidental `function.arguments` fields inside tool dicts."""
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        processed = dict(tool)
        _normalize_function_arguments_in_place(processed)
        normalized.append(processed)
    return normalized
