import json
import re
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from ...schemas.chat_schema import ChatMessage
from ...schemas.tools_schema import Tool, ToolCall


class BaseToolsHandler(ABC):
    """Base class for handling tools functionality in different models."""

    @abstractmethod
    def encode_tools(
        self,
        conversation: List[ChatMessage],
        tools: Optional[List[ToolCall]] = None,
        **kwargs,
    ) -> str:
        """Encode tools into the prompt format expected by the model.

        Args:
            conversation: List of chat messages
            tools: Optional list of tools to include
            **kwargs: Additional arguments passed to the template

        Returns:
            str: The formatted prompt including tools information
        """
        pass

    @abstractmethod
    def decode_tool_calls(self, text: str) -> Optional[List[ToolCall]]:
        """Decode tool calls from model output.

        Args:
            text: The text output from the model

        Returns:
            Optional[List[ToolCall]]: List of tool calls if found, None otherwise
        """
        pass


class DefaultToolsHandler(BaseToolsHandler):
    """Default implementation of tools handler."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode_tools(
        self,
        conversation: List[ChatMessage],
        tools: Optional[List[Tool]] = None,
        **kwargs,
    ) -> str:
        """Default implementation using tokenizer's chat template."""
        schema_tools = None
        if tools:
            # Convert tools to dict using model_dump
            schema_tools = [tool.model_dump(exclude_none=True) for tool in tools]

        return self.tokenizer.apply_chat_template(
            conversation=conversation,
            tools=schema_tools,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )

    def decode_tool_calls(self, text: str) -> Optional[List[ToolCall]]:
        """Parse tool calls from Llama model output.

        The model outputs function calls in the format:
        {"name": "function_name", "parameters": {"param1": "value1", ...}}
        """
        # Look for JSON patterns in the text
        json_pattern = r'\{[^{}]*"name":\s*"[^"]+"\s*,\s*"parameters":\s*\{[^{}]+\}\}'
        matches = re.finditer(json_pattern, text)

        tool_calls = []
        for match in matches:
            try:
                # Parse the original format
                tool_data = json.loads(match.group())

                # Create tool call using the helper method
                tool_calls.append(
                    ToolCall.from_llama_output(
                        name=tool_data["name"],
                        parameters=tool_data.get("parameters", {}),
                        call_id=f"call_{uuid.uuid4().hex[:8]}",
                    )
                )
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing tool call: {e}")
                continue

        return tool_calls if tool_calls else None
