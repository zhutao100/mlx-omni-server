import json
import re
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, Type

from transformers import PreTrainedTokenizer

from ...schemas.chat_schema import ChatMessage
from ...schemas.tools_schema import Tool, ToolCall


class BaseToolsHandler(ABC):
    """Base class for tools handlers."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def encode_tools(
        self,
        conversation: List[ChatMessage],
        tools: Optional[List[Tool]] = None,
        **kwargs,
    ) -> str:
        """Encode tools and conversation into a prompt string.

        This is a common implementation that uses the tokenizer's chat template.
        Subclasses can override this if they need different behavior.
        """
        schema_tools = None
        if tools:
            schema_tools = [tool.model_dump(exclude_none=True) for tool in tools]

        return self.tokenizer.apply_chat_template(
            conversation=conversation,
            tools=schema_tools,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )

    @abstractmethod
    def decode_tool_calls(self, text: str) -> Optional[List[ToolCall]]:
        """Parse tool calls from model output."""
        pass


class LlamaToolsHandler(BaseToolsHandler):
    """Tools handler for Llama models."""

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


def load_tools_handler(
    model_id: str, tokenizer: PreTrainedTokenizer
) -> BaseToolsHandler:
    """Factory function to load appropriate tools handler based on model ID."""
    handlers: dict[str, Type[BaseToolsHandler]] = {
        # Llama models
        "mlx-community/Llama-3.2-3B-Instruct-4bit": LlamaToolsHandler,
        "mlx-community/Llama-2-7b-chat-mlx-4bit": LlamaToolsHandler,
    }

    # Get handler class based on model ID or use Llama handler as default
    handler_class = handlers.get(model_id, LlamaToolsHandler)
    return handler_class(tokenizer)
