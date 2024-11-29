import json
import re
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, Type

from mlx_lm.tokenizer_utils import TokenizerWrapper

from ...schemas.chat_schema import ChatMessage, Role
from ...schemas.tools_schema import Tool, ToolCall
from ...utils.logger import logger


class ChatTokenizer(ABC):
    """Base class for tools handlers."""

    start_tool_calls: str
    end_tool_calls: str

    def __init__(self, tokenizer: TokenizerWrapper):
        self.tokenizer = tokenizer

    def encode(
        self,
        messages: List[ChatMessage],
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

        prompt = self.tokenizer.apply_chat_template(
            conversation=messages,
            tools=schema_tools,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )

        return prompt

    @abstractmethod
    def decode_stream(self, text: str) -> Optional[List[ToolCall]]:
        """Parse tool calls from model output."""
        pass

    @abstractmethod
    def decode(self, text: str) -> Optional[ChatMessage]:
        """Parse tool calls from model output."""
        pass


class LlamaChatTokenizer(ChatTokenizer):
    """Tools handler for Llama models."""

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(tokenizer)
        self.start_tool_calls = "<|python_tag|>"
        self.end_tool_calls = "<|eom_id|>"

    def decode_stream(self, text: str, delta_text: str) -> Optional[List[ToolCall]]:
        pass

    def decode(self, text: str) -> Optional[ChatMessage]:
        """Parse tool calls from Llama model output.

        The model outputs function calls in the format:
        {"name": "function_name", "parameters": {"param1": "value1", ...}}
        """
        # Look for JSON patterns in the text
        tool_calls = []
        logger.debug(f"text:\n{text}")

        if text.startswith(self.start_tool_calls):
            json_pattern = (
                r'\{[^{}]*"name":\s*"[^"]+"\s*,\s*"parameters":\s*\{[^{}]+\}\}'
            )
            matches = re.finditer(json_pattern, text)

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

        return ChatMessage(
            role=Role.ASSISTANT,
            content=None if tool_calls else text,
            tool_calls=tool_calls if tool_calls else None,
        )


def load_tools_handler(model_id: str, tokenizer: TokenizerWrapper) -> ChatTokenizer:
    """Factory function to load appropriate tools handler based on model ID."""
    handlers: dict[str, Type[ChatTokenizer]] = {
        # Llama models
        "mlx-community/Llama-3.2-3B-Instruct-4bit": LlamaChatTokenizer,
        "mlx-community/Llama-2-7b-chat-mlx-4bit": LlamaChatTokenizer,
    }

    # Get handler class based on model ID or use Llama handler as default
    handler_class = handlers.get(model_id, LlamaChatTokenizer)
    return handler_class(tokenizer)
