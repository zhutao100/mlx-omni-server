from typing import Type

from mlx_lm.tokenizer_utils import TokenizerWrapper

from .tools.chat_tokenizer import ChatTokenizer
from .tools.llama3 import LlamaChatTokenizer
from .tools.mistral import MistralChatTokenizer
from .tools.qwen2 import Qwen2ChatTokenizer


def load_tools_handler(model_id: str, tokenizer: TokenizerWrapper) -> ChatTokenizer:
    """Factory function to load appropriate tools handler based on model ID."""
    handlers: dict[str, Type[ChatTokenizer]] = {
        # Llama models
        "mlx-community/Llama-3.2-3B-Instruct-4bit": LlamaChatTokenizer,
        "mlx-community/Llama-2-7b-chat-mlx-4bit": LlamaChatTokenizer,
        "mistralai/Mistral-7B-Instruct-v0.3": MistralChatTokenizer,
        "mlx-community/Qwen2.5-14B-Instruct-4bit": Qwen2ChatTokenizer,
    }

    # Get handler class based on model ID or use Llama handler as default
    handler_class = handlers.get(model_id, LlamaChatTokenizer)
    return handler_class(tokenizer)
