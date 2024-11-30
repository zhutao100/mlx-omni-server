from typing import Type

from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load

from .base_models import BaseMLXModel
from .mlx_model import MLXModel
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


def load_model(model_id: str) -> BaseMLXModel:
    """Load a model and tokenizer from the given model ID."""
    model, tokenizer = load(
        model_id,
        tokenizer_config={"trust_remote_code": True},
    )

    chat_tokenizer = load_tools_handler(model_id, tokenizer)

    return MLXModel(model_id=model_id, model=model, tokenizer=chat_tokenizer)
