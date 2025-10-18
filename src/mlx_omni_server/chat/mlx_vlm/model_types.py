from typing import Type

from ..tools.chat_tokenizer import ChatTokenizer
from ..tools.glm4 import Glm4ChatTokenizer
from ..tools.hugging_face import HuggingFaceChatTokenizer


def load_tools_handler(model_type: str, tokenizer) -> ChatTokenizer:
    """Factory function to load appropriate tools handler based on model ID."""
    handlers: dict[str, Type[ChatTokenizer]] = {
        "glm4v_moe": Glm4ChatTokenizer,
    }

    # Get handler class based on model ID or use Llama handler as default
    handler_class = handlers.get(model_type, HuggingFaceChatTokenizer)
    return handler_class(tokenizer)
