from typing import Type

from ..tools.chat_tokenizer import ChatTokenizer
from ..tools.glm4 import Glm4ChatTokenizer
from ..tools.hugging_face import HuggingFaceChatTokenizer
from ..tools.llama3 import Llama3ChatTokenizer
from ..tools.minimax_m2 import MinimaxM2ChatTokenizer
from ..tools.mistral import MistralChatTokenizer
from ..tools.qwen3 import Qwen3ChatTokenizer
from ..tools.seed_oss import SeedOssChatTokenizer


def load_tools_handler(model_type: str, tokenizer) -> ChatTokenizer:
    """Factory function to load appropriate tools handler based on model ID."""
    handlers: dict[str, Type[ChatTokenizer]] = {
        # Llama models
        "llama": Llama3ChatTokenizer,
        "mistral": MistralChatTokenizer,
        "qwen2": HuggingFaceChatTokenizer,
        "qwen3": Qwen3ChatTokenizer,
        "qwen3_moe": Qwen3ChatTokenizer,
        "glm4": Glm4ChatTokenizer,
        "glm4_moe": Glm4ChatTokenizer,
        "glm4_moe_lite": Glm4ChatTokenizer,
        "minimax_m2": MinimaxM2ChatTokenizer,
        "seed_oss": SeedOssChatTokenizer
    }

    # Get handler class based on model ID or use Llama handler as default
    handler_class = handlers.get(model_type, HuggingFaceChatTokenizer)
    return handler_class(tokenizer)
