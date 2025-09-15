import os
from pathlib import Path
from typing import Type

from mlx_lm.tokenizer_utils import TokenizerWrapper

from ...utils.file_loader import get_project_root
from ...utils.logger import logger
from ..tools.chat_tokenizer import ChatTokenizer
from ..tools.glm4 import Glm4ChatTokenizer
from ..tools.hugging_face import HuggingFaceChatTokenizer
from ..tools.llama3 import Llama3ChatTokenizer
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
        "seed_oss": SeedOssChatTokenizer
    }

    # Get handler class based on model ID or use Llama handler as default
    handler_class = handlers.get(model_type, HuggingFaceChatTokenizer)
    return handler_class(tokenizer)


def load_chat_template(model_type: str) -> str | None:
    """Load chat template based on model type."""

    templates_dir = os.path.join(get_project_root(), "src/mlx_omni_server/chat/templates")
    template_files = {
        "qwen3": "qwen3_chat_template.jinja",
        "qwen3_moe": "qwen3_chat_template.jinja",
        "glm4": "glm4_chat_template.jinja",
        "glm4_moe": "glm4_chat_template.jinja",

    }
    if template_files.get(model_type):
        template_path = os.path.join(templates_dir, template_files[model_type])
        if os.path.exists(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            logger.error(f"Chat template file not found: {template_path}")

    return None
