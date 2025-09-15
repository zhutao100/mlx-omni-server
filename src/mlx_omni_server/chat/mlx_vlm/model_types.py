import os
from pathlib import Path
from typing import Type

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ...utils.file_loader import get_project_root
from ...utils.logger import logger
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


def load_chat_template(model_type: str) -> str | None:
    """Load chat template based on model type."""

    templates_dir = os.path.join(get_project_root(), "src/mlx_omni_server/chat/templates")
    template_files = {
        "glm4v_moe": "glm4v_chat_template.jinja",

    }
    if template_files.get(model_type):
        template_path = os.path.join(templates_dir, template_files[model_type])
        if os.path.exists(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            logger.error(f"Chat template file not found: {template_path}")

    return None