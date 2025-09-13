import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Type

import mlx.nn as nn
from mlx.core import clear_cache
from mlx_vlm import load
from mlx_vlm.utils import get_model_path, load_config
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ...utils.logger import logger
from ..models.models_service import ModelId
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

    templates_dir = os.path.join(Path(__file__).parent.parent, "templates")
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


@dataclass
class MlxVlmModelCache:
    """Cache for loaded VLM models and processors."""
    model_id: ModelId
    model: Optional[nn.Module] = None
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    chat_tokenizer: Optional[ChatTokenizer] = None
    draft_model: Optional[nn.Module] = None
    draft_tokenizer: Optional[PreTrainedTokenizerBase] = None

    def __post_init__(self):
        """Initialize and load the model."""
        self._load_model()

    def _load_model(self):
        """Load the VLM model and processor."""
        model_path = get_model_path(self.model_id.name)
        config = load_config(model_path)
        logger.info(f"Loading model {self.model_id.name} from {model_path}")
        logger.debug(f"Model config: {config}")
        tokenizer_config: dict[str, Any] = {"trust_remote_code": True}
        chat_template = load_chat_template(config["model_type"])
        if chat_template:
            logger.info(f"Using chat template \n{chat_template}\n")
            tokenizer_config["chat_template"] = chat_template

        self.model, self.tokenizer = load(
            self.model_id.name,
            tokenizer_config=tokenizer_config,
            adapter_path=self.model_id.adapter_path,
            draft_model=self.model_id.draft_model
        )

        self.chat_tokenizer = load_tools_handler(config["model_type"], self.tokenizer)

        # If needed, load the draft model
        if self.model_id.draft_model:
            self.draft_model, self.draft_tokenizer = load(
                self.model_id.draft_model,
                tokenizer_config={"trust_remote_code": True},
            )
            # Check if vocabulary sizes match
            if self.draft_tokenizer.vocab_size != self.tokenizer.vocab_size:
                logger.warning(
                    f"Draft model({self.model_id.draft_model}) tokenizer does not match model tokenizer."
                )
            logger.info(f"Loaded new draft model: {self.model_id.draft_model}")
