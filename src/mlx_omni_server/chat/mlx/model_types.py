from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Optional, Type

import mlx.nn as nn
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import get_model_path, load, load_config


from ...utils.logger import logger
from .tools.chat_tokenizer import ChatTokenizer
from .tools.hugging_face import HuggingFaceChatTokenizer
from .tools.llama3 import Llama3ChatTokenizer
from .tools.mistral import MistralChatTokenizer
from .tools.qwen3 import Qwen3ChatTokenizer
from .tools.glm4 import Glm4ChatTokenizer
from .tools.seed_oss import SeedOssChatTokenizer


def load_tools_handler(model_type: str, tokenizer: TokenizerWrapper) -> ChatTokenizer:
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

    templates_dir = os.path.join(Path(__file__).parent.parent, "templates")
    template_files = {
        "qwen3": "qwen3_chat_template.jinja",
        "qwen3_moe": "qwen3_chat_template.jinja",
    }
    if template_files.get(model_type):
        template_path = os.path.join(templates_dir, template_files[model_type])
        if os.path.exists(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            logger.error(f"Chat template file not found: {template_path}")

    return None


@dataclass(frozen=True)
class ModelId:
    """Entity class representing a unique model identifier.

    This class encapsulates all parameters that determine whether a model
    needs to be reloaded. It replaces the tuple-based approach with a more
    object-oriented design that's easier to extend and maintain.
    """

    name: str
    adapter_path: Optional[str] = None
    draft_model: Optional[str] = None

    def __str__(self) -> str:
        """Return a string representation of the model ID for debugging."""
        parts = [f"model_name={self.name}"]
        if self.adapter_path:
            parts.append(f"adapter_path={self.adapter_path}")
        if self.draft_model:
            parts.append(f"draft_model={self.draft_model}")
        return f"ModelId({', '.join(parts)})"


class MlxModelCache:
    """Model cache class to avoid reloading the same models.

    This class manages the cache of main models and draft models based on ModelId objects.
    """

    def __init__(self, model_id: ModelId):
        """Initialize the cache object.

        Args:
            model_id: Optional ModelId object for initialization
        """
        self.model_id: ModelId = model_id
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[TokenizerWrapper] = None
        self.chat_tokenizer: Optional[ChatTokenizer] = None
        self.draft_model: Optional[nn.Module] = None
        self.draft_tokenizer: Optional[TokenizerWrapper] = None

        # If a model ID is provided, load the models directly
        if model_id:
            self._load_models()

    def _load_models(self):
        """Load the main model and draft model (if needed)."""
        # Load the model configuration
        model_path = get_model_path(self.model_id.name)[0]
        config = load_config(model_path)

        tokenizer_config: dict[str, Any] = {"trust_remote_code": True}
        chat_template = load_chat_template(config["model_type"])
        if chat_template:
            logger.info(f"Using chat template \n{chat_template}\n")
            tokenizer_config["chat_template"] = chat_template

        # Load the main model
        self.model, self.tokenizer = load(
            self.model_id.name,
            tokenizer_config=tokenizer_config,
            adapter_path=self.model_id.adapter_path,
        )
        logger.info(f"Loaded new model: {self.model_id.name}")

        # create chat tokenizer
        self.chat_tokenizer = load_tools_handler(config["model_type"], self.tokenizer)

        # If needed, load the draft model
        if self.model_id.draft_model:
            self.draft_model, self.draft_tokenizer = load(
                self.model_id.draft_model,
                tokenizer_config={"trust_remote_code": True},
            )

            # Check if vocabulary sizes match
            if self.draft_tokenizer.vocab_size != self.tokenizer.vocab_size:
                logger.warn(
                    f"Draft model({self.model_id.draft_model}) tokenizer does not match model tokenizer."
                )

            logger.info(f"Loaded new draft model: {self.model_id.draft_model}")
