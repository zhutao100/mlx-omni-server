import importlib
import json
import os
from dataclasses import dataclass, field
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import mlx.nn as nn
from huggingface_hub import CachedRepoInfo, scan_cache_dir
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import MODEL_REMAPPING as LM_MODEL_REMAPPING
from mlx_lm.utils import get_model_path as lm_get_model_path
from mlx_lm.utils import load as lm_load
from mlx_lm.utils import load_config as lm_load_config
from mlx_vlm.utils import MODEL_REMAPPING as VLM_MODEL_REMAPPING
from mlx_vlm.utils import load as vlm_load
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ...utils.file_loader import get_project_root
from ...utils.logger import logger
from .schema import Model, ModelDeletion, ModelList

# Combine model remappings from both LM and VLM
MODEL_REMAPPING = {**LM_MODEL_REMAPPING, **VLM_MODEL_REMAPPING}
# Constants for module names
MLX_LM_MODULE = "mlx_lm.models"
MLX_VLM_MODULE = "mlx_vlm.models"
# Type aliases
ModelPath = Path
ModelConfig = Dict[str, Any]
ModelType = str
ModelModule = nn.Module
TokenizerType = Union[TokenizerWrapper, PreTrainedTokenizerBase]

# More accurate function type definitions
LoaderFunc = Callable[..., Tuple[ModelModule, TokenizerType]]
ChatTemplateFunc = Callable[[str], Optional[str]]


def load_lm_chat_template(model_type: str) -> str | None:
    """Load chat template based on model type for LM models."""
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


def load_vlm_chat_template(model_type: str) -> str | None:
    """Load chat template based on model type for VLM models."""
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


def _is_model_supported_by_module(raw_model_type: str, module_name: str) -> bool:
    """Check if a model type is supported by a given module."""
    if not raw_model_type:
        return False
    normalized_type = str(raw_model_type).lower()
    model_type = MODEL_REMAPPING.get(normalized_type, normalized_type)
    return importlib_util.find_spec(f"{module_name}.{model_type}") is not None


def is_model_supported(raw_model_type: str) -> bool:
    """Check if a model type is supported by either mlx-lm or mlx-vlm."""
    return any(
        _is_model_supported_by_module(raw_model_type, module)
        for module in (MLX_LM_MODULE, MLX_VLM_MODULE)
    )


@dataclass
class ModelId:
    """Identifier for a model with optional adapter and draft model paths."""
    name: str
    adapter_path: Optional[str] = None
    draft_model: Optional[str] = None


@dataclass
class MlxModelCache:
    """Unified model cache for LM and VLM models."""
    model_id: ModelId
    model_path: ModelPath = field(init=False)
    model_config: ModelConfig = field(init=False)
    model_type: ModelType = field(init=False)
    model: ModelModule = field(init=False)
    tokenizer: TokenizerType = field(init=False)
    draft_model: Optional[ModelModule] = None
    draft_tokenizer: Optional[TokenizerType] = None

    def __post_init__(self) -> None:
        self._load_model()

    def _setup_tokenizer_config(self, load_chat_template_func: ChatTemplateFunc) -> Dict[str, Any]:
        """Set up tokenizer configuration with chat template if available."""
        tokenizer_config: Dict[str, Any] = {"trust_remote_code": True}
        if chat_template := load_chat_template_func(self.model_type):
            logger.info(f"Using chat template {chat_template}")
            tokenizer_config["chat_template"] = chat_template
        return tokenizer_config

    def _load_draft_model(self, load_func: LoaderFunc) -> None:
        """Load draft model if specified."""
        if not self.model_id.draft_model:
            return
        self.draft_model, self.draft_tokenizer = load_func(
            self.model_id.draft_model,
            tokenizer_config={"trust_remote_code": True}
        )
        # Check if both tokenizers are available before comparing vocab sizes
        if (self.draft_tokenizer is not None and
            self.tokenizer is not None and
            hasattr(self.draft_tokenizer, 'vocab_size') and
            hasattr(self.tokenizer, 'vocab_size') and
                self.draft_tokenizer.vocab_size != self.tokenizer.vocab_size):
            logger.warning(
                f"Draft model '{self.model_id.draft_model}' tokenizer does not match main model tokenizer."
            )
        logger.info(f"Loaded draft model: {self.model_id.draft_model}")

    def _load_model_generic(
        self,
        loader_func: LoaderFunc,
        load_chat_template_func: ChatTemplateFunc,
        model_label: str,
    ) -> None:
        """Generic loader for LM/VLM models."""
        logger.info(f"Loading {model_label} model {self.model_id.name}")
        tokenizer_config = self._setup_tokenizer_config(load_chat_template_func)
        self.model, self.tokenizer = loader_func(
            self.model_id.name,
            tokenizer_config=tokenizer_config,
            adapter_path=self.model_id.adapter_path,
        )
        logger.info(f"Loaded {model_label} model: {self.model_id.name}")
        self._load_draft_model(loader_func)

    def _load_model(self) -> None:
        """Decide LM or VLM loader based on model support."""
        self.model_path, _ = lm_get_model_path(self.model_id.name)
        self.model_config = lm_load_config(self.model_path)
        self.model_type = self.model_config["model_type"]

        lm_supported = _is_model_supported_by_module(self.model_type, MLX_LM_MODULE)
        vlm_supported = _is_model_supported_by_module(self.model_type, MLX_VLM_MODULE)

        if vlm_supported and not lm_supported:
            self._load_model_generic(
                vlm_load,
                load_vlm_chat_template,
                "VLM"
            )
        else:  # LM preferred if both or fallback
            if not lm_supported:
                logger.warning(
                    f"Model type {self.model_type} not explicitly supported, attempting LM loader"
                )
            self._load_model_generic(
                lm_load,
                load_lm_chat_template,
                "LM"
            )


class ModelCacheScanner:
    """Scanner for finding and managing mlx-lm compatible models in the local cache."""

    def __init__(self) -> None:
        self._cache_info: Optional[Any] = None

    @property
    def cache_info(self) -> Any:
        if self._cache_info is None:
            self._cache_info = scan_cache_dir()
        return self._cache_info

    def _refresh_cache_info(self) -> None:
        self._cache_info = scan_cache_dir()

    @staticmethod
    def _read_model_config(config_file_path: Path) -> Optional[ModelConfig]:
        try:
            with open(config_file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading config.json: {e}")
            return None

    def _process_repo_info(
        self,
        repo_info: CachedRepoInfo
    ) -> Optional[Tuple[CachedRepoInfo, ModelConfig]]:
        if repo_info.repo_type != "model":
            return None

        first_revision = next(iter(repo_info.revisions), None)
        if not first_revision:
            return None

        config_file = next(
            (f for f in first_revision.files if f.file_name == "config.json"),
            None
        )
        if not config_file:
            return None

        config_data = self._read_model_config(config_file.file_path)
        if not config_data:
            return None

        model_type = config_data.get("model_type")
        if model_type is not None and is_model_supported(model_type):
            return repo_info, config_data

        logger.warning(f"Model {repo_info.repo_id} found but not compatible")
        return None

    def find_models_in_cache(self) -> List[Tuple[CachedRepoInfo, ModelConfig]]:
        return [
            res
            for repo in self.cache_info.repos
            if (res := self._process_repo_info(repo)) is not None
        ]

    def get_model_info(
        self,
        model_id: str
    ) -> Optional[Tuple[CachedRepoInfo, ModelConfig]]:
        for repo_info in self.cache_info.repos:
            if repo_info.repo_id == model_id:
                return self._process_repo_info(repo_info)
        return None

    def delete_model(self, model_id: str) -> bool:
        for repo_info in self.cache_info.repos:
            if repo_info.repo_id == model_id:
                revision_hashes = [rev.commit_hash for rev in repo_info.revisions]
                if not revision_hashes:
                    return False
                try:
                    delete_strategy = self.cache_info.delete_revisions(*revision_hashes)
                    logger.info(
                        f"Model '{model_id}': Will free {delete_strategy.expected_freed_size_str}"
                    )
                    delete_strategy.execute()
                    logger.info(f"Model '{model_id}': Cache deletion completed")
                    self._refresh_cache_info()
                    return True
                except Exception as e:
                    logger.error(f"Error deleting model '{model_id}': {e}")
                    raise
        return False


class ModelsService:
    """Service for managing models in the local cache."""

    def __init__(self) -> None:
        self.scanner = ModelCacheScanner()
        self.available_models = self._scan_models()

    def _scan_models(self) -> List[Tuple[CachedRepoInfo, ModelConfig]]:
        try:
            return self.scanner.find_models_in_cache()
        except Exception as e:
            logger.error(f"Error scanning cache: {e}")
            return []

    @staticmethod
    def _get_model_owner(model_id: str) -> str:
        return model_id.split("/")[0] if "/" in model_id else model_id

    def _create_model_instance(
        self,
        repo_info: CachedRepoInfo,
        config_data: ModelConfig,
        include_details: bool = False
    ) -> Model:
        model_kwargs = {
            "id": repo_info.repo_id,
            "created": int(repo_info.last_modified),
            "owned_by": self._get_model_owner(repo_info.repo_id),
        }
        if include_details:
            model_kwargs["details"] = config_data
        return Model(**model_kwargs)

    def list_models(self, include_details: bool = False) -> ModelList:
        models = [
            self._create_model_instance(repo_info, config_data, include_details)
            for repo_info, config_data in self.available_models
        ]
        return ModelList(data=models)

    def get_model(
        self,
        model_id: str,
        include_details: bool = False
    ) -> Optional[Model]:
        if model_info := self.scanner.get_model_info(model_id):
            repo_info, config_data = model_info
            return self._create_model_instance(repo_info, config_data, include_details)
        return None

    def delete_model(self, model_id: str) -> ModelDeletion:
        if not self.scanner.delete_model(model_id):
            raise ValueError(f"Model '{model_id}' not found in cache")
        self.available_models = self._scan_models()
        return ModelDeletion(id=model_id, deleted=True)
