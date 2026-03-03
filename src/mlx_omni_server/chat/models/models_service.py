import json
from dataclasses import dataclass, field
from enum import Enum, auto
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import mlx.nn as nn
from huggingface_hub import CachedRepoInfo, scan_cache_dir, snapshot_download
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import MODEL_REMAPPING as LM_MODEL_REMAPPING
from mlx_lm.utils import hf_repo_to_path
from mlx_lm.utils import load as lm_load
from mlx_vlm.utils import MODEL_REMAPPING as VLM_MODEL_REMAPPING
from mlx_vlm.utils import load as vlm_load
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ...utils.file_loader import read_package_text
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
TokenizerType = Union[TokenizerWrapper, PreTrainedTokenizerBase, Any]

# More accurate function type definitions
LoaderFunc = Callable[..., Tuple[ModelModule, TokenizerType]]
ChatTemplateFunc = Callable[[str], Optional[str]]

_CHAT_TEMPLATES_BASE = ("chat", "templates")


class ModelLabel(Enum):
    LM = auto()
    VLM = auto()


def _load_chat_template_file(template_filename: str) -> str | None:
    resource_parts = (*_CHAT_TEMPLATES_BASE, template_filename)
    resource_identifier = "/".join(resource_parts)
    try:
        return read_package_text(*resource_parts)
    except FileNotFoundError:
        logger.error(f"Chat template file not found: {resource_identifier}")
    except OSError as exc:
        logger.error(f"Unable to read chat template '{resource_identifier}': {exc}")
    return None


def load_lm_chat_template(model_type: str) -> str | None:
    """Load chat template based on model type for LM models."""
    template_files = {
        "qwen3": "qwen3_chat_template.jinja",
        "qwen3_moe": "qwen3_chat_template.jinja",
        "qwen3_5": "qwen3_5_chat_template.jinja",
        "qwen3_5_moe": "qwen3_5_chat_template.jinja",
        "glm4": "glm4_chat_template.jinja",
        "glm4_moe": "glm4_chat_template.jinja",
        "glm4_moe_lite": "glm4_chat_template.jinja",
    }
    template_name = template_files.get(model_type)
    if not template_name:
        return None
    return _load_chat_template_file(template_name)


def load_vlm_chat_template(model_type: str) -> str | None:
    """Load chat template based on model type for VLM models."""
    template_files = {
        "glm4v_moe": "glm4v_chat_template.jinja",
        "qwen3_5": "qwen3_5_chat_template.jinja",
        "qwen3_5_moe": "qwen3_5_chat_template.jinja",
    }
    template_name = template_files.get(model_type)
    if not template_name:
        return None
    return _load_chat_template_file(template_name)


def _read_model_config_file(config_path: Path) -> Optional[ModelConfig]:
    """Read a config.json file and return the parsed contents."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
    except json.JSONDecodeError as exc:
        logger.error(f"Invalid JSON in config file {config_path}: {exc}")
    return None


def _config_from_cache(model_id: str) -> Optional[Tuple[ModelPath, ModelConfig]]:
    """Attempt to resolve a cached model path and config for the given model id."""
    try:
        cache_info = scan_cache_dir()
    except Exception as exc:
        logger.warning(f"Unable to scan Hugging Face cache: {exc}")
        return None

    for repo_info in cache_info.repos:
        if repo_info.repo_id != model_id:
            continue
        for revision in repo_info.revisions:
            config_file = next(
                (f for f in revision.files if f.file_name == "config.json"),
                None,
            )
            if not config_file:
                continue
            config_path = Path(config_file.file_path)
            if config_data := _read_model_config_file(config_path):
                return config_path.parent, config_data
    return None


def _resolve_model_path_and_config(model_name: str) -> Tuple[ModelPath, ModelConfig]:
    """Resolve the local path and configuration for a model."""
    candidate_path = Path(model_name)
    if candidate_path.exists():
        config_path = candidate_path / "config.json"
        if config_data := _read_model_config_file(config_path):
            return candidate_path, config_data
        raise FileNotFoundError(f"Config not found at {config_path}")

    if cached := _config_from_cache(model_name):
        return cached

    try:
        repo_path = hf_repo_to_path(model_name)
        config_path = repo_path / "config.json"
        if config_data := _read_model_config_file(config_path):
            return repo_path, config_data
    except Exception as exc:
        logger.debug(f"Model '{model_name}' not found in local HF cache: {exc}")

    try:
        snapshot_path = Path(snapshot_download(repo_id=model_name, allow_patterns=["config.json"]))
        config_path = snapshot_path / "config.json"
        if config_data := _read_model_config_file(config_path):
            return snapshot_path, config_data
    except Exception as exc:
        logger.error(f"Failed to retrieve config for model '{model_name}': {exc}")

    raise FileNotFoundError(
        f"Unable to locate config.json for model '{model_name}'. "
        "Ensure the model is downloaded or available locally."
    )


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
    model_label: ModelLabel = field(init=False)

    def __post_init__(self) -> None:
        self._load_model()

    def _setup_tokenizer_config(
        self, load_chat_template_func: ChatTemplateFunc
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Set up tokenizer configuration with chat template if available."""
        tokenizer_config: Dict[str, Any] = {"trust_remote_code": True}
        chat_template = load_chat_template_func(self.model_type)
        if chat_template:
            logger.debug("Applying custom chat template for model type '%s'", self.model_type)
            tokenizer_config["chat_template"] = chat_template
        return tokenizer_config, chat_template

    def _load_draft_model(
        self,
        load_func: LoaderFunc,
        tokenizer_config: Dict[str, Any],
    ) -> None:
        """Load draft model if specified."""
        if not self.model_id.draft_model:
            return
        draft_kwargs: Dict[str, Any] = {}
        if load_func is lm_load:
            draft_kwargs["tokenizer_config"] = dict(tokenizer_config)
        else:
            draft_kwargs["trust_remote_code"] = True
        self.draft_model, self.draft_tokenizer = load_func(
            self.model_id.draft_model,
            **draft_kwargs,
        )
        # Check if both tokenizers are available before comparing vocab sizes
        if (
            self.draft_tokenizer is not None
            and self.tokenizer is not None
            and hasattr(self.draft_tokenizer, "vocab_size")
            and hasattr(self.tokenizer, "vocab_size")
            and self.draft_tokenizer.vocab_size != self.tokenizer.vocab_size
        ):
            logger.warning(
                f"Draft model '{self.model_id.draft_model}' tokenizer does not match main model tokenizer."
            )
        logger.info(f"Loaded draft model: {self.model_id.draft_model}")

    def _load_model_generic(
        self,
        loader_func: LoaderFunc,
        load_chat_template_func: ChatTemplateFunc,
    ) -> None:
        """Generic loader for LM/VLM models."""
        logger.info(f"Loading {self.model_label.name} model: {self.model_id.name}")
        tokenizer_config, chat_template = self._setup_tokenizer_config(load_chat_template_func)

        loader_kwargs: Dict[str, Any] = {"adapter_path": self.model_id.adapter_path}
        if loader_func is lm_load:
            loader_kwargs["tokenizer_config"] = tokenizer_config
        else:
            loader_kwargs["trust_remote_code"] = True

        self.model, self.tokenizer = loader_func(
            self.model_id.name,
            **loader_kwargs,
        )
        logger.info(f"Loaded {self.model_label.name} model: {self.model_id.name}")
        self._load_draft_model(loader_func, tokenizer_config)

    def _load_model(self) -> None:
        """Decide LM or VLM loader based on model support."""
        self.model_path, self.model_config = _resolve_model_path_and_config(self.model_id.name)
        raw_model_type = self.model_config.get("model_type")
        if raw_model_type is None:
            logger.warning(
                "Model config for '%s' does not define 'model_type'.",
                self.model_id.name,
            )
            self.model_type = ""
        else:
            self.model_type = str(raw_model_type).lower()

        lm_supported = _is_model_supported_by_module(self.model_type, MLX_LM_MODULE)
        vlm_supported = _is_model_supported_by_module(self.model_type, MLX_VLM_MODULE)

        if vlm_supported:  # Prefer VLM loader if both claim support, as VLMs are more specialized
            self.model_label = ModelLabel.VLM
            self._load_model_generic(vlm_load, load_vlm_chat_template)
        else:
            if not lm_supported:
                logger.warning(
                    "Model type '%s' not explicitly supported, attempting LM loader",
                    self.model_type or raw_model_type,
                )
            self.model_label = ModelLabel.LM
            self._load_model_generic(lm_load, load_lm_chat_template)  # type: ignore


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
        self, repo_info: CachedRepoInfo
    ) -> Optional[Tuple[CachedRepoInfo, ModelConfig]]:
        if repo_info.repo_type != "model":
            return None

        first_revision = next(iter(repo_info.revisions), None)
        if not first_revision:
            return None

        config_file = next((f for f in first_revision.files if f.file_name == "config.json"), None)
        if not config_file:
            return None

        config_data = self._read_model_config(config_file.file_path)
        if not config_data:
            return None

        model_type = config_data.get("model_type")
        if model_type is not None and is_model_supported(model_type):
            return repo_info, config_data

        return None

    def find_models_in_cache(self) -> List[Tuple[CachedRepoInfo, ModelConfig]]:
        return [
            res
            for repo in self.cache_info.repos
            if (res := self._process_repo_info(repo)) is not None
        ]

    def get_model_info(self, model_id: str) -> Optional[Tuple[CachedRepoInfo, ModelConfig]]:
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
        include_details: bool = False,
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
