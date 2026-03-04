import gc
from threading import Lock

from mlx.core import clear_cache

from ...utils.logger import logger
from ..mlx_lm.mlx_lm_model import MlxLmModel
from ..mlx_vlm.mlx_vlm_model import MlxVlmModel
from ..text_models import BaseTextModel
from .models_service import MlxModelCache, ModelId, ModelLabel


def _normalize_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_model_id(model_id: ModelId) -> ModelId:
    name = model_id.name.strip()
    if not name:
        raise ValueError("model_id.name cannot be empty")
    return ModelId(
        name=name,
        adapter_path=_normalize_optional_str(model_id.adapter_path),
        draft_model=_normalize_optional_str(model_id.draft_model),
    )


class MlxModelCacheManager:
    """Singleton class that manages lifecycle of MlxModelCache and ensures
    only one model (LM or VLM) is loaded at a time."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._model_cache: MlxModelCache | None = None
        self._mlx_model: BaseTextModel | None = None
        self._lock = Lock()
        self._initialized = True

    @staticmethod
    def _construct_model(model_cache: MlxModelCache) -> BaseTextModel:
        match model_cache.model_label:
            case ModelLabel.VLM:
                return MlxVlmModel(model_cache=model_cache)
            case ModelLabel.LM:
                return MlxLmModel(model_cache=model_cache)
            case _:
                raise ValueError(
                    f"Unexpected model label: {model_cache.model_label} for model_id: {model_cache.model_id}"
                )

    def load_model(self, model_id: ModelId) -> BaseTextModel:
        """Load (or reuse) a model and return a BaseTextModel instance."""

        with self._lock:
            normalized_model_id = _normalize_model_id(model_id)

            if (
                self._model_cache is not None
                and self._mlx_model is not None
                and self._model_cache.model_id == normalized_model_id
            ):
                logger.debug("Reusing existing model with ID: %s", normalized_model_id)
                return self._mlx_model

            if self._model_cache is not None:
                if self._model_cache.model_id == normalized_model_id:
                    logger.warning(
                        "Model cache exists for ID %s but model instance is missing. Reloading.",
                        normalized_model_id,
                    )
                else:
                    logger.info(
                        "Switching models: %s -> %s",
                        self._model_cache.model_id,
                        normalized_model_id,
                    )
                self._release()
            elif self._mlx_model is not None:
                # Should be impossible, but keep internal state consistent.
                logger.warning(
                    "Model instance exists without a cache; releasing and reloading model_id=%s",
                    normalized_model_id,
                )
                self._release()

            logger.info("Loading model with ID: %s", normalized_model_id)
            try:
                model_cache = MlxModelCache(normalized_model_id)
                mlx_model = self._construct_model(model_cache)
            except Exception:
                # Best-effort cleanup for partially loaded MLX state.
                clear_cache()
                gc.collect()
                raise

            self._model_cache = model_cache
            self._mlx_model = mlx_model
            return mlx_model

    def _release(self):
        """Release current models and force memory cleanup."""

        self._model_cache = None
        self._mlx_model = None
        clear_cache()
        gc.collect()

    def clear(self):
        """Public method to clear cache (e.g., in tests)."""
        with self._lock:
            self._release()


# Create a single shared instance
model_cache_manager = MlxModelCacheManager()


def load_model(model_id: ModelId) -> BaseTextModel:
    """Module-level wrapper for convenience."""
    return model_cache_manager.load_model(model_id)
