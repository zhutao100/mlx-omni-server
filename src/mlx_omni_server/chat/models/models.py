import gc
from threading import Lock
from typing import Optional

from mlx.core import clear_cache

from ...utils.logger import logger
from ..mlx_lm.mlx_lm_model import MlxLmModel
from ..mlx_vlm.mlx_vlm_model import MlxVlmModel
from ..text_models import BaseTextModel
from .models_service import MlxModelCache, ModelId


class MlxModelCacheManager:
    """Singleton class that manages lifecycle of MlxModelCache and ensures
    only one model (LM or VLM) is loaded at a time."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._model_cache: MlxModelCache | None = None
        self._mlx_model: BaseTextModel | None = None
        self._lock = Lock()

    def load_model(self, model_id: ModelId) -> BaseTextModel:
        """Load (or reuse) a model and return a BaseTextModel instance."""

        with self._lock:
            if (
                self._model_cache is None
                or self._model_cache.model_id != model_id
            ):
                # Release old models first
                self._release()

                # Create new caches
                self._model_cache = MlxModelCache(model_id)

                # Determine if this is a VLM or LM model based on the model config
                if self._is_vlm_model(self._model_cache):
                    self._mlx_model = MlxVlmModel(model_cache=self._model_cache)
                else:
                    self._mlx_model = MlxLmModel(model_cache=self._model_cache)
            else:
                if not self._mlx_model:
                    logger.error("Unexpected: model cache exists but model is missing.")
                    self._model_cache = MlxModelCache(model_id)
                    # Determine if this is a VLM or LM model based on the model config
                    if self._is_vlm_model(self._model_cache):
                        self._mlx_model = MlxVlmModel(model_cache=self._model_cache)
                    else:
                        self._mlx_model = MlxLmModel(model_cache=self._model_cache)

            return self._mlx_model

    def _is_vlm_model(self, model_cache: MlxModelCache) -> bool:
        """Determine if the model is a VLM model based on its configuration."""
        # Check if model is supported by mlx_vlm but not mlx_lm
        from ..models.models_service import (
            MLX_VLM_MODULE,
            _is_model_supported_by_module,
        )

        vlm_supported = _is_model_supported_by_module(model_cache.model_type, MLX_VLM_MODULE)

        return vlm_supported

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
