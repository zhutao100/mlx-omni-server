import gc
from threading import Lock

from mlx.core import clear_cache

from ...utils.logger import logger
from ..models.models_service import ModelId
from ..text_models import BaseTextModel
from .mlx_vlm_model import MlxVlmModel
from .model_types import MlxVlmModelCache


class MlxVlmModelCacheManager:
    """Singleton class that manages lifecycle of MlxVlmModelCache and MlxVlmModel."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._model_cache: MlxVlmModelCache | None = None
        self._mlx_model: MlxVlmModel | None = None
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
                self._model_cache = MlxVlmModelCache(model_id)
                self._mlx_model = MlxVlmModel(model_cache=self._model_cache)
            else:
                if not self._mlx_model:
                    logger.error("Unexpected: model cache exists but MlxVlmModel is missing.")
                    self._model_cache = MlxVlmModelCache(model_id)
                    self._mlx_model = MlxVlmModel(model_cache=self._model_cache)

            return self._mlx_model

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
model_cache_manager = MlxVlmModelCacheManager()


def load_model(model_id: ModelId) -> BaseTextModel:
    """Module-level wrapper for convenience."""
    return model_cache_manager.load_model(model_id)