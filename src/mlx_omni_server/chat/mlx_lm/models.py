import gc
from threading import Lock

from mlx.core import clear_cache

from ...utils.logger import logger
from ..models.models_service import ModelId
from ..text_models import BaseTextModel
from .mlx_lm_model import MlxLmModel
from .model_types import MlxLmModelCache


class MlxLmModelCacheManager:
    """Singleton class that manages lifecycle of MlxLmModelCache and MlxLmModel."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._model_cache: MlxLmModelCache | None = None
        self._mlx_model: MlxLmModel | None = None
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
                self._model_cache = MlxLmModelCache(model_id)
                self._mlx_model = MlxLmModel(model_cache=self._model_cache)
            else:
                if not self._mlx_model:
                    logger.error("Unexpected: model cache exists but MlxLmModel is missing.")
                    self._model_cache = MlxLmModelCache(model_id)
                    self._mlx_model = MlxLmModel(model_cache=self._model_cache)

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
model_cache_manager = MlxLmModelCacheManager()


def load_model(model_id: ModelId) -> BaseTextModel:
    """Module-level wrapper for convenience."""
    return model_cache_manager.load_model(model_id)
