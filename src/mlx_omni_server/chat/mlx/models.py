from threading import Lock
import gc
from mlx.core import clear_cache

from ..text_models import BaseTextModel
from .mlx_model import MLXModel
from .model_types import MlxModelCache, ModelId

from ...utils.logger import logger

# Initialize global cache objects
_model_cache = None
_mlx_model_cache = None


class ModelCacheManager:
    """Manages lifecycle of MlxModelCache and MLXModel."""

    def __init__(self):
        self._model_cache: MlxModelCache | None = None
        self._mlx_model_cache: MLXModel | None = None
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
                self._mlx_model_cache = MLXModel(model_cache=self._model_cache)
            else:
                if not self._mlx_model_cache:
                    logger.error("Unexpected: model cache exists but MLXModel is missing.")
                    self._model_cache = MlxModelCache(model_id)
                    self._mlx_model_cache = MLXModel(model_cache=self._model_cache)

            return self._mlx_model_cache

    def _release(self):
        """Release current models and force memory cleanup."""
        self._model_cache = None
        self._mlx_model_cache = None
        clear_cache()
        gc.collect()

    def clear(self):
        """Public method to clear cache (e.g., in tests)."""
        with self._lock:
            self._release()


# Create a single shared instance
model_cache_manager = ModelCacheManager()


def load_model(model_id: ModelId) -> BaseTextModel:
    """Module-level wrapper for convenience."""
    return model_cache_manager.load_model(model_id)
