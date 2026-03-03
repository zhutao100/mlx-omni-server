import gc
from threading import Lock

from mlx.core import clear_cache

from ...utils.logger import logger
from ..mlx_lm.mlx_lm_model import MlxLmModel
from ..mlx_vlm.mlx_vlm_model import MlxVlmModel
from ..text_models import BaseTextModel
from .models_service import MlxModelCache, ModelId, ModelLabel


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
                logger.info(f"Loading new model with ID: {model_id}")
                # Release old models first
                self._release()

                self._model_cache = MlxModelCache(model_id)
                match self._model_cache.model_label:
                    case ModelLabel.VLM:
                        self._mlx_model = MlxVlmModel(model_cache=self._model_cache)
                    case ModelLabel.LM:
                        self._mlx_model = MlxLmModel(model_cache=self._model_cache)
                    case _:
                        raise ValueError(
                            f"Unexpected model label: {self._model_cache.model_label} for model_id: {model_id}"
                        )
            else:
                if not self._mlx_model:
                    logger.error("Unexpected: model cache exists but model is missing.")
                    self._model_cache = MlxModelCache(model_id)
                    match self._model_cache.model_label:
                        case ModelLabel.VLM:
                            self._mlx_model = MlxVlmModel(model_cache=self._model_cache)
                        case ModelLabel.LM:
                            self._mlx_model = MlxLmModel(model_cache=self._model_cache)
                        case _:
                            raise ValueError(
                                f"Unexpected model label: {self._model_cache.model_label} for model_id: {model_id}"
                            )

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
model_cache_manager = MlxModelCacheManager()


def load_model(model_id: ModelId) -> BaseTextModel:
    """Module-level wrapper for convenience."""
    return model_cache_manager.load_model(model_id)
