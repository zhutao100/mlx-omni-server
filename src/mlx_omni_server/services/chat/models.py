from mlx_lm.utils import load

from .base_models import BaseMLXModel
from .mlx_model import MLXModel


def load_model(model_id: str) -> BaseMLXModel:
    model, tokenizer = load(
        model_id,
        tokenizer_config={"trust_remote_code": True},
    )

    return MLXModel(model, tokenizer)
