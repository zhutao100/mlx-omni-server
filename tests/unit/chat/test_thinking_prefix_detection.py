from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mlx_omni_server.chat.mlx_lm.mlx_lm_model import MlxLmModel
from mlx_omni_server.chat.mlx_lm.prompt_cache import PromptCache
from mlx_omni_server.chat.schema import ChatCompletionRequest, ChatMessage, Role


def test_mlx_lm_prepare_generation_detects_prefilled_think_with_trailing_newline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_tokenizer = Mock()
    mock_tokenizer.encode.return_value = [1, 2, 3]

    model_cache = SimpleNamespace(
        model_type="qwen3_5",
        tokenizer=mock_tokenizer,
        model_config={"max_position_embeddings": 128},
        model_id=SimpleNamespace(name="model"),
        model=object(),
        draft_model=None,
    )
    model = MlxLmModel(model_cache)

    # Avoid depending on any specific upstream chat template; just force an encoded prompt.
    model._chat_tokenizer.encode = lambda *args, **kwargs: f"PREFIX{model._reasoning_decoder.thinking_start_tag}\n"  # type: ignore[method-assign]

    def fake_get_or_create_cache(_model_cache, prompt_tokens, *, session_key=None):  # noqa: ANN001
        cache = PromptCache(max_position_embeddings=128)
        cache.model_key = "model"
        cache.cache = [object()]
        return cache, prompt_tokens, 0

    model._prompt_cache_manager.get_or_create_cache = fake_get_or_create_cache  # type: ignore[method-assign]

    monkeypatch.setattr(
        "mlx_omni_server.chat.mlx_lm.mlx_lm_model.build_logits_processors",
        lambda *_args, **_kwargs: None,
    )

    request = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role=Role.USER, content="hi")],
    )

    model._prepare_generation(request)

    assert model._reasoning_decoder.add_thinking_prefix is True
