from types import SimpleNamespace
from unittest.mock import patch

from mlx_omni_server.chat.mlx_lm.prompt_cache import PromptCache as LmPromptCache
from mlx_omni_server.chat.mlx_lm.prompt_cache import (
    PromptCacheManager as LmPromptCacheManager,
)
from mlx_omni_server.chat.mlx_lm.prompt_cache import tokens_key as lm_tokens_key
from mlx_omni_server.chat.mlx_vlm.prompt_cache import PromptCache as VlmPromptCache
from mlx_omni_server.chat.mlx_vlm.prompt_cache import (
    PromptCacheManager as VlmPromptCacheManager,
)
from mlx_omni_server.chat.mlx_vlm.prompt_cache import tokens_key as vlm_tokens_key


def test_lm_prompt_cache_manager_filters_by_session_key() -> None:
    manager = LmPromptCacheManager(max_position_embeddings=4096, max_caches=10)
    prompt = list(range(200))

    cache_a = LmPromptCache(
        max_position_embeddings=4096,
        session_key="sess-a",
        tokens=prompt[:150],
        model_key="model",
    )
    cache_b = LmPromptCache(
        max_position_embeddings=4096,
        session_key="sess-b",
        tokens=prompt[:180],
        model_key="model",
    )
    manager.caches[("sess-a", "a")] = cache_a
    manager.caches[("sess-b", "b")] = cache_b

    model_cache = SimpleNamespace(
        model_id=SimpleNamespace(name="model"),
        model=None,
        draft_model=None,
    )

    selected_a, _suffix_a, cached_a = manager.get_or_create_cache(
        model_cache,
        prompt,
        session_key="sess-a",
    )
    assert selected_a is cache_a
    assert cached_a == 150

    selected_b, _suffix_b, cached_b = manager.get_or_create_cache(
        model_cache,
        prompt,
        session_key="sess-b",
    )
    assert selected_b is cache_b
    assert cached_b == 180


def test_lm_prompt_cache_manager_uses_session_key_in_cache_key() -> None:
    manager = LmPromptCacheManager(max_position_embeddings=4096, max_caches=10)
    prompt = list(range(32))

    model_cache = SimpleNamespace(
        model_id=SimpleNamespace(name="model"),
        model=object(),
        draft_model=None,
    )

    with patch("mlx_omni_server.chat.mlx_lm.prompt_cache.make_prompt_cache", return_value=[]):
        manager.get_or_create_cache(model_cache, prompt, session_key="sess-a")
        manager.get_or_create_cache(model_cache, prompt, session_key="sess-b")

    assert len(manager.caches) == 2
    assert ("sess-a", lm_tokens_key(prompt)) in manager.caches
    assert ("sess-b", lm_tokens_key(prompt)) in manager.caches


def test_vlm_prompt_cache_manager_filters_by_session_key() -> None:
    manager = VlmPromptCacheManager(max_position_embeddings=4096, max_caches=10)
    prompt = list(range(200))

    cache_a = VlmPromptCache(
        max_position_embeddings=4096,
        session_key="sess-a",
        tokens=prompt[:150],
        model_key="model",
    )
    cache_b = VlmPromptCache(
        max_position_embeddings=4096,
        session_key="sess-b",
        tokens=prompt[:180],
        model_key="model",
    )
    manager.caches[("sess-a", "a")] = cache_a
    manager.caches[("sess-b", "b")] = cache_b

    selected_a, _suffix_a, cached_a = manager.get_or_create_cache(
        object(),
        "model",
        prompt,
        session_key="sess-a",
    )
    assert selected_a is cache_a
    assert cached_a == 150

    selected_b, _suffix_b, cached_b = manager.get_or_create_cache(
        object(),
        "model",
        prompt,
        session_key="sess-b",
    )
    assert selected_b is cache_b
    assert cached_b == 180


def test_vlm_prompt_cache_manager_uses_session_key_in_cache_key() -> None:
    manager = VlmPromptCacheManager(max_position_embeddings=4096, max_caches=10)
    prompt = list(range(32))

    with patch("mlx_omni_server.chat.mlx_vlm.prompt_cache.make_prompt_cache", return_value=[]):
        manager.get_or_create_cache(object(), "model", prompt, session_key="sess-a")
        manager.get_or_create_cache(object(), "model", prompt, session_key="sess-b")

    assert len(manager.caches) == 2
    assert ("sess-a", vlm_tokens_key(prompt)) in manager.caches
    assert ("sess-b", vlm_tokens_key(prompt)) in manager.caches
