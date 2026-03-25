from types import SimpleNamespace
from unittest.mock import patch

from mlx_omni_server.chat.mlx_lm.prompt_cache import PromptCache, PromptCacheManager


def test_lm_prompt_cache_manager_fork_leaves_one_token_suffix() -> None:
    manager = PromptCacheManager(max_position_embeddings=4096, max_caches=10)

    prompt = list(range(200))
    cached_tokens = list(range(400))
    existing_cache = PromptCache(
        max_position_embeddings=4096,
        session_key="default",
        tokens=cached_tokens,
        cache=[object()],
        model_key="model",
    )
    manager.caches[("default", "existing")] = existing_cache

    model_cache = SimpleNamespace(
        model_id=SimpleNamespace(name="model"),
        model=object(),
        draft_model=None,
    )

    called: dict[str, int] = {}

    def clone_up_to(self: PromptCache, prefix_len: int, _model_cache) -> PromptCache:
        called["prefix_len"] = prefix_len
        return PromptCache(
            max_position_embeddings=self.max_position_embeddings,
            session_key=self.session_key,
            tokens=list(self.tokens[:prefix_len]),
            cache=[object()],
            model_key=self.model_key,
        )

    with (
        patch(
            "mlx_omni_server.chat.mlx_lm.prompt_cache.can_trim_prompt_cache",
            return_value=True,
        ),
        patch(
            "mlx_omni_server.chat.mlx_lm.prompt_cache.PromptCache.clone_up_to",
            new=clone_up_to,
        ),
    ):
        _selected, suffix_tokens, cached_count = manager.get_or_create_cache(
            model_cache,
            prompt,
            session_key="default",
        )

    assert called["prefix_len"] == len(prompt) - 1
    assert cached_count == len(prompt) - 1
    assert suffix_tokens == [prompt[-1]]
