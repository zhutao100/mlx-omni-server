import copy
import gc
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from mlx_lm.models.cache import (
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)

from ...utils.logger import logger
from ..prompt_cache_utils import common_prefix_len, hash_tokens

CacheKey = tuple[str, str]


def tokens_key(tokens: list[int]) -> str:
    """Stable cache key for a token sequence."""
    return hash_tokens(tokens)


@dataclass
class PromptCache:
    max_position_embeddings: int
    session_key: str = ""
    tokens: list[int] = field(default_factory=list)
    cache: list[Any] | None = field(default_factory=list)
    model_key: str = ""

    def reset_prompt_cache(self, model_cache) -> None:
        """
        Build a fresh prompt cache structure using model_cache.
        """
        logger.debug("Resetting prompt cache from scratch.")
        # model_key to detect model swaps
        self.model_key = model_cache.model_id.name
        # build base cache(s)
        if getattr(model_cache, "model", None) is not None:
            self.cache = make_prompt_cache(
                model_cache.model,
                max_kv_size=self.max_position_embeddings,
            )
        else:
            logger.error("Model cache has no model attribute; setting empty cache.")
            self.cache = []
        # include draft_model if present
        if getattr(model_cache, "draft_model", None) is not None:
            self.cache += make_prompt_cache(
                model_cache.draft_model, max_kv_size=self.max_position_embeddings
            )

        # Tokens represent committed KV state; new cache starts empty and is
        # advanced transactionally as generation progresses.
        self.tokens = []

    def get_prompt_cache(self, model_cache, prompt: list[int]) -> tuple[list[int], int]:
        """
        Determine suffix of prompt that needs processing, attempting to reuse/trim
        this cache in-place if it is safe (this is used in 'extend' flows).
        Returns (prompt_suffix_to_process, prompt_cached_tokens_count)
        """
        cache_len = len(self.tokens)
        prompt_len = len(prompt)
        prefix_len = common_prefix_len(self.tokens, prompt)

        # leave at least one token to process (so model gets some new input)
        prefix_len = min(prefix_len, max(0, prompt_len - 1))

        # Reset if model changed or no common prefix
        if (
            self.model_key != getattr(model_cache.model_id, "name", self.model_key)
            or prefix_len == 0
        ):
            self.reset_prompt_cache(model_cache)
            return prompt, 0

        # Case: cache is prefix of prompt -> process suffix
        if prefix_len == cache_len:
            logger.debug(f"Cache is prefix (cache_len={cache_len}); processing suffix.")
            return prompt[prefix_len:], prefix_len

        # Case: prompt shorter than cached tokens (should be handled by manager for branching),
        # or attempt to trim (here we support in-place trim)
        logger.debug(
            "Common prefix (%d) shorter than cache (%d). Attempting trim.",
            prefix_len,
            cache_len,
        )
        if self.cache and can_trim_prompt_cache(self.cache):
            num_to_trim = cache_len - prefix_len
            logger.debug("Trimming %d tokens from cache (in-place).", num_to_trim)
            trim_prompt_cache(self.cache, num_to_trim)
            # trim_prompt_cache mutates cache state in-place
            self.tokens = self.tokens[:prefix_len]
            return prompt[prefix_len:], prefix_len

        logger.debug("Cache cannot be trimmed in-place. Resetting cache.")
        self.reset_prompt_cache(model_cache)
        return prompt, 0

    def clone_up_to(self, prefix_len: int, model_cache) -> "PromptCache":
        """
        Create a forked PromptCache from this one up to prefix_len tokens.

        Strategy:
          - If self.cache exists and is trimmable: construct new instances of each
            per-layer cache type, copy their state/meta_state, then call trim_prompt_cache
            on the cloned cache (so we avoid mutating the original).
          - If cloning/trim unsupported -> fallback to recomputing via reset_prompt_cache().

        Returns a new PromptCache instance which is independent of self.
        """
        logger.debug(f"Cloning prompt cache up to {prefix_len} tokens.")
        new_cache = PromptCache(max_position_embeddings=self.max_position_embeddings)
        new_cache.session_key = self.session_key
        new_cache.model_key = self.model_key
        new_cache.tokens = list(self.tokens[:prefix_len])

        # If there is no underlying cache or trimming isn't supported, recompute
        if not self.cache:
            logger.debug("No underlying cache to clone; recomputing for prefix.")
            new_cache.reset_prompt_cache(model_cache)
            return new_cache

        # If trimmable, try to clone per-layer objects carefully (avoid deepcopy of MX tensors)
        try:
            # create fresh instances of same types and copy states
            cloned_layers = []
            for layer_cache in self.cache:
                try:
                    LayerType = type(layer_cache)
                    from_state = getattr(LayerType, "from_state", None)
                    if callable(from_state):
                        cloned_inst = from_state(
                            layer_cache.state,
                            getattr(layer_cache, "meta_state", ""),
                        )
                    else:
                        cloned_inst = copy.deepcopy(layer_cache)
                except Exception as e:
                    logger.debug(
                        "Failed to clone cache layer via from_state; falling back to deepcopy: %s",
                        e,
                    )
                    cloned_inst = copy.deepcopy(layer_cache)

                cloned_layers.append(cloned_inst)

            # Now attempt to trim the cloned cache to prefix length
            num_to_trim = len(self.tokens) - prefix_len
            if num_to_trim > 0 and can_trim_prompt_cache(cloned_layers):
                logger.debug("Trimming cloned cache by %d tokens.", num_to_trim)
                trim_prompt_cache(cloned_layers, num_to_trim)
                new_cache.cache = cloned_layers
                return new_cache
            else:
                logger.debug(
                    "Cloned layers not trimmable or nothing to trim; recomputing prefix cache."
                )
                new_cache.reset_prompt_cache(model_cache)
                return new_cache

        except Exception as e:
            logger.exception("Exception while cloning cache: %s. Falling back to recompute.", e)
            new_cache.reset_prompt_cache(model_cache)
            return new_cache


class PromptCacheManager:
    """
    Manager that keeps multiple PromptCache branches (LRU-evicted) and selects
    the best branch for incoming prompts. It forks on divergence (preserving
    original caches) and reuses caches for append/extend flows.
    """

    def __init__(self, max_position_embeddings: int, max_caches: int = 2):
        self.max_position_embeddings = max_position_embeddings
        self.caches: "OrderedDict[CacheKey, PromptCache]" = OrderedDict()
        self.max_caches = max_caches

    def _evict_if_needed(self):
        """
        Evict old cache entries if we exceed max_entries.
        Explicitly clears MLX cache tensors so memory is released quickly.
        """
        did_evict = False
        while len(self.caches) > self.max_caches:
            did_evict = True
            # Pop oldest (FIFO)
            evicted_key, evicted_cache = self.caches.popitem(last=False)
            logger.debug("Evicting prompt cache: %s", evicted_key)

            # Explicitly clear MLX tensors inside the evicted cache
            if hasattr(evicted_cache, "cache") and evicted_cache.cache:
                for c in evicted_cache.cache:
                    # KVCache, RotatingKVCache, QuantizedKVCache, etc.
                    if hasattr(c, "keys"):
                        c.keys = None
                    if hasattr(c, "values"):
                        c.values = None
                    if hasattr(c, "offset"):
                        c.offset = 0
                    if hasattr(c, "_idx"):
                        c._idx = 0
                    if hasattr(c, "cache"):
                        c.cache = None

            # Drop the reference entirely
            del evicted_cache

        if did_evict:
            # Force Python to finalize objects & free memory back to MLX
            gc.collect()

    def _register_cache(
        self,
        cache_namespace: str,
        prompt: list[int],
        cache: PromptCache,
    ) -> None:
        key = (cache_namespace, hash_tokens(prompt))
        cache.session_key = cache_namespace
        self.caches[key] = cache
        self._evict_if_needed()

    def get_or_create_cache(
        self,
        model_cache,
        prompt: list[int],
        *,
        session_key: str | None = None,
    ) -> tuple[PromptCache, list[int], int]:
        """
        Returns (active_cache, suffix_tokens_to_process, num_cached_tokens).
        Behavior:
          - If a cache fully prefixes `prompt` and is shorter than prompt: extend it (reuse).
          - If a cache shares a common prefix but is longer (divergence): fork via clone_up_to(prefix_len).
          - If no cache matches: create a new cache from scratch.
        """
        cache_namespace = session_key or "default"
        best_cache = None
        best_key = None
        best_prefix_len = 0

        # find longest prefix match among existing caches
        for key, cache in self.caches.items():
            if key[0] != cache_namespace:
                continue
            prefix_len = common_prefix_len(cache.tokens, prompt)
            if prefix_len > best_prefix_len:
                best_cache = cache
                best_key = key
                best_prefix_len = prefix_len

        if (
            best_cache is not None and best_prefix_len > 100
        ):  # set min length 100 for a worthy cache reuse.
            # Case A: common prefix is at least 80% of the cache.
            if best_prefix_len >= 0.8 * len(best_cache.tokens):
                logger.debug(f"Re-using existing cache {best_key} (prefix match >= 80%).")
                suffix, cached_tokens = best_cache.get_prompt_cache(model_cache, prompt)
                # mark as recently used
                assert best_key is not None
                self.caches.move_to_end(best_key, last=True)
                return best_cache, suffix, cached_tokens

            # Case B: divergence -> fork a new cache from the prefix (do NOT mutate original)
            logger.debug(
                "Divergent prompt (common prefix=%d). Forking new branch.",
                best_prefix_len,
            )
            if best_cache.cache and can_trim_prompt_cache(best_cache.cache):
                try:
                    forked = best_cache.clone_up_to(best_prefix_len, model_cache)
                except Exception:
                    logger.exception(
                        "Failed to clone prompt cache at prefix_len=%d; falling back to cold cache.",
                        best_prefix_len,
                    )
                    forked = None
            else:
                forked = None

            # If we cannot safely fork a cache for the shared prefix (e.g. non-trimmable caches),
            # fall back to a cold cache but keep the original branch intact.
            if forked is None:
                logger.debug("Cache fork not supported; creating new cache from scratch.")
                new_cache = PromptCache(max_position_embeddings=self.max_position_embeddings)
                new_cache.reset_prompt_cache(model_cache)
                self._register_cache(cache_namespace, prompt, new_cache)
                return new_cache, prompt, 0

            suffix_tokens = prompt[best_prefix_len:]
            forked.session_key = cache_namespace
            self._register_cache(cache_namespace, prompt, forked)
            return forked, suffix_tokens, best_prefix_len

        # No cache to reuse -> create brand-new cache
        logger.debug("No matching cache found; creating new.")
        new_cache = PromptCache(max_position_embeddings=self.max_position_embeddings)
        new_cache.reset_prompt_cache(model_cache)
        self._register_cache(cache_namespace, prompt, new_cache)
        return new_cache, prompt, 0
