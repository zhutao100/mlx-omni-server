import gc
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache
from mlx_vlm.models.cache import make_prompt_cache
from mlx_vlm.prompt_cache import PromptCacheBundle

from ...utils.logger import logger
from ..prompt_cache_utils import common_prefix_len, hash_tokens_with_media

CacheKey = tuple[str, str]


def tokens_key(tokens: list[int], media_hashes: list[str] | None = None) -> str:
    """Stable cache key for a token sequence and optional media."""
    return hash_tokens_with_media(tokens, media_hashes)


@dataclass
class PromptCache:
    max_position_embeddings: int
    session_key: str = ""
    tokens: list[int] = field(default_factory=list)
    bundle: PromptCacheBundle | None = None
    model_key: str = ""
    media_hashes: list[str] = field(default_factory=list)  # New field for media files
    is_multimodal: bool = False  # Track if cache contains multimodal content

    def reset_prompt_cache(
        self,
        model,
        model_key: str,
        media_hashes: list[str] | None = None,
    ):
        """
        Build a fresh prompt cache structure using the model.
        """
        logger.debug("Resetting prompt cache from scratch.")
        # model_key to detect model swaps
        self.model_key = model_key
        # store media hashes for multimodal content
        self.media_hashes = media_hashes or []
        self.is_multimodal = bool(media_hashes)
        # build base cache(s) for the language model
        language_model = getattr(model, "language_model", model)
        kv_cache = make_prompt_cache(
            language_model,
            max_kv_size=self.max_position_embeddings,
        )
        self.bundle = PromptCacheBundle(kv_cache=kv_cache)
        # Tokens represent committed KV state; new cache starts empty and is
        # advanced as generation succeeds.
        self.tokens = []

    def get_prompt_cache(
        self,
        model,
        model_key: str,
        prompt: list[int],
        media_hashes: list[str] | None = None,
    ) -> tuple[list[int], int]:
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
        if self.model_key != model_key or prefix_len == 0:
            self.reset_prompt_cache(model, model_key, media_hashes)
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
        bundle = getattr(self, "bundle", None)
        kv_cache = getattr(bundle, "kv_cache", None) if bundle is not None else None
        if kv_cache and can_trim_prompt_cache(kv_cache):
            num_to_trim = cache_len - prefix_len
            logger.debug("Trimming %d tokens from VLM KV cache (in-place).", num_to_trim)
            trim_prompt_cache(kv_cache, num_to_trim)
            self.tokens = self.tokens[:prefix_len]
            if bundle is not None:
                bundle.tokens_processed = len(self.tokens)
            return prompt[prefix_len:], prefix_len

        logger.debug("VLM cache cannot be trimmed in-place. Resetting cache.")
        self.reset_prompt_cache(model, model_key, media_hashes)
        return prompt, 0


def get_vlm_cache_config(model_name: str) -> dict:
    """
    Get VLM-specific cache configuration based on model type.
    """
    vlm_models = {
        "llava": {"max_caches": 5, "multimodal_ratio": 0.4},
        "bakllava": {"max_caches": 4, "multimodal_ratio": 0.3},
        "cogvlm": {"max_caches": 3, "multimodal_ratio": 0.5},
        "qwen-vl": {"max_caches": 6, "multimodal_ratio": 0.4},
        "paligemma": {"max_caches": 4, "multimodal_ratio": 0.3},
    }

    # Check if this is a VLM model
    for vlm_key in vlm_models:
        if vlm_key in model_name.lower():
            return vlm_models[vlm_key]

    # Default configuration for non-VLM models
    return {"max_caches": 10, "multimodal_ratio": 0.1}


class PromptCacheManager:
    """
    Manager that keeps multiple PromptCache branches (LRU-evicted) and selects
    the best branch for incoming prompts. It forks on divergence (preserving
    original caches) and reuses caches for append/extend flows.
    """

    def __init__(self, max_position_embeddings: int, max_caches: int):
        self.max_position_embeddings = max_position_embeddings
        self.caches: "OrderedDict[CacheKey, PromptCache]" = OrderedDict()
        self.max_caches = max_caches
        self.multimodal_cache_ratio = 0.3  # Reserve 30% of cache slots for multimodal content

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
            bundle = getattr(evicted_cache, "bundle", None)
            if bundle is not None:
                kv_cache = getattr(bundle, "kv_cache", None)
                if kv_cache:
                    stack: list[Any] = list(kv_cache)
                    while stack:
                        c = stack.pop()
                        if c is None:
                            continue
                        if isinstance(c, (list, tuple)):
                            stack.extend(c)
                            continue
                        nested = getattr(c, "caches", None)
                        if nested:
                            stack.extend(list(nested))

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
                # Cached multimodal context tensors can be large too.
                try:
                    bundle.context = None
                except Exception:
                    pass

            # Drop the reference entirely
            del evicted_cache

        if did_evict:
            # Force Python to finalize objects & free memory back to MLX
            gc.collect()

    def get_or_create_cache(
        self,
        model,
        model_key: str,
        prompt: list[int],
        media_hashes: list[str] | None = None,
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
        if media_hashes:
            min_prefix_len = 10
        else:
            min_prefix_len = 100  # Standard threshold for text-only

        cache_namespace = session_key or "default"
        best_cache = None
        best_key = None
        best_prefix_len = 0

        logger.debug(f"Number of existing caches: {len(self.caches)}")
        # Find longest prefix match among existing caches
        for key, cache in self.caches.items():
            if key[0] != cache_namespace:
                continue
            # For multimodal content, require exact media hash match
            if media_hashes:
                if not cache.is_multimodal:
                    continue
                if cache.media_hashes != media_hashes:
                    continue  # Skip caches with different media content (order matters)
            elif cache.is_multimodal:
                continue

            prefix_len = common_prefix_len(cache.tokens, prompt)
            if prefix_len > best_prefix_len:
                best_cache = cache
                best_key = key
                best_prefix_len = prefix_len

        logger.debug(f"Best prefix length found: {best_prefix_len}")
        logger.debug(f"Best cache key: {best_key}")
        logger.debug(f"Best cache tokens length: {len(best_cache.tokens) if best_cache else 'N/A'}")
        if best_cache is not None and best_prefix_len >= min_prefix_len:
            logger.debug(
                "Re-using existing cache %s (common prefix=%d).",
                best_key,
                best_prefix_len,
            )
            suffix, cached_tokens = best_cache.get_prompt_cache(
                model,
                model_key,
                prompt,
                media_hashes,
            )
            # mark as recently used
            assert best_key is not None
            self.caches.move_to_end(best_key, last=True)
            return best_cache, suffix, cached_tokens

        # No cache to reuse -> create brand-new cache
        logger.debug("No matching cache found; creating new.")
        new_cache = PromptCache(max_position_embeddings=self.max_position_embeddings)
        new_cache.reset_prompt_cache(model, model_key, media_hashes)
        key = (cache_namespace, hash_tokens_with_media(prompt, media_hashes))
        new_cache.session_key = cache_namespace
        self.caches[key] = new_cache
        self._evict_if_needed()
        return new_cache, prompt, 0
