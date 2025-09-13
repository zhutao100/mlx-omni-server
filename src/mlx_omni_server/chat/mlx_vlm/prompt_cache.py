import gc
import logging
import struct
from collections import OrderedDict
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, List, Optional, Tuple

from cv2 import log
from mlx_vlm.models.cache import make_prompt_cache

from ...utils.logger import logger

logger = logging.getLogger(__name__)


def common_prefix_len(a: list[int], b: list[int]) -> int:
    min_len = min(len(a), len(b))
    for i in range(min_len):
        if a[i] != b[i]:
            return i
    return min_len


def tokens_key(tokens: list[int], media_hashes: list[str] | None = None) -> str:
    """
    Pack tokens as 4-byte little-endian ints then hash.
    Optionally include media file hashes for multimodal content.
    """
    if not tokens:
        return "empty"

    # Create hash from tokens
    b = b"".join(struct.pack("<I", int(t)) for t in tokens)
    base_hash = sha256(b).hexdigest()

    # For multimodal content, include media file hashes
    if media_hashes:
        media_hash_str = "|".join(sorted(media_hashes))
        combined = f"{base_hash}:{media_hash_str}"
        return sha256(combined.encode()).hexdigest()

    return base_hash


@dataclass
class PromptCache:
    max_position_embeddings: int
    tokens: list[int] = field(default_factory=list)
    cache: list[Any] | None = field(default_factory=list)
    model_key: str = ""
    media_hashes: list[str] = field(default_factory=list)  # New field for media files
    is_multimodal: bool = False  # Track if cache contains multimodal content

    def extend_completion_cache(self, completion_tokens: list[int]):
        self.tokens.extend(completion_tokens)

    def reset_prompt_cache(self, model, model_key: str, prompt_tokens: list[int], media_hashes: list[str] | None = None):
        """
        Build a fresh prompt cache for `prompt_tokens` using the model.
        """
        logger.debug("Resetting prompt cache from scratch.")
        # model_key to detect model swaps
        self.model_key = model_key
        # store media hashes for multimodal content
        self.media_hashes = media_hashes or []
        self.is_multimodal = bool(media_hashes)
        # build base cache(s)
        self.cache = make_prompt_cache(model, max_kv_size=self.max_position_embeddings)

        # store tokens
        self.tokens = list(prompt_tokens)

    def get_prompt_cache(self, model, model_key: str, prompt: list[int]) -> Tuple[list[int], int]:
        """
        Determine suffix of prompt that needs processing, attempting to reuse/trim
        this cache in-place if it is safe (this is used in 'extend' flows).
        Returns (prompt_suffix_to_process, prompt_cached_tokens_count)
        """
        cache_len = len(self.tokens)
        prompt_len = len(prompt)
        com_prefix = common_prefix_len(self.tokens, prompt)
        prompt_cached_tokens = 0

        # leave at least one token to process (so model gets some new input)
        com_prefix = min(com_prefix, max(0, prompt_len - 1))

        # Reset if model changed or no common prefix
        if self.model_key != model_key or com_prefix == 0:
            self.reset_prompt_cache(model, model_key, prompt)
            return prompt, 0

        # Case: cache is prefix of prompt -> process suffix
        if com_prefix == cache_len:
            logger.debug(f"Cache is prefix (cache_len={cache_len}); processing suffix.")
            suffix = prompt[com_prefix:]
            # update tokens to include appended suffix
            self.tokens.extend(suffix)
            prompt_cached_tokens = com_prefix
            return suffix, prompt_cached_tokens

        # Case: prompt shorter than cached tokens (should be handled by manager for branching),
        # or attempt to trim (here we support in-place trim)
        if com_prefix < cache_len:
            logger.debug(f"Common prefix ({com_prefix}) shorter than cache ({cache_len}).")
            # For VLM models, we don't attempt to trim as it's complex with multimodal content
            logger.debug("Resetting cache for VLM model due to divergence.")
            self.reset_prompt_cache(model, model_key, prompt)
            return prompt, 0

        # Fallback: return whole prompt
        logger.debug("No reuse path found; returning full prompt.")
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
        self.caches: "OrderedDict[str, PromptCache]" = OrderedDict()
        self.max_caches = max_caches
        self.multimodal_cache_ratio = 0.3  # Reserve 30% of cache slots for multimodal content

    def _evict_if_needed(self):
        """
        Evict old cache entries if we exceed max_entries.
        Explicitly clears MLX cache tensors so memory is released quickly.
        """
        while len(self.caches) > self.max_caches:
            # Pop oldest (FIFO)
            evicted_key, evicted_cache = self.caches.popitem(last=False)
            logger.debug("Evicting prompt cache: %s", evicted_key)

            # Explicitly clear MLX tensors inside the evicted cache
            if hasattr(evicted_cache, "cache") and evicted_cache.cache:
                for c in evicted_cache.cache:
                    # KVCache, RotatingKVCache, etc.
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

        # Force Python to finalize objects & free memory back to MLX
        gc.collect()

    def get_or_create_cache(self, model, model_key: str, prompt: list[int], media_hashes: list[str] | None = None) -> Tuple[PromptCache, list[int], int]:
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

        best_cache = None
        best_key = None
        best_prefix_len = 0

        logger.debug(f"Number of existing caches: {len(self.caches)}")
        # Find longest prefix match among existing caches
        for key, cache in self.caches.items():
            # For multimodal content, require exact media hash match
            if media_hashes and cache.is_multimodal:
                if set(cache.media_hashes) != set(media_hashes):
                    continue  # Skip caches with different media content

            prefix_len = common_prefix_len(cache.tokens, prompt)
            if prefix_len > best_prefix_len:
                best_cache = cache
                best_key = key
                best_prefix_len = prefix_len

        logger.debug(f"Best prefix length found: {best_prefix_len}")
        logger.debug(f"Best cache key: {best_key}")
        logger.debug(f"Best cache tokens length: {len(best_cache.tokens) if best_cache else 'N/A'}")
        if best_cache is not None and best_prefix_len >= min_prefix_len:
            # Case A: common prefix is at least 95% of the cache.
            if best_prefix_len == len(best_cache.tokens):
                logger.debug(f"Re-using existing cache {best_key}.")
                suffix, cached_tokens = best_cache.get_prompt_cache(model, model_key, prompt)
                # mark as recently used
                assert best_key is not None
                self.caches.move_to_end(best_key, last=True)
                return best_cache, suffix, cached_tokens

        # No cache to reuse -> create brand-new cache
        logger.debug("No matching cache found; creating new.")
        new_cache = PromptCache(max_position_embeddings=self.max_position_embeddings)
        new_cache.reset_prompt_cache(model, model_key, prompt, media_hashes)
        key = tokens_key(prompt, media_hashes)
        self.caches[key] = new_cache
        self._evict_if_needed()
        return new_cache, prompt, 0
