import gc
import struct
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Tuple

from mlx_lm.models.cache import (
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)

from ...utils.logger import logger


logger = logging.getLogger(__name__)


def common_prefix_len(a: list[int], b: list[int]) -> int:
    min_len = min(len(a), len(b))
    for i in range(min_len):
        if a[i] != b[i]:
            return i
    return min_len


def tokens_key(tokens: list[int]) -> str:
    """
    Pack tokens as 4-byte little-endian ints then hash.
    Works regardless of token id range.
    """
    if not tokens:
        return "empty"
    b = b"".join(struct.pack("<I", int(t)) for t in tokens)
    return sha256(b).hexdigest()


@dataclass
class PromptCache:
    max_position_embeddings: int
    tokens: list[int] = field(default_factory=list)
    cache: list[Any] | None = field(default_factory=list)
    model_key: str = ""

    def extend_completion_cache(self, completion_tokens: list[int]):
        self.tokens.extend(completion_tokens)

    def reset_prompt_cache(self, model_cache, prompt_tokens: list[int]):
        """
        Build a fresh prompt cache for `prompt_tokens` using model_cache.
        """
        logger.debug("Resetting prompt cache from scratch.")
        # model_key to detect model swaps
        self.model_key = model_cache.model_id.name
        # build base cache(s)
        if getattr(model_cache, "model", None) is not None:
            self.cache = make_prompt_cache(model_cache.model, max_kv_size=self.max_position_embeddings)
        else:
            logger.error("Model cache has no model attribute; setting empty cache.")
            self.cache = []
        # include draft_model if present
        if getattr(model_cache, "draft_model", None) is not None:
            self.cache += make_prompt_cache(model_cache.draft_model, max_kv_size=self.max_position_embeddings)

        # store tokens
        self.tokens = list(prompt_tokens)

    def get_prompt_cache(self, model_cache, prompt: list[int]) -> Tuple[list[int], int]:
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
        if self.model_key != getattr(model_cache.model_id, "name", self.model_key) or com_prefix == 0:
            self.reset_prompt_cache(model_cache, prompt)
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
            logger.debug(f"Common prefix ({com_prefix}) shorter than cache ({cache_len}). Attempting trim.")
            if self.cache and can_trim_prompt_cache(self.cache):
                num_to_trim = cache_len - com_prefix
                logger.debug(f"Trimming {num_to_trim} tokens from cache (in-place).")
                trimmed = trim_prompt_cache(self.cache, num_to_trim)
                # trim_prompt_cache returns number trimmed (per layer), but state mutated in-place
                self.tokens = self.tokens[:com_prefix]
                suffix = prompt[com_prefix:]
                self.tokens.extend(suffix)
                prompt_cached_tokens = com_prefix
                return suffix, prompt_cached_tokens
            else:
                logger.debug("Cache cannot be trimmed in-place. Resetting cache to prompt.")
                self.reset_prompt_cache(model_cache, prompt)
                return prompt, 0

        # Fallback: return whole prompt
        logger.debug("No reuse path found; returning full prompt.")
        return prompt, 0

    def clone_up_to(self, prefix_len: int, model_cache) -> "PromptCache":
        """
        Create a forked PromptCache from this one up to prefix_len tokens.

        Strategy:
          - If self.cache exists and is trimmable: construct new instances of each
            per-layer cache type, copy their state/meta_state, then call trim_prompt_cache
            on the cloned cache (so we avoid mutating the original).
          - If cloning/trim unsupported -> fallback to recomputing via reset_prompt_cache(prefix_tokens).

        Returns a new PromptCache instance which is independent of self.
        """
        logger.debug(f"Cloning prompt cache up to {prefix_len} tokens.")
        new_cache = PromptCache(max_position_embeddings=self.max_position_embeddings)
        new_cache.model_key = self.model_key
        new_cache.tokens = list(self.tokens[:prefix_len])

        # If there is no underlying cache or trimming isn't supported, recompute
        if not self.cache:
            logger.debug("No underlying cache to clone; recomputing for prefix.")
            new_cache.reset_prompt_cache(model_cache, new_cache.tokens)
            return new_cache

        # If trimmable, try to clone per-layer objects carefully (avoid deepcopy of MX tensors)
        try:
            # create fresh instances of same types and copy states
            cloned_layers = []
            for layer_cache in self.cache:
                # instantiate a new object of the same class
                LayerType = type(layer_cache)
                try:
                    cloned_inst = LayerType()
                except Exception:
                    # If constructor requires args, fallback to deepcopy attempt
                    import copy as _copy
                    cloned_inst = _copy.deepcopy(layer_cache)

                # copy state (state setter should accept the same object shape)
                try:
                    cloned_inst.state = layer_cache.state
                except Exception as e:
                    logger.debug("Failed to copy state via setter; attempting deepcopy as fallback: %s", e)
                    import copy as _copy
                    cloned_inst = _copy.deepcopy(layer_cache)

                # copy meta_state if present
                if hasattr(layer_cache, "meta_state"):
                    try:
                        cloned_inst.meta_state = layer_cache.meta_state
                    except Exception:
                        # ignore meta state failures - not critical
                        pass

                cloned_layers.append(cloned_inst)

            # Now attempt to trim the cloned cache to prefix length
            num_to_trim = len(self.tokens) - prefix_len
            if num_to_trim > 0 and can_trim_prompt_cache(cloned_layers):
                logger.debug("Trimming cloned cache by %d tokens.", num_to_trim)
                trim_prompt_cache(cloned_layers, num_to_trim)
                new_cache.cache = cloned_layers
                return new_cache
            else:
                logger.debug("Cloned layers not trimmable or nothing to trim; recomputing prefix cache.")
                new_cache.reset_prompt_cache(model_cache, new_cache.tokens)
                return new_cache

        except Exception as e:
            logger.exception("Exception while cloning cache: %s. Falling back to recompute.", e)
            new_cache.reset_prompt_cache(model_cache, new_cache.tokens)
            return new_cache


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

        # Force Python to finalize objects & free memory back to MLX
        gc.collect()

    def get_or_create_cache(self, model_cache, prompt: list[int]) -> Tuple[PromptCache, list[int], int]:
        """
        Returns (active_cache, suffix_tokens_to_process, num_cached_tokens).
        Behavior:
          - If a cache fully prefixes `prompt` and is shorter than prompt: extend it (reuse).
          - If a cache shares a common prefix but is longer (divergence): fork via clone_up_to(prefix_len).
          - If no cache matches: create a new cache from scratch.
        """
        best_cache = None
        best_key = None
        best_prefix_len = 0

        # find longest prefix match among existing caches
        for key, cache in self.caches.items():
            prefix_len = common_prefix_len(cache.tokens, prompt)
            if prefix_len > best_prefix_len:
                best_cache = cache
                best_key = key
                best_prefix_len = prefix_len

        if best_cache is not None and best_prefix_len > 0:
            # Case A: prompt extends the cache fully (cache is prefix)
            if best_prefix_len >= 0.95 * len(best_cache.tokens) and best_prefix_len < len(prompt):
                logger.debug(f"Re-using existing cache {best_key} (prefix match >= 95%).")
                suffix, cached_tokens = best_cache.get_prompt_cache(model_cache, prompt)
                # mark as recently used
                assert best_key is not None
                self.caches.move_to_end(best_key, last=True)
                return best_cache, suffix, cached_tokens

            # Case B: divergence -> fork a new cache from the prefix (do NOT mutate original)
            logger.debug("Divergent prompt (common prefix=%d). Forking new branch.", best_prefix_len)
            forked = best_cache.clone_up_to(best_prefix_len, model_cache)
            suffix_tokens = prompt[best_prefix_len:]
            # append suffix tokens into forked token list so later extension uses them
            forked.tokens.extend(suffix_tokens)
            new_key = tokens_key(prompt)
            self.caches[new_key] = forked
            self._evict_if_needed()
            return forked, suffix_tokens, best_prefix_len

        # No cache to reuse -> create brand-new cache
        logger.debug("No matching cache found; creating new.")
        new_cache = PromptCache(max_position_embeddings=self.max_position_embeddings)
        new_cache.reset_prompt_cache(model_cache, prompt)
        key = tokens_key(prompt)
        self.caches[key] = new_cache
        self._evict_if_needed()
        return new_cache, prompt, 0
