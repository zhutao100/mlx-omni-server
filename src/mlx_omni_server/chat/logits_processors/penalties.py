from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable, Mapping, Sequence

import mlx.core as mx
from mlx_lm.sample_utils import make_logits_processors

from ...utils.logger import logger
from ..schema import ChatCompletionRequest


def _unwrap_tokenizer(tokenizer: Any) -> Any:
    return tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer


def _maybe_vocab_size(tokenizer: Any) -> int | None:
    tokenizer = _unwrap_tokenizer(tokenizer)

    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int) and vocab_size > 0:
        return vocab_size

    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocab = get_vocab()
        if isinstance(vocab, dict) and vocab:
            return len(vocab)

    return None


def normalize_logit_bias(
    logit_bias: Mapping[str, float] | None,
    *,
    vocab_size: int | None = None,
    clamp_range: tuple[float, float] = (-100.0, 100.0),
) -> dict[int, float] | None:
    if not logit_bias:
        return None

    lower, upper = clamp_range
    normalized: dict[int, float] = {}
    invalid_keys: list[str] = []
    out_of_range: list[int] = []
    dropped_values = 0
    clamped_values = 0

    for raw_key, raw_value in logit_bias.items():
        try:
            token_id = int(raw_key)
        except (TypeError, ValueError):
            invalid_keys.append(str(raw_key))
            continue

        if token_id < 0:
            out_of_range.append(token_id)
            continue

        if vocab_size is not None and token_id >= vocab_size:
            out_of_range.append(token_id)
            continue

        try:
            bias_value = float(raw_value)
        except (TypeError, ValueError):
            dropped_values += 1
            continue

        if not math.isfinite(bias_value):
            dropped_values += 1
            continue

        if bias_value < lower:
            bias_value = lower
            clamped_values += 1
        elif bias_value > upper:
            bias_value = upper
            clamped_values += 1

        if bias_value != 0.0:
            normalized[token_id] = bias_value

    if invalid_keys:
        sample = ", ".join(invalid_keys[:5])
        suffix = "…" if len(invalid_keys) > 5 else ""
        logger.warning("Dropping invalid logit_bias token ids: %s%s", sample, suffix)

    if out_of_range:
        sample = ", ".join(str(v) for v in out_of_range[:5])
        suffix = "…" if len(out_of_range) > 5 else ""
        logger.warning("Dropping out-of-range logit_bias token ids: %s%s", sample, suffix)

    if dropped_values:
        logger.warning("Dropping %d non-numeric/non-finite logit_bias values.", dropped_values)

    if clamped_values:
        logger.warning(
            "Clamped %d logit_bias values to [%s, %s].",
            clamped_values,
            lower,
            upper,
        )

    return normalized or None


class PresenceFrequencyPenaltyProcessor:
    def __init__(
        self,
        prompt_tokens: Sequence[int],
        *,
        presence_penalty: float,
        frequency_penalty: float,
    ) -> None:
        self._presence_penalty = float(presence_penalty)
        self._frequency_penalty = float(frequency_penalty)

        self._base_counts: Counter[int] = Counter(int(t) for t in prompt_tokens)
        self._counts: Counter[int] = Counter(self._base_counts)

        self._processed_token_count: int | None = None
        self._penalty_vector: mx.array | None = None
        self._vocab_size: int | None = None

    def _ensure_penalty_vector(self, *, vocab_size: int, dtype: mx.Dtype) -> None:
        if (
            self._penalty_vector is not None
            and self._vocab_size == vocab_size
            and self._penalty_vector.dtype == dtype
        ):
            return

        self._vocab_size = vocab_size
        self._penalty_vector = mx.zeros((vocab_size,), dtype=dtype)

        if not self._counts:
            return

        presence = self._presence_penalty
        frequency = self._frequency_penalty

        indices: list[int] = []
        values: list[float] = []
        for token_id, count in self._counts.items():
            if token_id < 0 or token_id >= vocab_size:
                continue
            penalty = 0.0
            if presence != 0.0:
                penalty += presence
            if frequency != 0.0:
                penalty += frequency * count
            if penalty == 0.0:
                continue
            indices.append(token_id)
            values.append(penalty)

        if not indices:
            return

        idx_arr = mx.array(indices)
        val_arr = mx.array(values, dtype=dtype)
        self._penalty_vector[idx_arr] = val_arr

    @staticmethod
    def _flatten_tokens(tokens: mx.array) -> mx.array:
        if tokens.ndim == 1:
            return tokens
        if tokens.ndim == 2 and tokens.shape[0] == 1:
            return tokens[0]
        raise ValueError(f"Unsupported tokens shape: {tokens.shape}")

    def _reset_tracking(self) -> None:
        self._counts = Counter(self._base_counts)
        self._processed_token_count = None
        self._penalty_vector = None
        self._vocab_size = None

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        seq = self._flatten_tokens(tokens)
        seq_len = int(seq.shape[0])

        vocab_size = int(logits.shape[-1])
        self._ensure_penalty_vector(vocab_size=vocab_size, dtype=logits.dtype)

        # Track only *new* tokens appended since the last call. We seed counts
        # with the full prompt tokens (including cached prompt) at construction
        # time, so we intentionally do not count tokens from the first call.
        if self._processed_token_count is None:
            self._processed_token_count = seq_len
        else:
            if seq_len < self._processed_token_count:
                self._reset_tracking()
                self._processed_token_count = seq_len
            elif seq_len > self._processed_token_count:
                new_tokens = seq[self._processed_token_count :].tolist()
                presence = self._presence_penalty
                frequency = self._frequency_penalty
                penalty_vector = self._penalty_vector
                assert penalty_vector is not None
                for token in new_tokens:
                    token_id = int(token)
                    self._counts[token_id] += 1
                    if token_id < 0 or token_id >= vocab_size:
                        continue
                    penalty = 0.0
                    if presence != 0.0:
                        penalty += presence
                    if frequency != 0.0:
                        penalty += frequency * self._counts[token_id]
                    penalty_vector[token_id] = penalty
                self._processed_token_count = seq_len

        penalty_vector = self._penalty_vector
        if penalty_vector is None:
            return logits

        if logits.ndim == 1:
            return logits - penalty_vector
        if logits.ndim == 2:
            return logits - penalty_vector[None, :]
        raise ValueError(f"Unsupported logits shape: {logits.shape}")


def build_logits_processors(
    request: ChatCompletionRequest,
    tokenizer: Any,
    *,
    prompt_tokens: Sequence[int] | None = None,
) -> list[Callable[[mx.array, mx.array], mx.array]]:
    processors: list[Callable[[mx.array, mx.array], mx.array]] = []

    repetition_penalty = getattr(request, "repetition_penalty", None)
    repetition_context_size = getattr(request, "repetition_context_size", None)
    if repetition_context_size is None:
        repetition_context_size = 20

    rep: float | None
    if repetition_penalty is None or float(repetition_penalty) == 1.0:
        rep = None
    else:
        rep = float(repetition_penalty)

    if rep is not None and 0.0 < rep < 1.0:
        logger.warning(
            "repetition_penalty=%s encourages repetition (values >= 1.0 reduce repetition).",
            rep,
        )

    if rep is not None:
        processors.extend(
            make_logits_processors(
                repetition_penalty=rep,
                repetition_context_size=int(repetition_context_size),
            )
        )

    presence_penalty = float(request.presence_penalty or 0.0)
    frequency_penalty = float(request.frequency_penalty or 0.0)
    if presence_penalty != 0.0 or frequency_penalty != 0.0:
        processors.append(
            PresenceFrequencyPenaltyProcessor(
                prompt_tokens or [],
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
        )

    vocab_size = _maybe_vocab_size(tokenizer)
    normalized_bias = normalize_logit_bias(
        request.logit_bias,
        vocab_size=vocab_size,
    )
    if normalized_bias:
        processors.extend(make_logits_processors(logit_bias=normalized_bias))

    if request.response_format and request.response_format.json_schema:
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        from ..mlx_lm.json_logits_processor import JsonLogitsProcessor

        tokenizer = _unwrap_tokenizer(tokenizer)
        wrapper = (
            tokenizer if isinstance(tokenizer, TokenizerWrapper) else TokenizerWrapper(tokenizer)
        )
        processors.append(JsonLogitsProcessor(wrapper, request.response_format))

    return processors
