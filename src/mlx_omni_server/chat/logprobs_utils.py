from __future__ import annotations

from typing import Any

import mlx.core as mx

from .utils import normalize_to_list, normalize_token, safe_decode_token


def process_logprobs_for_token(
    tokenizer: Any,
    *,
    token_id: int,
    token_logprobs: mx.array,
    top_k: int | None,
) -> dict[str, Any] | None:
    """Convert backend token logprobs into the OpenAI-style per-token dict.

    This matches the structure produced by the LM path so callers can aggregate into
    `{"content": [...]}` for non-streaming responses and pass directly for streamed chunks.
    """
    if top_k is not None and top_k <= 0:
        top_k = None

    try:
        token_str = normalize_token(safe_decode_token(tokenizer, token_id))
        token_value = mx.clip(token_logprobs[token_id], a_min=-100, a_max=None).item()
    except Exception:
        return None

    token_bytes = token_str.encode("utf-8")
    token_info: dict[str, Any] = {
        "token": token_str,
        "logprob": token_value,
        "bytes": list(token_bytes),
    }

    top_logprobs: list[dict[str, Any]] = []
    if top_k is not None:
        try:
            top_indices = mx.argpartition(-token_logprobs, kth=top_k - 1)[:top_k]
            top_probs = mx.clip(token_logprobs[top_indices], a_min=-100, a_max=None)

            top_indices_list = normalize_to_list(top_indices, int)
            top_probs_list = normalize_to_list(top_probs, float)

            for idx, logprob in zip(top_indices_list, top_probs_list):
                decoded = normalize_token(safe_decode_token(tokenizer, int(idx)))
                top_logprobs.append(
                    {
                        "token": decoded,
                        "logprob": float(logprob),
                        "bytes": list(decoded.encode("utf-8")),
                    }
                )
        except Exception:
            top_logprobs = []

    return {**token_info, "top_logprobs": top_logprobs}
