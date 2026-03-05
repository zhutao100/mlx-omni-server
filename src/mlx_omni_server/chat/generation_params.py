from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from ..utils.logger import logger
from .text_models import GenerationParams

SAMPLER_PARAM_KEYS: frozenset[str] = frozenset(
    {
        "top_k",
        "min_tokens_to_keep",
        "min_p",
        "xtc_probability",
        "xtc_threshold",
        "xtc_special_tokens",
    }
)

MODEL_PARAM_KEYS: frozenset[str] = frozenset(
    {
        "adapter_path",
        "draft_model",
        # Additional config for `apply_chat_template`
        "chat_template_config",
    }
)

TEMPLATE_PARAM_KEYS: frozenset[str] = frozenset(
    {
        # Qwen3
        "enable_thinking",
        "thinking_budget",
        # Claude
        "thinking",
        # Gemini
        "thinkingConfig",
        # Grok
        "reasoning_effort",
        # Others
        "reasoning",
    }
)

INCOMPATIBLE_PARAM_KEYS: frozenset[str] = frozenset({"include"})

LM_GENERATE_STEP_PARAM_KEYS: frozenset[str] = frozenset(
    {
        "max_kv_size",
        "prefill_step_size",
        "kv_bits",
        "kv_group_size",
        "quantized_kv_start",
        "num_draft_tokens",
    }
)

VLM_GENERATE_STEP_PARAM_KEYS: frozenset[str] = frozenset(
    {
        "max_kv_size",
        "prefill_step_size",
        "kv_bits",
        "kv_group_size",
        "quantized_kv_start",
    }
)


def split_generation_params(
    params: Mapping[str, Any],
    *,
    supported_generate_params: Collection[str],
) -> GenerationParams:
    sampler_kwargs: dict[str, Any] = {}
    model_kwargs: dict[str, Any] = {}
    generate_kwargs: dict[str, Any] = {}
    template_kwargs: dict[str, Any] = {}
    dropped_generate_kwargs: list[str] = []

    supported_generate_params_set = frozenset(supported_generate_params)

    for key, value in params.items():
        if key in SAMPLER_PARAM_KEYS:
            sampler_kwargs[key] = value
        elif key in MODEL_PARAM_KEYS:
            model_kwargs[key] = value
        elif key in TEMPLATE_PARAM_KEYS:
            template_kwargs[key] = value
        elif key in INCOMPATIBLE_PARAM_KEYS:
            logger.warning("Generation parameter '%s: %s' is not supported; dropping.", key, value)
        else:
            if key in supported_generate_params_set:
                generate_kwargs[key] = value
            else:
                dropped_generate_kwargs.append(key)

    if dropped_generate_kwargs:
        dropped_generate_kwargs.sort()
        logger.warning(
            "Dropping unsupported generation parameter(s): %s",
            ", ".join(dropped_generate_kwargs),
        )

    return {
        "sampler_kwargs": sampler_kwargs,
        "model_kwargs": model_kwargs,
        "generate_kwargs": generate_kwargs,
        "template_kwargs": template_kwargs,
    }
