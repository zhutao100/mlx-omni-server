import mlx.core as mx
import pytest

from mlx_omni_server.chat.logits_processors.penalties import (
    PresenceFrequencyPenaltyProcessor,
    build_logits_processors,
    normalize_logit_bias,
)
from mlx_omni_server.chat.schema import ChatCompletionRequest


class _DummyTokenizer:
    vocab_size = 10


def test_normalize_logit_bias_returns_none_for_empty() -> None:
    assert normalize_logit_bias(None) is None
    assert normalize_logit_bias({}) is None


def test_normalize_logit_bias_parses_filters_and_clamps(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING")
    bias = {
        "5": 500.0,  # clamp to 100
        "2": -200.0,  # clamp to -100
        "abc": 1.0,  # invalid key
        "-1": 1.0,  # out of range
        "999": 1.0,  # out of vocab range
        "0": 0.0,  # dropped as no-op
    }

    assert normalize_logit_bias(bias, vocab_size=10) == {2: -100.0, 5: 100.0}


def test_presence_frequency_processor_applies_prompt_counts_and_updates_incrementally() -> None:
    processor = PresenceFrequencyPenaltyProcessor(
        [1, 2, 2, 3],
        presence_penalty=0.5,
        frequency_penalty=0.2,
    )

    logits = mx.zeros((1, 5), dtype=mx.float32)

    # First call: tokens reflect backend state, but prompt counts are already seeded.
    out1 = processor(mx.array([3]), logits)
    assert out1.tolist()[0] == pytest.approx([0.0, -0.7, -0.9, -0.7, 0.0])

    # Second call: one new generated token (2) appended, frequency penalty increases for token 2.
    out2 = processor(mx.array([3, 2]), logits)
    assert out2.tolist()[0] == pytest.approx([0.0, -0.7, -1.1, -0.7, 0.0])


def test_build_logits_processors_orders_penalties_and_bias() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "repetition_penalty": 1.2,
            "repetition_context_size": 10,
            "presence_penalty": 0.1,
            "logit_bias": {"4": 2.0},
        }
    )

    processors = build_logits_processors(
        request,
        _DummyTokenizer(),
        prompt_tokens=[1, 2, 3],
    )

    assert processors[0].__name__ == "repetition_penalty_processor"
    assert isinstance(processors[1], PresenceFrequencyPenaltyProcessor)
    assert processors[2].__name__ == "logit_bias_processor"


def test_build_logits_processors_skips_repetition_when_disabled() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "repetition_penalty": 1.0,
            "logit_bias": {"4": 2.0},
        }
    )

    processors = build_logits_processors(
        request,
        _DummyTokenizer(),
        prompt_tokens=[1, 2, 3],
    )

    assert len(processors) == 1
    assert processors[0].__name__ == "logit_bias_processor"
