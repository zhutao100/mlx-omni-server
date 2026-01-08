from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from mlx_lm.tokenizer_utils import TokenizerWrapper

from mlx_omni_server.chat.mlx_lm.mlx_lm_model import MlxLmModel
from mlx_omni_server.chat.mlx_vlm.mlx_vlm_model import MlxVlmModel
from mlx_omni_server.chat.schema import (
    ChatCompletionRequest,
    ChatMessage,
    FunctionCall,
    Role,
    ToolCall,
    ToolType,
)
from mlx_omni_server.chat.text_models import GenerateResult


def _make_tool_call(call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        type=ToolType.FUNCTION,
        function=FunctionCall(name="f", arguments="{}"),
    )


def test_mlx_lm_stream_tool_calls_have_indexes(monkeypatch) -> None:
    model_cache = SimpleNamespace(
        model_type="glm4",
        tokenizer=Mock(spec=TokenizerWrapper),
        model_config={"max_position_embeddings": 128},
        draft_model=None,
    )
    model = MlxLmModel(model_cache)

    monkeypatch.setattr(
        model._reasoning_decoder,
        "stream_decode",
        lambda text: {"delta_content": text, "delta_reasoning": None},
    )

    decode_calls = {"count": 0}

    def fake_decode_stream(_text, _tools=None):  # noqa: ANN001
        decode_calls["count"] += 1
        call_id = "call_a" if decode_calls["count"] == 1 else "call_b"
        return ChatMessage(role=Role.ASSISTANT, content=None, tool_calls=[_make_tool_call(call_id)])

    monkeypatch.setattr(model._chat_tokenizer, "decode_stream", fake_decode_stream)
    monkeypatch.setattr(
        model._chat_tokenizer,
        "parse_buffer",
        lambda _tools=None: ChatMessage(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[_make_tool_call("call_a"), _make_tool_call("call_b")],
        ),
    )

    def fake_stream_generate(*_args, **_kwargs):  # noqa: ANN001
        yield GenerateResult(
            text="chunk1",
            token=0,
            finish_reason=None,
            prompt_tokens=1,
            generation_tokens=1,
        )
        yield GenerateResult(
            text="chunk2",
            token=1,
            finish_reason="stop",
            prompt_tokens=1,
            generation_tokens=2,
        )

    monkeypatch.setattr(model, "_stream_generate", fake_stream_generate)

    request = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role=Role.USER, content="hi")],
        tools=[],
        stream=True,
    )

    chunks = list(model.stream_generate(request))
    assert chunks

    deltas = [
        chunk.choices[0].delta for chunk in chunks if chunk.choices and chunk.choices[0].delta
    ]
    tool_call_indexes = {
        tool_call.id: tool_call.index for delta in deltas for tool_call in (delta.tool_calls or [])
    }

    assert tool_call_indexes["call_a"] == 0
    assert tool_call_indexes["call_b"] == 1


def test_mlx_vlm_stream_tool_calls_have_indexes(monkeypatch) -> None:
    model_cache = SimpleNamespace(
        model_type="glm4v_moe",
        tokenizer=Mock(spec=TokenizerWrapper),
        model_config={},
        draft_model=None,
        model_id=SimpleNamespace(name="test-model"),
    )
    model = MlxVlmModel(model_cache)

    decode_calls = {"count": 0}

    def fake_decode_stream(_text, _tools=None):  # noqa: ANN001
        decode_calls["count"] += 1
        call_id = "call_a" if decode_calls["count"] == 1 else "call_b"
        return ChatMessage(role=Role.ASSISTANT, content=None, tool_calls=[_make_tool_call(call_id)])

    monkeypatch.setattr(model._chat_tokenizer, "decode_stream", fake_decode_stream)
    monkeypatch.setattr(
        model._chat_tokenizer,
        "parse_buffer",
        lambda _tools=None: ChatMessage(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[_make_tool_call("call_a"), _make_tool_call("call_b")],
        ),
    )

    def fake_stream_generate(*_args, **_kwargs):  # noqa: ANN001
        yield GenerateResult(
            text="chunk1",
            token=0,
            finish_reason=None,
            prompt_tokens=1,
            generation_tokens=1,
        )
        yield GenerateResult(
            text="chunk2",
            token=1,
            finish_reason="stop",
            prompt_tokens=1,
            generation_tokens=2,
        )

    monkeypatch.setattr(model, "_stream_generate", fake_stream_generate)

    request = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role=Role.USER, content="hi")],
        tools=[],
        stream=True,
        include_thinking_in_content=True,
    )

    chunks = list(model.stream_generate(request))
    assert chunks

    deltas = [
        chunk.choices[0].delta for chunk in chunks if chunk.choices and chunk.choices[0].delta
    ]
    tool_call_indexes = {
        tool_call.id: tool_call.index for delta in deltas for tool_call in (delta.tool_calls or [])
    }

    assert tool_call_indexes["call_a"] == 0
    assert tool_call_indexes["call_b"] == 1
