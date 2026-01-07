from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from mlx_lm.tokenizer_utils import TokenizerWrapper

from mlx_omni_server.chat.generation_service import _restore_tool_loop_reasoning
from mlx_omni_server.chat.mlx_lm.mlx_lm_model import MlxLmModel
from mlx_omni_server.chat.schema import (
    ChatCompletionRequest,
    ChatMessage,
    Function,
    FunctionCall,
    FunctionParameters,
    Role,
    Tool,
    ToolCall,
    ToolType,
)
from mlx_omni_server.chat.text_models import GenerateResult
from mlx_omni_server.chat.tool_loop_reasoning_cache import tool_loop_reasoning_cache


def test_chat_message_reasoning_content_alias_roundtrips() -> None:
    message = ChatMessage.model_validate(
        {"role": "assistant", "content": "hi", "reasoning_content": "r"}
    )
    assert message.reasoning == "r"
    dumped = message.model_dump(exclude_none=True)
    assert dumped["reasoning"] == "r"
    assert dumped["reasoning_content"] == "r"


def test_restore_tool_loop_reasoning_injects_cached_reasoning() -> None:
    call_id = "call_test"
    tool_call = ToolCall(
        id=call_id,
        type=ToolType.FUNCTION,
        function=FunctionCall(name="f", arguments="{}"),
    )
    assistant = ChatMessage(role=Role.ASSISTANT, content=None, tool_calls=[tool_call])
    tool = ChatMessage(role=Role.TOOL, content="ok", tool_call_id=call_id)
    request = ChatCompletionRequest(model="test-model", messages=[assistant, tool])

    tool_loop_reasoning_cache.set(call_id, "cached reasoning")
    _restore_tool_loop_reasoning(request)

    assert assistant.reasoning == "cached reasoning"


def _make_glm4_model() -> MlxLmModel:
    mock_tokenizer = Mock(spec=TokenizerWrapper)
    model_cache = SimpleNamespace(
        model_type="glm4",
        tokenizer=mock_tokenizer,
        model_config={"max_position_embeddings": 128},
        draft_model=None,
    )
    model = MlxLmModel(model_cache)
    model._reasoning_decoder.enable_thinking = True
    return model


def _make_tool() -> Tool:
    return Tool(
        type=ToolType.FUNCTION,
        function=Function(
            name="get_current_weather",
            description="Get the current weather",
            parameters=FunctionParameters(
                type="object",
                properties={"location": {"type": "string"}},
                required=["location"],
            ),
        ),
    )


def test_mlx_lm_non_stream_tool_call_includes_reasoning(monkeypatch) -> None:
    model = _make_glm4_model()
    tool = _make_tool()
    request = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role=Role.USER, content="hi")],
        tools=[tool],
    )

    completion = (
        "<think>abc</think>"
        "<tool_call>get_current_weather"
        "<arg_key>location</arg_key><arg_value>Boston</arg_value>"
        "</tool_call>"
    )

    def fake_stream_generate(*_args, **_kwargs):  # noqa: ANN001
        yield GenerateResult(
            text=completion,
            token=0,
            finish_reason="stop",
            prompt_tokens=1,
            generation_tokens=1,
        )

    monkeypatch.setattr(model, "_stream_generate", fake_stream_generate)

    response = model.generate(request)
    message = response.choices[0].message
    assert message.tool_calls
    assert message.reasoning == "abc"
    assert message.reasoning_content == "abc"

    call_id = message.tool_calls[0].id
    assert tool_loop_reasoning_cache.get(call_id) == "abc"


def test_mlx_lm_stream_final_tool_call_chunk_includes_reasoning(monkeypatch) -> None:
    model = _make_glm4_model()
    tool = _make_tool()
    request = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role=Role.USER, content="hi")],
        tools=[tool],
        stream=True,
    )

    chunks = [
        "<think>",
        "abc",
        "</think>",
        "<tool_call>get_current_weather",
        "<arg_key>location</arg_key>",
        "<arg_value>Boston</arg_value>",
        "</tool_call>",
    ]

    def fake_stream_generate(*_args, **_kwargs):  # noqa: ANN001
        for idx, text in enumerate(chunks):
            yield GenerateResult(
                text=text,
                token=idx,
                finish_reason=("stop" if idx == len(chunks) - 1 else None),
                prompt_tokens=1,
                generation_tokens=idx + 1,
            )

    monkeypatch.setattr(model, "_stream_generate", fake_stream_generate)

    streamed = list(model.stream_generate(request))
    final_delta = streamed[-1].choices[0].delta
    assert final_delta.tool_calls
    assert final_delta.reasoning == "abc"
    assert final_delta.reasoning_content == "abc"

    call_id = final_delta.tool_calls[0].id
    assert tool_loop_reasoning_cache.get(call_id) == "abc"
