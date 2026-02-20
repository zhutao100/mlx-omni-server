from mlx_omni_server.chat.schema import (
    ChatCompletionChunk,
    ChatCompletionUsage,
    ChatMessage,
    FunctionCall,
    Role,
    ToolCall,
)
from mlx_omni_server.responses.adapter import (
    ResponseStreamAdapter,
    response_output_items_to_chat_messages,
    response_request_to_chat_request,
)
from mlx_omni_server.responses.reasoning_envelope import unseal
from mlx_omni_server.responses.schema import ResponseRequest


def test_response_request_to_chat_request_enables_stream_usage_chunk():
    request = ResponseRequest(
        model="test-model",
        stream=True,
        input=[{"type": "message", "role": "user", "content": "Hello"}],
    )

    chat_request = response_request_to_chat_request(request)

    assert chat_request.stream is True
    assert chat_request.stream_options is not None
    assert chat_request.stream_options.include_usage is True


def test_response_request_to_chat_request_drops_empty_assistant_message_items():
    request = ResponseRequest(
        model="test-model",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": ""}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "shell_command",
                "arguments": '{"command":["ls"]}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "done",
            },
        ],
    )

    chat_request = response_request_to_chat_request(request)

    assert [message.role for message in chat_request.messages] == [
        Role.USER,
        Role.ASSISTANT,
        Role.TOOL,
    ]
    assert chat_request.messages[1].tool_calls is not None
    assert chat_request.messages[1].tool_calls[0].id == "call_1"
    assert chat_request.messages[2].tool_call_id == "call_1"


def test_response_request_to_chat_request_coalesces_assistant_message_between_tool_call_and_output():
    request = ResponseRequest(
        model="test-model",
        input=[
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "shell_command",
                "arguments": '{"command":["ls"]}',
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Running…"}],
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "done",
            },
        ],
    )

    chat_request = response_request_to_chat_request(request)

    assert [message.role for message in chat_request.messages] == [
        Role.ASSISTANT,
        Role.TOOL,
    ]
    assert chat_request.messages[0].tool_calls is not None
    assert [tool_call.id for tool_call in chat_request.messages[0].tool_calls] == ["call_1"]
    assert chat_request.messages[0].content == "Running…"
    assert chat_request.messages[1].tool_call_id == "call_1"


def test_response_request_to_chat_request_coalesces_multiple_tool_calls_for_tool_output_order():
    request = ResponseRequest(
        model="test-model",
        input=[
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "a",
                "arguments": "{}",
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "b",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "out1",
            },
            {
                "type": "function_call_output",
                "call_id": "call_2",
                "output": "out2",
            },
        ],
    )

    chat_request = response_request_to_chat_request(request)

    assert [message.role for message in chat_request.messages] == [
        Role.ASSISTANT,
        Role.TOOL,
        Role.TOOL,
    ]
    assert chat_request.messages[0].tool_calls is not None
    assert [tool_call.id for tool_call in chat_request.messages[0].tool_calls] == [
        "call_1",
        "call_2",
    ]
    assert chat_request.messages[1].tool_call_id == "call_1"
    assert chat_request.messages[2].tool_call_id == "call_2"


def test_response_output_items_to_chat_messages_skips_empty_message_items():
    messages = response_output_items_to_chat_messages(
        [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": ""}],
            }
        ]
    )
    assert messages == []

    messages = response_output_items_to_chat_messages(
        [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi"}],
            }
        ]
    )
    assert len(messages) == 1
    assert messages[0].role == Role.ASSISTANT
    assert messages[0].content == "hi"


def test_response_output_items_to_chat_messages_coalesces_tool_call_and_message():
    messages = response_output_items_to_chat_messages(
        [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "shell",
                "arguments": '{"command":["ls"]}',
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Tool call ready"}],
            },
        ]
    )

    assert len(messages) == 1
    assert messages[0].role == Role.ASSISTANT
    assert messages[0].tool_calls is not None
    assert [tool_call.id for tool_call in messages[0].tool_calls] == ["call_1"]
    assert messages[0].content == "Tool call ready"


def test_response_output_items_to_chat_messages_coalesces_multiple_tool_calls():
    messages = response_output_items_to_chat_messages(
        [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "a",
                "arguments": "{}",
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "b",
                "arguments": "{}",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ready"}],
            },
        ]
    )

    assert len(messages) == 1
    assert messages[0].role == Role.ASSISTANT
    assert messages[0].tool_calls is not None
    assert [tool_call.id for tool_call in messages[0].tool_calls] == [
        "call_1",
        "call_2",
    ]
    assert messages[0].content == "ready"


def test_response_stream_adapter_skips_empty_text_chunk_before_tool_call():
    adapter = ResponseStreamAdapter(response_id="resp_test", model="test-model")

    empty_text_chunk = ChatCompletionChunk(
        id="chatcmpl-test",
        created=0,
        model="test-model",
        choices=[
            {
                "index": 0,
                "delta": ChatMessage(role=Role.ASSISTANT, content=""),
                "finish_reason": None,
            }
        ],
    )
    adapter.on_chunk(empty_text_chunk)

    tool_call_chunk = ChatCompletionChunk(
        id="chatcmpl-test",
        created=0,
        model="test-model",
        choices=[
            {
                "index": 0,
                "delta": ChatMessage(
                    role=Role.ASSISTANT,
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            function=FunctionCall(name="shell", arguments='{"command":["ls"]}'),
                        )
                    ],
                ),
                "finish_reason": "tool_calls",
            }
        ],
    )
    adapter.on_chunk(tool_call_chunk)

    events = adapter.on_done()
    completed = next(event for event in events if event.event == "response.completed")
    output = completed.data["response"]["output"]

    assert not any(item.get("type") == "message" for item in output)


def test_response_stream_adapter_clamps_reasoning_tokens_from_upstream_usage():
    adapter = ResponseStreamAdapter(response_id="resp_test", model="test-model")

    usage_chunk = ChatCompletionChunk(
        id="chatcmpl-test",
        created=0,
        model="test-model",
        choices=[
            {
                "index": 0,
                "delta": ChatMessage(role=Role.ASSISTANT),
                "finish_reason": None,
            }
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=1,
            completion_tokens=5,
            total_tokens=6,
            completion_tokens_details={"reasoning_tokens": 10},
        ),
    )
    adapter.on_chunk(usage_chunk)

    events = adapter.on_done()
    completed = next(event for event in events if event.event == "response.completed")
    usage = completed.data["response"]["usage"]

    assert usage["output_tokens"] == 5
    assert usage["output_tokens_details"]["reasoning_tokens"] == 5


def test_response_request_to_chat_request_maps_reasoning_effort_to_thinking_params():
    request = ResponseRequest(
        model="test-model",
        input="Hello",
        reasoning={"effort": "low"},
    )

    chat_request = response_request_to_chat_request(request)
    extra = chat_request.get_extra_params()

    assert extra["enable_thinking"] is True
    assert extra["thinking_budget"] == 512


def test_response_request_to_chat_request_does_not_override_explicit_thinking_params():
    request = ResponseRequest(
        model="test-model",
        input="Hello",
        reasoning={"effort": "none"},
        enable_thinking=True,
        thinking_budget=999,
    )

    chat_request = response_request_to_chat_request(request)
    extra = chat_request.get_extra_params()

    assert extra["enable_thinking"] is True
    assert extra["thinking_budget"] == 999


def test_response_stream_adapter_emits_reasoning_deltas_without_include():
    adapter = ResponseStreamAdapter(
        response_id="resp_test",
        model="test-model",
        request_echo={"include": []},
    )

    # Seed a message item so reasoning output_index is non-zero.
    adapter.on_chunk(
        ChatCompletionChunk(
            id="chatcmpl-test",
            created=123,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "delta": ChatMessage(role=Role.ASSISTANT, content="hi"),
                    "finish_reason": None,
                }
            ],
        )
    )

    events1 = adapter.on_chunk(
        ChatCompletionChunk(
            id="chatcmpl-test",
            created=123,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "delta": ChatMessage(role=Role.ASSISTANT, reasoning="a"),
                    "finish_reason": None,
                }
            ],
        )
    )
    events2 = adapter.on_chunk(
        ChatCompletionChunk(
            id="chatcmpl-test",
            created=123,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "delta": ChatMessage(role=Role.ASSISTANT, reasoning="b"),
                    "finish_reason": None,
                }
            ],
        )
    )
    done_events = adapter.on_done()

    all_events = [*events1, *events2, *done_events]

    reasoning_added = [
        event
        for event in all_events
        if event.event == "response.output_item.added"
        and event.data.get("item", {}).get("type") == "reasoning"
    ]
    assert reasoning_added
    added_item = reasoning_added[0].data["item"]
    assert added_item["content"] == [{"type": "reasoning_text", "text": ""}]
    assert added_item["summary"] == []

    deltas = [
        event.data["delta"]
        for event in all_events
        if event.event == "response.reasoning_text.delta"
    ]
    assert deltas == ["a", "b"]
    assert all(
        event.data.get("content_index") == 0
        for event in all_events
        if event.event == "response.reasoning_text.delta"
    )

    reasoning_done = next(
        event for event in done_events if event.event == "response.reasoning_text.done"
    )
    assert reasoning_done.data["content_index"] == 0
    assert reasoning_done.data["text"] == "ab"

    reasoning_output_index = reasoning_done.data["output_index"]
    completed = next(event for event in done_events if event.event == "response.completed")
    output = completed.data["response"]["output"]

    assert output[reasoning_output_index]["type"] == "reasoning"
    assert output[reasoning_output_index]["content"] == [{"type": "reasoning_text", "text": "ab"}]
    assert output[reasoning_output_index]["summary"] == []
    assert "encrypted_content" not in output[reasoning_output_index]


def test_response_stream_adapter_reasoning_encrypted_content_remains_include_gated():
    adapter = ResponseStreamAdapter(
        response_id="resp_test",
        model="test-model",
        request_echo={"include": ["reasoning.encrypted_content"]},
    )

    adapter.on_chunk(
        ChatCompletionChunk(
            id="chatcmpl-test",
            created=123,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "delta": ChatMessage(role=Role.ASSISTANT, reasoning="a"),
                    "finish_reason": None,
                }
            ],
        )
    )
    adapter.on_chunk(
        ChatCompletionChunk(
            id="chatcmpl-test",
            created=123,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "delta": ChatMessage(role=Role.ASSISTANT, reasoning="b"),
                    "finish_reason": None,
                }
            ],
        )
    )

    done_events = adapter.on_done()
    completed = next(event for event in done_events if event.event == "response.completed")
    reasoning_item = next(
        item for item in completed.data["response"]["output"] if item["type"] == "reasoning"
    )

    assert "encrypted_content" in reasoning_item
    envelope = unseal(reasoning_item["encrypted_content"])
    assert envelope.model == "test-model"
    assert envelope.reasoning == "ab"


def test_response_stream_adapter_emits_reasoning_delta_at_done_when_missing():
    adapter = ResponseStreamAdapter(
        response_id="resp_test",
        model="test-model",
        request_echo={"include": []},
    )
    adapter._reasoning_by_choice[0] = "ab"

    events = adapter.on_done()

    added_idx = next(
        idx
        for idx, event in enumerate(events)
        if event.event == "response.output_item.added"
        and event.data.get("item", {}).get("type") == "reasoning"
    )
    delta_idx = next(
        idx for idx, event in enumerate(events) if event.event == "response.reasoning_text.delta"
    )
    assert added_idx < delta_idx

    delta_event = events[delta_idx]
    assert delta_event.data["delta"] == "ab"

    done_event = next(event for event in events if event.event == "response.reasoning_text.done")
    assert done_event.data["text"] == "ab"


def test_response_stream_adapter_tool_call_interleaving_does_not_drop_reasoning():
    adapter = ResponseStreamAdapter(response_id="resp_test", model="test-model")

    events1 = adapter.on_chunk(
        ChatCompletionChunk(
            id="chatcmpl-test",
            created=123,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "delta": ChatMessage(role=Role.ASSISTANT, reasoning="a"),
                    "finish_reason": None,
                }
            ],
        )
    )
    events2 = adapter.on_chunk(
        ChatCompletionChunk(
            id="chatcmpl-test",
            created=123,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "delta": ChatMessage(
                        role=Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(
                                id="call_1",
                                function=FunctionCall(name="shell", arguments="{}"),
                            )
                        ],
                    ),
                    "finish_reason": "tool_calls",
                }
            ],
        )
    )
    events3 = adapter.on_chunk(
        ChatCompletionChunk(
            id="chatcmpl-test",
            created=123,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "delta": ChatMessage(role=Role.ASSISTANT, reasoning="b"),
                    "finish_reason": None,
                }
            ],
        )
    )
    done_events = adapter.on_done()

    reasoning_deltas = [
        event.data["delta"]
        for event in [*events1, *events2, *events3, *done_events]
        if event.event == "response.reasoning_text.delta"
    ]
    assert reasoning_deltas == ["a", "b"]
