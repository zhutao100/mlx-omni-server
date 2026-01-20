from mlx_omni_server.chat.schema import (
    ChatCompletionChunk,
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
from mlx_omni_server.responses.schema import ResponseRequest


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
