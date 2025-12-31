import json
import time
from unittest.mock import patch

import pytest

from mlx_omni_server.chat.schema import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    FunctionCall,
    Role,
    ToolCall,
)
from mlx_omni_server.chat.text_models import BaseTextModel
from mlx_omni_server.responses.adapter import response_request_to_chat_request
from mlx_omni_server.responses.schema import ResponseRequest


class MockTextModel(BaseTextModel):
    """Mock text model for responses tests."""

    def __init__(self):
        self.call_count = 0
        self.stream_call_count = 0

    def generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel=None,
    ) -> ChatCompletionResponse:
        self.call_count += 1
        return ChatCompletionResponse(
            id="resp-id",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello, world!"},
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
        )

    def stream_generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel=None,
    ):
        self.stream_call_count += 1
        chunk1 = ChatCompletionChunk(
            id="resp-id",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "delta": ChatMessage(role=Role.ASSISTANT, content="Hello"),
                    "finish_reason": None,
                }
            ],
        )
        yield chunk1

        chunk2 = ChatCompletionChunk(
            id="resp-id",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "delta": ChatMessage(role=Role.ASSISTANT, content=" there!"),
                    "finish_reason": "stop",
                }
            ],
        )
        yield chunk2


class MockToolCallModel(BaseTextModel):
    def generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel=None,
    ) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id="resp-id",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "shell",
                                    "arguments": '{"command":["ls"]}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
        )

    def stream_generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel=None,
    ):
        chunk1 = ChatCompletionChunk(
            id="resp-id",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "delta": ChatMessage(
                        role=Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(
                                id="call_1",
                                function=FunctionCall(
                                    name="shell",
                                    arguments='{"command":["ls"]',
                                ),
                            )
                        ],
                    ),
                    "finish_reason": None,
                }
            ],
        )
        yield chunk1

        chunk2 = ChatCompletionChunk(
            id="resp-id",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "delta": ChatMessage(
                        role=Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(
                                id="call_1",
                                function=FunctionCall(
                                    name="shell",
                                    arguments="}",
                                ),
                            )
                        ],
                    ),
                    "finish_reason": "tool_calls",
                }
            ],
        )
        yield chunk2


@pytest.fixture
def response_payload():
    return {
        "model": "test-model",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Say hi"},
                ],
            }
        ],
    }


@patch("mlx_omni_server.chat.generation_service._create_text_model")
def test_responses_non_stream(mock_create_model, client, response_payload):
    mock_model = MockTextModel()
    mock_create_model.return_value = mock_model

    response = client.post("/v1/responses", json=response_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["output"][0]["content"][0]["text"] == "Hello, world!"
    assert "x-idempotent-replay" not in response.headers
    assert mock_model.call_count == 1


@patch("mlx_omni_server.chat.generation_service._create_text_model")
def test_responses_non_stream_cache(mock_create_model, client, response_payload):
    mock_model = MockTextModel()
    mock_create_model.return_value = mock_model

    first = client.post("/v1/responses", json=response_payload)
    second = client.post("/v1/responses", json=response_payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.headers["x-idempotent-replay"] == "true"
    assert first.json() == second.json()
    assert mock_model.call_count == 1


@pytest.mark.asyncio
@patch("mlx_omni_server.chat.generation_service._create_text_model")
async def test_responses_streaming(mock_create_model, async_client, response_payload):
    mock_model = MockTextModel()
    mock_create_model.return_value = mock_model

    payload = {**response_payload, "stream": True}
    events = []
    done_sentinel_lines: list[str] = []

    async with async_client.stream(
        "POST",
        "/v1/responses",
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    ) as response:
        assert response.status_code == 200
        current_event = None
        async for line in response.aiter_lines():
            if not line:
                continue
            if line.strip() == "data: [DONE]":
                done_sentinel_lines.append(line)
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and current_event:
                data = json.loads(line.split(":", 1)[1].strip())
                events.append((current_event, data))
                current_event = None

    event_names = [event for event, _ in events]
    assert "response.created" in event_names
    assert "response.output_item.added" in event_names
    assert "response.content_part.added" in event_names
    assert "response.output_text.delta" in event_names
    assert "response.output_text.done" in event_names
    assert "response.content_part.done" in event_names
    assert "response.output_item.done" in event_names
    assert "response.completed" in event_names
    assert done_sentinel_lines == []

    deltas = [data["delta"] for event, data in events if event == "response.output_text.delta"]
    assert "".join(deltas) == "Hello there!"

    completed_event = next(data for event, data in events if event == "response.completed")
    assert completed_event["response"]["output"][0]["content"][0]["text"] == "Hello there!"
    assert mock_model.stream_call_count == 1


@pytest.mark.asyncio
@patch("mlx_omni_server.chat.generation_service._create_text_model")
async def test_responses_streaming_tool_call(mock_create_model, async_client):
    mock_model = MockToolCallModel()
    mock_create_model.return_value = mock_model

    payload = {
        "model": "test-model",
        "input": "Hello",
        "stream": True,
    }

    events = []
    done_sentinel_lines: list[str] = []
    async with async_client.stream(
        "POST",
        "/v1/responses",
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    ) as response:
        assert response.status_code == 200
        current_event = None
        async for line in response.aiter_lines():
            if not line:
                continue
            if line.strip() == "data: [DONE]":
                done_sentinel_lines.append(line)
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and current_event:
                data = json.loads(line.split(":", 1)[1].strip())
                events.append((current_event, data))
                current_event = None

    event_names = [event for event, _ in events]
    assert "response.output_item.added" in event_names
    assert "response.function_call_arguments.delta" in event_names
    assert "response.function_call_arguments.done" in event_names
    assert "response.output_item.done" in event_names
    assert "response.completed" in event_names
    assert done_sentinel_lines == []

    deltas = [
        data["delta"] for event, data in events if event == "response.function_call_arguments.delta"
    ]
    assert "".join(deltas) == '{"command":["ls"]}'

    completed_event = next(data for event, data in events if event == "response.completed")
    output_item = completed_event["response"]["output"][0]
    assert output_item["type"] == "function_call"
    assert output_item["arguments"] == '{"command":["ls"]}'
    assert output_item["name"] == "shell"


def test_response_request_to_chat_request_text_format_json_schema():
    request = ResponseRequest(
        model="test-model",
        input="Hello",
        text={
            "format": {
                "type": "json_schema",
                "name": "greeting",
                "schema": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
                "strict": True,
            }
        },
    )

    chat_request = response_request_to_chat_request(request)

    assert chat_request.response_format is not None
    assert chat_request.response_format.type == "json_schema"
    assert chat_request.response_format.json_schema is not None
    assert chat_request.response_format.json_schema.name == "greeting"
    assert chat_request.response_format.json_schema.schema_def == {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }


def test_response_request_to_chat_request_text_only():
    request = ResponseRequest(
        model="test-model",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Hello"},
                    {"type": "input_text", "text": "Second message"},
                ],
            }
        ],
        instructions="You are helpful",
        tools=[{"type": "function", "function": {"name": "noop", "parameters": {}}}],
        tool_choice="auto",
    )

    chat_request = response_request_to_chat_request(request)

    assert chat_request.model == "test-model"
    assert chat_request.tool_choice == "auto"
    assert chat_request.tools is not None
    assert chat_request.messages[0].role == Role.SYSTEM
    assert chat_request.messages[0].content == "You are helpful"
    assert chat_request.messages[1].role == Role.USER
    assert chat_request.messages[1].content == "Hello\n\nSecond message"


def test_response_request_to_chat_request_tool_normalization():
    request = ResponseRequest(
        model="test-model",
        input="Hello",
        tools=[
            {
                "type": "function",
                "name": "shell",
                "description": "run command",
                "parameters": {"type": "object"},
            }
        ],
    )

    chat_request = response_request_to_chat_request(request)

    assert chat_request.tools is not None
    first_tool = chat_request.tools[0]
    assert first_tool.type.value == "function"
    assert first_tool.function.name == "shell"
    assert first_tool.function.description == "run command"


def test_response_request_to_chat_request_with_history():
    request = ResponseRequest(
        model="test-model",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Hello"},
                ],
            },
            {
                "type": "function_call",
                "id": "call_1",
                "function": {
                    "name": "shell",
                    "arguments": '{"command":["ls"]}',
                },
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "done",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "completed"},
                ],
            },
        ],
    )

    chat_request = response_request_to_chat_request(request)

    assert chat_request.messages[0].role == Role.USER
    assert chat_request.messages[0].content == "Hello"

    assistant_call = chat_request.messages[1]
    assert assistant_call.role == Role.ASSISTANT
    assert assistant_call.tool_calls is not None
    assert assistant_call.tool_calls[0].function.arguments == '{"command":["ls"]}'

    tool_response = chat_request.messages[2]
    assert tool_response.role == Role.TOOL
    assert tool_response.content == "done"
    assert tool_response.tool_call_id == "call_1"

    final_message = chat_request.messages[3]
    assert final_message.role == Role.ASSISTANT
    assert final_message.content == "completed"
