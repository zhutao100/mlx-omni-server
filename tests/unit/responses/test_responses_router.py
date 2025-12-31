import json
import time
from unittest.mock import patch

import pytest
from httpx import AsyncClient

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
    assert data["id"].startswith("resp_")
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
    assert first.json()["id"] != second.json()["id"]
    assert (
        first.json()["output"][0]["content"][0]["text"]
        == second.json()["output"][0]["content"][0]["text"]
    )
    assert mock_model.call_count == 1


@patch("mlx_omni_server.chat.generation_service._create_text_model")
def test_responses_retrieve_non_stream(mock_create_model, client, response_payload):
    mock_model = MockTextModel()
    mock_create_model.return_value = mock_model

    create = client.post("/v1/responses", json=response_payload)
    assert create.status_code == 200
    created = create.json()

    retrieve = client.get(f"/v1/responses/{created['id']}")
    assert retrieve.status_code == 200
    assert retrieve.json() == created


@patch("mlx_omni_server.chat.generation_service._create_text_model")
def test_responses_delete_response(mock_create_model, client, response_payload):
    mock_model = MockTextModel()
    mock_create_model.return_value = mock_model

    create = client.post("/v1/responses", json=response_payload)
    assert create.status_code == 200
    response_id = create.json()["id"]

    deleted = client.delete(f"/v1/responses/{response_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {"id": response_id, "object": "response", "deleted": True}

    missing = client.get(f"/v1/responses/{response_id}")
    assert missing.status_code == 404


@patch("mlx_omni_server.chat.generation_service._create_text_model")
def test_responses_input_items_pagination(mock_create_model, client):
    mock_model = MockTextModel()
    mock_create_model.return_value = mock_model

    payload = {
        "model": "test-model",
        "input": [
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
            {"role": "user", "content": "three"},
        ],
    }

    create = client.post("/v1/responses", json=payload)
    assert create.status_code == 200
    response_id = create.json()["id"]

    first_page = client.get(f"/v1/responses/{response_id}/input_items?order=asc&limit=2")
    assert first_page.status_code == 200
    first_data = first_page.json()
    assert first_data["object"] == "list"
    assert first_data["has_more"] is True
    assert len(first_data["data"]) == 2
    assert first_data["first_id"] == first_data["data"][0]["id"]
    assert first_data["last_id"] == first_data["data"][-1]["id"]
    assert first_data["data"][0]["type"] == "message"
    assert first_data["data"][0]["role"] == "user"

    after = first_data["data"][1]["id"]
    second_page = client.get(
        f"/v1/responses/{response_id}/input_items?order=asc&limit=2&after={after}"
    )
    assert second_page.status_code == 200
    second_data = second_page.json()
    assert second_data["object"] == "list"
    assert second_data["has_more"] is False
    assert len(second_data["data"]) == 1
    assert second_data["first_id"] == second_data["data"][0]["id"]
    assert second_data["last_id"] == second_data["data"][-1]["id"]


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

    created_event = next(data for event, data in events if event == "response.created")
    assert created_event["response"]["id"].startswith("resp_")

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


def test_responses_reject_conversation(client):
    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "Hello",
            "conversation": "conv_123",
        },
    )
    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["code"] == "invalid_request"


def test_responses_reject_include(client):
    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "Hello",
            "include": ["message.output_text.logprobs"],
        },
    )
    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["code"] == "invalid_request"


class MockFailModel(BaseTextModel):
    def generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel=None,
    ) -> ChatCompletionResponse:
        raise RuntimeError("boom")

    def stream_generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel=None,
    ):
        yield from ()


@patch("mlx_omni_server.chat.generation_service._create_text_model")
def test_responses_non_stream_error_envelope(mock_create_model, client):
    mock_create_model.return_value = MockFailModel()

    response = client.post("/v1/responses", json={"model": "test-model", "input": "Hello"})
    assert response.status_code == 500
    payload = response.json()
    assert payload["error"]["type"] == "server_error"
    assert payload["error"]["code"] == "server_error"
    assert "boom" in payload["error"]["message"]


class MockStreamFailModel(BaseTextModel):
    def generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel=None,
    ) -> ChatCompletionResponse:
        raise RuntimeError("boom")

    def stream_generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel=None,
    ):
        raise RuntimeError("stream boom")


@pytest.mark.asyncio
@patch("mlx_omni_server.chat.generation_service._create_text_model")
async def test_responses_streaming_error_event(mock_create_model, async_client):
    mock_create_model.return_value = MockStreamFailModel()

    payload = {"model": "test-model", "input": "Hello", "stream": True}
    events: list[tuple[str, dict]] = []
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
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and current_event:
                data = json.loads(line.split(":", 1)[1].strip())
                events.append((current_event, data))
                current_event = None

    event_names = [event for event, _ in events]
    assert "response.created" in event_names
    assert "response.in_progress" in event_names
    assert "error" in event_names
    assert "response.completed" in event_names

    error_event = next(data for event, data in events if event == "error")
    assert error_event["code"] == "server_error"

    completed = next(data for event, data in events if event == "response.completed")
    assert completed["response"]["status"] == "failed"
    assert completed["response"]["error"]["code"] == "server_error"


@pytest.mark.asyncio
@patch("mlx_omni_server.chat.generation_service._create_text_model")
async def test_responses_retrieve_stream_replay(mock_create_model, async_client, response_payload):
    mock_model = MockTextModel()
    mock_create_model.return_value = mock_model

    create = await async_client.post("/v1/responses", json=response_payload)
    assert create.status_code == 200
    response_id = create.json()["id"]

    events: list[tuple[str, dict]] = []
    done_sentinel_lines: list[str] = []
    async with async_client.stream(
        "GET",
        f"/v1/responses/{response_id}?stream=true",
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

    assert done_sentinel_lines == []
    event_names = [event for event, _ in events]
    assert event_names[0] == "response.created"
    assert "response.completed" in event_names


class MockHistoryModel(BaseTextModel):
    def __init__(self) -> None:
        self.call_count = 0

    def generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel=None,
    ) -> ChatCompletionResponse:
        self.call_count += 1
        if self.call_count == 2:
            assert any(
                message.role == Role.ASSISTANT and message.content == "first"
                for message in request.messages
            )

        content = "first" if self.call_count == 1 else "second"
        response_id = "resp-first" if self.call_count == 1 else "resp-second"
        return ChatCompletionResponse(
            id=response_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
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
        yield from ()


@patch("mlx_omni_server.chat.generation_service._create_text_model")
def test_responses_previous_response_id_prepends_history(mock_create_model, client):
    mock_model = MockHistoryModel()
    mock_create_model.return_value = mock_model

    first = client.post("/v1/responses", json={"model": "test-model", "input": "hello"})
    assert first.status_code == 200
    first_id = first.json()["id"]

    second = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "follow-up", "previous_response_id": first_id},
    )
    assert second.status_code == 200


class MockCancellableModel(BaseTextModel):
    def generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel=None,
    ) -> ChatCompletionResponse:
        from mlx_lm.generate import GenerationCancelled

        deadline = time.time() + 2
        while should_cancel is not None and not should_cancel() and time.time() < deadline:
            time.sleep(0.01)

        if should_cancel is not None and should_cancel():
            raise GenerationCancelled()

        return ChatCompletionResponse(
            id="resp-cancellable",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "done"},
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
        yield from ()


@pytest.mark.asyncio
@patch("mlx_omni_server.chat.generation_service._create_text_model")
async def test_responses_background_cancel(mock_create_model, async_client: AsyncClient):
    mock_create_model.return_value = MockCancellableModel()

    create = await async_client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "hello", "background": True},
    )
    assert create.status_code == 200
    created = create.json()
    assert created["status"] == "queued"

    cancelled = await async_client.post(f"/v1/responses/{created['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"


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
