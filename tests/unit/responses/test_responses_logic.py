import asyncio
import json
import logging
import time
from contextlib import contextmanager
from typing import Iterator
from unittest.mock import patch

import pytest
from httpx import AsyncClient
from openai import AsyncOpenAI, OpenAI

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockResponsesModel(BaseTextModel):
    """Mock model producing both text and tool call outputs."""

    def generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel=None,
    ) -> ChatCompletionResponse:
        tool_call = {
            "id": "call_sync",
            "type": "function",
            "function": {
                "name": "shell",
                "arguments": json.dumps({"command": ["ls"]}, separators=(",", ":")),
            },
        }

        return ChatCompletionResponse(
            id="resp-id",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [tool_call],
                        "content": "Tool call ready",
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
        # Emit tool call delta in multiple pieces followed by text
        chunk_tool_start = ChatCompletionChunk(
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
                                id="call_stream",
                                function=FunctionCall(
                                    name="shell",
                                    arguments='{"command":["ls"',
                                ),
                            )
                        ],
                    ),
                    "finish_reason": None,
                }
            ],
        )
        yield chunk_tool_start

        chunk_tool_end = ChatCompletionChunk(
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
                                id="call_stream",
                                function=FunctionCall(
                                    name="shell",
                                    arguments="]}",
                                ),
                            )
                        ],
                    ),
                    "finish_reason": "tool_calls",
                }
            ],
        )
        yield chunk_tool_end

        chunk_text = ChatCompletionChunk(
            id="resp-id",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 1,
                    "delta": ChatMessage(role=Role.ASSISTANT, content="Tool call complete"),
                    "finish_reason": "stop",
                }
            ],
        )
        yield chunk_text


class MockTextStreamModel(BaseTextModel):
    def generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel=None,
    ) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id="resp-text",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello world",
                    },
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            },
        )

    def stream_generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel=None,
    ):
        chunk = ChatCompletionChunk(
            id="resp-text",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "delta": ChatMessage(role=Role.ASSISTANT, content="Hello world"),
                    "finish_reason": "stop",
                }
            ],
        )
        yield chunk


class MockSequentialToolModel(BaseTextModel):
    def generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel=None,
    ) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id="resp-seq",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Plan step",
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            usage={
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "total_tokens": 15,
            },
        )

    def stream_generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel=None,
    ):
        # First text before first tool call
        yield ChatCompletionChunk(
            id="resp-seq",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "delta": ChatMessage(role=Role.ASSISTANT, content="Step 1"),
                    "finish_reason": None,
                }
            ],
        )

        # Tool call 1 with text prefix
        yield ChatCompletionChunk(
            id="resp-seq",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "delta": ChatMessage(
                        role=Role.ASSISTANT,
                        content="Step 2",
                        tool_calls=[
                            ToolCall(
                                id="call_a",
                                function=FunctionCall(
                                    name="shell",
                                    arguments='{"command":["ls"]}',
                                ),
                            )
                        ],
                    ),
                    "finish_reason": "tool_calls",
                }
            ],
        )

        # Tool call 2 with text prefix
        yield ChatCompletionChunk(
            id="resp-seq",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "delta": ChatMessage(
                        role=Role.ASSISTANT,
                        content="Step 3",
                        tool_calls=[
                            ToolCall(
                                id="call_b",
                                function=FunctionCall(
                                    name="shell",
                                    arguments='{"command":["pwd"]}',
                                ),
                            )
                        ],
                    ),
                    "finish_reason": "tool_calls",
                }
            ],
        )


class MockResponsesExtraBodyModel(BaseTextModel):
    """Mock model for testing extra body parameters."""

    def generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel=None,
    ) -> ChatCompletionResponse:
        # Access extra parameters to verify they're passed through
        extra_params = request.get_extra_params()
        top_k = extra_params.get("top_k", 40)  # Default value from mlx_lm

        return ChatCompletionResponse(
            id="resp-extra-body",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Extra body param top_k: {top_k}",
                    },
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15,
            },
        )

    def stream_generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel=None,
    ):
        extra_params = request.get_extra_params()
        top_k = extra_params.get("top_k", 40)

        yield ChatCompletionChunk(
            id="resp-extra-body",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "delta": ChatMessage(
                        role=Role.ASSISTANT, content=f"Extra body param top_k: {top_k}"
                    ),
                    "finish_reason": "stop",
                }
            ],
        )


@contextmanager
def patched_model(mock):
    with patch(
        "mlx_omni_server.chat.generation_service._create_text_model",
        return_value=mock,
    ):
        yield


class TestResponsesUnit:

    def test_responses_non_stream_integration(self, client):
        mock_model = MockResponsesModel()
        with patched_model(mock_model):
            openai_client = OpenAI(
                base_url="http://test/v1",
                api_key="test",
                http_client=client,
            )
            response = openai_client.responses.create(
                model="test-model",
                input="Hello",
            )

        assert response.id.startswith("resp_")
        assert response.status == "completed"
        assert response.output[0].type == "function_call"
        assert response.output[0].arguments == '{"command":["ls"]}'
        assert response.output[1].type == "message"
        assert response.output[1].content[0].text == "Tool call ready"

    @pytest.mark.asyncio
    async def test_responses_streaming_integration(self, async_client):
        mock_model = MockResponsesModel()
        with patched_model(mock_model):
            async with AsyncOpenAI(
                base_url="http://test/v1",
                api_key="test",
                http_client=async_client,
            ) as client:
                events = []
                async with client.responses.stream(
                    model="test-model",
                    input="Hello",
                ) as stream:
                    async for event in stream:
                        events.append(event)

                    final = await stream.get_final_response()

        event_types = [event.type for event in events]
        assert "response.output_item.added" in event_types
        assert "response.function_call_arguments.delta" in event_types
        assert "response.function_call_arguments.done" in event_types
        assert "response.output_item.done" in event_types
        assert "response.output_text.delta" in event_types
        assert "response.output_text.done" in event_types
        assert "response.content_part.done" in event_types
        assert event_types[-1] == "response.completed"

        deltas = [
            event.delta
            for event in events
            if event.type == "response.function_call_arguments.delta"
        ]
        assert "".join(deltas) == '{"command":["ls"]}'

        function_call = next(item for item in final.output if item.type == "function_call")
        assert function_call.arguments == '{"command":["ls"]}'

        message_item = next(item for item in final.output if item.type == "message")
        assert message_item.content[0].text == "Tool call complete"

    @pytest.mark.asyncio
    async def test_responses_streaming_text_no_duplicate(self, async_client):
        mock_model = MockTextStreamModel()
        with patched_model(mock_model):
            async with AsyncOpenAI(
                base_url="http://test/v1",
                api_key="test",
                http_client=async_client,
            ) as client:
                deltas = []
                async with client.responses.stream(
                    model="test-model",
                    input="Hi",
                ) as stream:
                    async for event in stream:
                        if event.type == "response.output_text.delta":
                            deltas.append(event.delta)

        assert deltas == ["Hello world"]

    @pytest.mark.asyncio
    async def test_responses_streaming_sequential_tool_calls(self, async_client):
        mock_model = MockSequentialToolModel()
        with patched_model(mock_model):
            async with AsyncOpenAI(
                base_url="http://test/v1",
                api_key="test",
                http_client=async_client,
            ) as client:
                deltas = []
                tool_events = []
                async with client.responses.stream(
                    model="test-model",
                    input="Hi",
                ) as stream:
                    async for event in stream:
                        if event.type == "response.output_text.delta":
                            deltas.append(event.delta)
                        if event.type == "response.function_call_arguments.delta":
                            tool_events.append(event.delta)

        assert deltas == ["Step 1", "Step 2", "Step 3"]
        assert tool_events == ['{"command":["ls"]}', '{"command":["pwd"]}']
