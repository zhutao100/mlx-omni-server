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

from mlx_omni_server.chat.schema import (ChatCompletionChunk,
                                         ChatCompletionRequest,
                                         ChatCompletionResponse, ChatMessage,
                                         FunctionCall, Role, ToolCall)
from mlx_omni_server.chat.text_models import BaseTextModel

MODEL = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockResponsesModel(BaseTextModel):
    """Mock model producing both text and tool call outputs."""

    def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
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

    def stream_generate(self, request: ChatCompletionRequest):  # type: ignore[override]
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
                                    arguments=']}',
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
    def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
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

    def stream_generate(self, request: ChatCompletionRequest):  # type: ignore[override]
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
    def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
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

    def stream_generate(self, request: ChatCompletionRequest):  # type: ignore[override]
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

    def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
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

    def stream_generate(self, request: ChatCompletionRequest):
        extra_params = request.get_extra_params()
        top_k = extra_params.get("top_k", 40)

        yield ChatCompletionChunk(
            id="resp-extra-body",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "delta": ChatMessage(role=Role.ASSISTANT, content=f"Extra body param top_k: {top_k}"),
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

        assert response.id == "resp-id"
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
        assert event_types[-1] == "response.completed"

        deltas = [event.delta for event in events if event.type == "response.function_call_arguments.delta"]
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


class TestResponsesIntegration:

    def test_responses_normal(self, openai_client):
        """Test basic non-streaming responses functionality"""
        try:
            response = openai_client.responses.create(
                model=MODEL,
                input=[{"role": "user", "content": "hello"}],
            )
            logger.info(f"Responses Response:\n{response}\n")

            # Validate response
            assert response.model == MODEL, "Model name is not correct"
            assert response.usage is not None, "No usage in response"
            assert response.object == "response", "Incorrect response object type"
            assert response.status == "completed", "Response status is not completed"
            assert len(response.output) > 0, "No output in response"

            # Check that we have a message output
            message_output = next((item for item in response.output if item.type == "message"), None)
            assert message_output is not None, "No message output found"
            assert len(message_output.content) > 0, "No content in message output"
            assert message_output.content[0].text.strip(), "Incorrect content in message output"
        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_responses_extra_body(self, openai_client):
        """Test responses with extra body parameters"""
        try:
            response = openai_client.responses.create(
                model=MODEL,
                input=[{"role": "user", "content": "hello"}],
                max_output_tokens=50,
                extra_body={
                    "top_k": 50,
                    "min_p": 0.0,
                    "min_tokens_to_keep": 1,
                },
            )
            logger.info(f"Responses Response with extra body:\n{response}\n")

            # Validate response
            assert response.model == MODEL, "Model name is not correct"
            assert response.usage is not None, "No usage in response"
            assert response.object == "response", "Incorrect response object type"
            assert response.status == "completed", "Response status is not completed"

            # Check that extra body parameters were processed
            message_output = next((item for item in response.output if item.type == "message"), None)
            assert message_output is not None, "No message output found"
            assert message_output.content[0].text.strip(), "Extra body parameter not processed correctly"
        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_responses_draft_model(self, openai_client):
        try:
            model = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr5e-8"
            response = openai_client.responses.create(
                model=model,
                input=[{"role": "user", "content": "hello"}],
                max_output_tokens=50,
                extra_body={
                    "draft-model": MODEL,
                },
            )
            logger.info(f"Responses Response with draft model:\n{response}\n")

            # Validate response
            assert response.model == model, "Model name is not correct"
            assert response.usage is not None, "No usage in response"
            assert response.object == "response", "Incorrect response object type"
            assert response.status == "completed", "Response status is not completed"
            assert len(response.output) > 0, "No output in response"
            message_output = next((item for item in response.output if item.type == "message"), None)
            assert message_output is not None, "No message output found"
            assert len(message_output.content) > 0, "No content in message output"
            assert message_output.content[0].text.strip(), "Generated content is empty"

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_responses_stream(self, openai_client):
        """Test basic streaming responses functionality"""
        try:
            events = []
            with openai_client.responses.stream(
                model=MODEL,
                input=[{"role": "user", "content": "hi"}]
            ) as stream:
                for event in stream:
                    events.append(event)
                # Get final response
                final = stream.get_final_response()

            logger.info(f"Received {len(events)} stream events")

            # Validate events
            event_types = [event.type for event in events]
            assert "response.created" in event_types, "No response.created event received"
            assert "response.completed" in event_types, "No response.completed event received"
            assert len(events) > 0, "No events received"

            # Check for text delta events
            text_deltas = [event.delta for event in events if event.type == "response.output_text.delta"]
            content = "".join(text_deltas)

            assert content.strip(), "Generated content is empty"
            logger.info(f"Complete generated content: {content}")

            # Validate final response
            assert final.status == "completed", "Final response status is not completed"
            assert len(final.output) > 0, "No output in final response"

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_responses_stream_with_options(self, openai_client):
        """Test streaming responses with additional options"""
        try:
            events = []
            with openai_client.responses.stream(
                model=MODEL,
                input=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Write a short greeting."},
                ],
                max_output_tokens=200,
            ) as stream:
                for event in stream:
                    events.append(event)
                # Get final response
                final = stream.get_final_response()

            logger.info(f"Received {len(events)} stream events")

            # Validate events
            assert len(events) > 0, "No events received"

            # Collect different types of events
            text_deltas = [event.delta for event in events if event.type == "response.output_text.delta"]

            assert len(text_deltas) > 0, "No text delta events received"

            # Check content
            text_content = "".join(text_deltas)
            assert text_content.strip(), "Missing expected text content"

            logger.info(f"Complete generated text content: {text_content}")

            # Validate final response
            assert final.status == "completed", "Final response status is not completed"
            assert len(final.output) > 0, "No output in final response"

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    @pytest.mark.asyncio
    async def test_retry_canceled_stream_responses(self, async_client: AsyncClient):
        """
        Tests that retrying a canceled streaming request for responses starts a new generation.
        """
        payload = {
            "model": MODEL,
            "input": [
                {"role": "user", "content": "Write a detailed essay about the history of artificial intelligence and machine learning."}
            ],
            "stream": True,
            "max_output_tokens": 500,  # Make it longer so we have time to cancel
        }

        # --- First request, which we will cancel ---
        lines_received = []
        is_cancelled = False
        logger.info("\n--- Starting first (canceled) request ---")
        try:
            async with async_client.stream("POST", "/v1/responses", json=payload, timeout=5) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line and line.startswith("data:"):
                        lines_received.append(line)
                        if len(lines_received) >= 5:
                            logger.info("--- Canceling first request by breaking early ---")
                            is_cancelled = True
                            break
        except Exception as e:
            logger.info(f"--- First request terminated as expected: {e} ---")
            pass

        assert is_cancelled, "Test failed to cancel the first stream mid-generation."

        full_first_response = "\n".join(lines_received)
        assert 'data: {"type":"response.completed"}' not in full_first_response, "Canceled stream should not be complete"

        # Give the server a moment to process the disconnection
        await asyncio.sleep(1)

        # --- Second request, which should succeed ---
        all_lines = []
        logger.info("--- Starting second (retry) request ---")
        async with async_client.stream("POST", "/v1/responses", json=payload, timeout=15) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    all_lines.append(line)

        full_response = "\n".join(all_lines)
        logger.info(f"--- Full response from second request: ---{full_response}")

        # The second request should complete successfully
        assert 'response.completed' in full_response, "Full stream should end with completed event"

        # Verify the content of the full stream
        assert len(all_lines) > len(lines_received), "Second request should return more lines than the canceled one"

        # Check that the content is what we expect from a fresh generation
        content = ""
        for line in all_lines:
            if line.startswith("data:"):
                data_part = line[len("data: "):].strip()
                if data_part and data_part != "[DONE]":
                    try:
                        chunk_json = json.loads(data_part)
                        if chunk_json.get("type") == "response.output_text.delta":
                            content += chunk_json.get("delta", "")
                    except json.JSONDecodeError:
                        pytest.fail(f"Failed to decode JSON chunk: {data_part}")

        logger.info(f"--- Reconstructed content from second request: ---{content}")
        # Check for expected content in the response
        assert "artificial" in content.lower() or "machine" in content.lower()
        assert len(content) > 50, "Should have generated a reasonable amount of content"
