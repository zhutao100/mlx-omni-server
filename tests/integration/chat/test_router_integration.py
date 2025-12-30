import asyncio
import concurrent.futures
import json
import time
from unittest.mock import Mock, patch

import pytest

from mlx_omni_server.chat.generation_service import (
    NonStreamCacheEntry,
    StreamCacheEntry,
    response_cache,
)
from mlx_omni_server.chat.schema import ChatCompletionRequest
from mlx_omni_server.chat.text_models import (
    BaseTextModel,
    ChatCompletionChunk,
    ChatCompletionResponse,
)

# Constants
MODEL_ID = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"


# Mock Classes
class MockTextModel(BaseTextModel):
    """Mock text model for testing"""

    def __init__(self):
        self.call_count = 0
        self.stream_call_count = 0

    def generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel=None,
    ) -> ChatCompletionResponse:
        """Mock generate method"""
        self.call_count += 1
        content = "Hello, world!"
        if request.messages and len(request.messages) > 0:
            if request.messages[0].content == "World":
                content = "Hello, Universe!"

        return ChatCompletionResponse(
            id="test-id",
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
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )

    def stream_generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel=None,
    ):
        """Mock stream_generate method"""
        self.stream_call_count += 1
        chunk = ChatCompletionChunk(
            id="test-id",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello"},
                    "finish_reason": None,
                }
            ],
        )
        yield chunk

        chunk = ChatCompletionChunk(
            id="test-id",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ", world!"},
                    "finish_reason": "stop",
                }
            ],
        )
        yield chunk


# Integration Tests
def test_non_streaming_cache_integration(openai_client):
    """Test caching for non-streaming responses with a real model call."""
    request_payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Tell me a joke."}],
    }

    response1_with_raw = openai_client.chat.completions.with_raw_response.create(**request_payload)
    response1 = response1_with_raw.parse()
    assert "x-idempotent-replay" not in response1_with_raw.headers
    assert len(response_cache) == 1
    req_hash = list(response_cache.keys())[0]
    assert isinstance(response_cache[req_hash], NonStreamCacheEntry)

    response2_with_raw = openai_client.chat.completions.with_raw_response.create(**request_payload)
    response2 = response2_with_raw.parse()
    assert response2_with_raw.headers["x-idempotent-replay"] == "true"
    assert response1.id == response2.id
    assert response1.choices[0].message.content == response2.choices[0].message.content
    assert len(response_cache) == 1


@pytest.mark.asyncio
async def test_streaming_cache_two_clients_integration(async_client):
    """Test two clients connecting to the same stream with a real model call."""
    request_payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Write a short story about a robot."}],
        "stream": True,
    }
    json_payload = json.dumps(request_payload)

    async def stream_request():
        chunks = []
        async with async_client.stream(
            "POST",
            "/v1/chat/completions",
            content=json_payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data = line[len("data: ") :]
                    if data.strip() == "[DONE]":
                        break
                    chunks.append(json.loads(data))
        return chunks

    results = await asyncio.gather(stream_request(), stream_request())

    assert len(results[0]) > 1
    assert results[0] == results[1]

    assert len(response_cache) == 1
    req_hash = list(response_cache.keys())[0]
    cache_entry = response_cache[req_hash]
    assert isinstance(cache_entry, StreamCacheEntry)
    await asyncio.sleep(0.1)
    assert cache_entry.active_clients == 0


@pytest.mark.asyncio
async def test_streaming_cache_late_client_integration(async_client):
    """Test a client connecting to a completed stream with a real model call."""
    request_payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": True,
    }
    json_payload = json.dumps(request_payload)

    async def stream_request():
        chunks = []
        headers = {}
        async with async_client.stream(
            "POST",
            "/v1/chat/completions",
            content=json_payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            assert response.status_code == 200
            headers = response.headers
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data = line[len("data: ") :]
                    if data.strip() == "[DONE]":
                        break
                    chunks.append(json.loads(data))
        return chunks, headers

    result1, headers1 = await stream_request()
    assert len(result1) > 0
    assert "x-idempotent-replay" in headers1
    assert headers1["x-idempotent-replay"] == "live"

    result2, headers2 = await stream_request()
    assert len(result2) > 0
    assert "x-idempotent-replay" in headers2
    assert headers2["x-idempotent-replay"] == "true"

    assert result1 == result2


@pytest.mark.asyncio
async def test_streaming_emits_final_chunk_before_done(async_client):
    """Ensure final chunk notifies before the [DONE] sentinel."""
    scheduled = []

    def fake_run_coroutine_threadsafe(coro, loop):
        fut = concurrent.futures.Future()
        scheduled.append((coro, fut))
        return fut

    with patch(
        "mlx_omni_server.chat.generation_service.asyncio.run_coroutine_threadsafe",
        side_effect=fake_run_coroutine_threadsafe,
    ):
        with patch(
            "mlx_omni_server.chat.generation_service._create_text_model"
        ) as mock_create_model:
            mock_model = MockTextModel()
            mock_create_model.return_value = mock_model

            request_payload = {
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "stream please"}],
                "stream": True,
            }
            json_payload = json.dumps(request_payload)

            async def stream_request():
                chunks = []
                async with async_client.stream(
                    "POST",
                    "/v1/chat/completions",
                    content=json_payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    assert response.status_code == 200
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data: ") :]
                        if data.strip() == "[DONE]":
                            break
                        chunks.append(json.loads(data))
                return chunks

            async def wait_for(predicate, timeout: float = 1.0):
                loop = asyncio.get_running_loop()
                deadline = loop.time() + timeout
                while not predicate():
                    if loop.time() >= deadline:
                        raise AssertionError("Timed out waiting for condition")
                    await asyncio.sleep(0)

            async def run_next(expected_name: str):
                coro, fut = scheduled.pop(0)
                name = getattr(getattr(coro, "cr_code", None), "co_name", "")
                assert name == expected_name
                try:
                    await coro
                finally:
                    fut.set_result(None)
                await asyncio.sleep(0)

            stream_task = asyncio.create_task(stream_request())

            await wait_for(lambda: len(scheduled) >= 1)
            await run_next("notify")

            await wait_for(lambda: len(scheduled) >= 1)
            assert not any(
                getattr(getattr(coro, "cr_code", None), "co_name", "") == "notify_done"
                for coro, _ in scheduled
            ), "notify_done scheduled before final chunk"

            await run_next("notify")

            await wait_for(lambda: len(scheduled) >= 1)
            await run_next("notify_done")

            chunks = await stream_task
            assert chunks, "Expected at least one streamed chunk"
            assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_streaming_cache_with_error_integration(async_client):
    """Test error handling in streaming responses."""
    # This test would require mocking the model to raise an exception
    # For now, we'll just verify the structure of the test
    pass
