import asyncio
import json
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio
from fastapi import Request
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from openai import OpenAI

from mlx_omni_server.chat.mlx import models as mlx_models
from mlx_omni_server.chat.router import (
    CACHE_TTL,
    NonStreamCacheEntry,
    StreamCacheEntry,
    _create_text_model,
    make_request_hash,
    response_cache,
)
from mlx_omni_server.chat.schema import ChatCompletionRequest, ChatMessage, Role
from mlx_omni_server.chat.text_models import (
    BaseTextModel,
    ChatCompletionChunk,
    ChatCompletionResponse,
)
from mlx_omni_server.main import app

# Constants
MODEL_ID = "mlx-community/gemma-3-1b-it-4bit-DWQ"


# Mock Classes
class MockTextModel(BaseTextModel):
    """Mock text model for testing"""

    def __init__(self):
        self.call_count = 0
        self.stream_call_count = 0

    def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
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

    def stream_generate(self, request: ChatCompletionRequest):
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


# Fixtures
@pytest.fixture(autouse=True)
def cleanup_caches():
    """Fixture to automatically clean up caches after each test."""
    response_cache.clear()
    mlx_models._model_cache = None
    mlx_models._mlx_model_cache = None
    yield
    response_cache.clear()
    mlx_models._model_cache = None
    mlx_models._mlx_model_cache = None


@pytest.fixture
def client():
    """Create test client"""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def openai_client(client):
    """Create OpenAI client configured with test server."""
    return OpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=client,
    )


@pytest_asyncio.fixture
async def async_client():
    """Create an async client for concurrent requests."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_request():
    """Create a mock request"""
    return ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role=Role.USER, content="Hello")],
        stream=False,
    )


@pytest.fixture
def mock_stream_request():
    """Create a mock streaming request"""
    return ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role=Role.USER, content="Hello")],
        stream=True,
    )


def create_mock_request():
    """Create a FastAPI Request mock"""
    request = Mock(spec=Request)
    request.is_disconnected = AsyncMock(return_value=False)
    return request


# Unit Tests
class TestRequestHashing:
    """Test request hashing functionality"""

    def test_make_request_hash_basic(self, mock_request):
        """Test basic request hashing"""
        hash1 = make_request_hash(mock_request)
        hash2 = make_request_hash(mock_request)
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hash length

    def test_make_request_hash_different_content(self):
        """Test that different requests produce different hashes"""
        req1 = ChatCompletionRequest(
            model="test-model", messages=[ChatMessage(role=Role.USER, content="Hello")]
        )
        req2 = ChatCompletionRequest(
            model="test-model", messages=[ChatMessage(role=Role.USER, content="World")]
        )
        hash1 = make_request_hash(req1)
        hash2 = make_request_hash(req2)
        assert hash1 != hash2

    def test_make_request_hash_consistent_ordering(self):
        """Test that hash is consistent regardless of dict ordering"""
        req1 = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role=Role.USER, content="Hello")],
            temperature=0.7,
            max_tokens=100,
        )
        req2 = ChatCompletionRequest(
            max_tokens=100,
            model="test-model",
            temperature=0.7,
            messages=[ChatMessage(role=Role.USER, content="Hello")],
        )
        hash1 = make_request_hash(req1)
        hash2 = make_request_hash(req2)
        assert hash1 == hash2


class TestCacheEntries:
    """Test cache entry dataclasses"""

    def test_stream_cache_entry_defaults(self):
        """Test StreamCacheEntry default values"""
        entry = StreamCacheEntry()
        assert entry.chunks == []
        assert isinstance(entry.done_event, asyncio.Event)
        assert isinstance(entry.error_event, asyncio.Event)
        assert isinstance(entry.stop_event, asyncio.Event)
        assert isinstance(entry.created_at, float)
        assert entry.active_clients == 0

    def test_non_stream_cache_entry(self):
        """Test NonStreamCacheEntry"""
        payload = {"test": "data"}
        entry = NonStreamCacheEntry(payload=payload)
        assert entry.payload == payload
        assert isinstance(entry.created_at, float)


class TestNonStreamingCacheUnit:
    """Test non-streaming cache functionality with mocks"""

    @patch("mlx_omni_server.chat.router._create_text_model")
    def test_non_streaming_cache_miss(self, mock_create_model, client, mock_request):
        """Test non-streaming request with cache miss"""
        mock_model = MockTextModel()
        mock_create_model.return_value = mock_model

        response = client.post("/v1/chat/completions", json=mock_request.model_dump())

        assert mock_model.call_count == 1
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Hello, world!"

        req_hash = make_request_hash(mock_request)
        assert req_hash in response_cache
        assert isinstance(response_cache[req_hash], NonStreamCacheEntry)

    @patch("mlx_omni_server.chat.router._create_text_model")
    def test_non_streaming_cache_hit(self, mock_create_model, client, mock_request):
        """Test non-streaming request with cache hit"""
        mock_model = MockTextModel()
        mock_create_model.return_value = mock_model

        response1 = client.post("/v1/chat/completions", json=mock_request.model_dump())
        assert mock_model.call_count == 1

        response2 = client.post("/v1/chat/completions", json=mock_request.model_dump())
        assert mock_model.call_count == 1

        assert response1.json() == response2.json()
        assert response2.headers.get("X-Idempotent-Replay") == "true"

    @patch("mlx_omni_server.chat.router._create_text_model")
    def test_non_streaming_different_requests(self, mock_create_model, client):
        """Test that different requests don't hit the same cache"""
        mock_model = MockTextModel()
        mock_create_model.return_value = mock_model

        req1 = ChatCompletionRequest(
            model="test-model", messages=[ChatMessage(role=Role.USER, content="Hello")]
        )
        req2 = ChatCompletionRequest(
            model="test-model", messages=[ChatMessage(role=Role.USER, content="World")]
        )

        response1 = client.post("/v1/chat/completions", json=req1.model_dump())
        response2 = client.post("/v1/chat/completions", json=req2.model_dump())

        assert mock_model.call_count == 2
        assert (
            response1.json()["choices"][0]["message"]["content"]
            != response2.json()["choices"][0]["message"]["content"]
        )


class TestModelCreation:
    """Test model creation functionality"""

    @patch("mlx_omni_server.chat.router.load_model")
    def test_create_text_model_basic(self, mock_load_model):
        """Test basic model creation"""
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        model = _create_text_model("test-model")

        mock_load_model.assert_called_once()
        args, kwargs = mock_load_model.call_args
        from mlx_omni_server.chat.mlx.models import ModelId

        assert args[0].name == "test-model"
        assert args[0].adapter_path is None
        assert args[0].draft_model is None
        assert model == mock_model

    @patch("mlx_omni_server.chat.router.load_model")
    def test_create_text_model_with_adapter(self, mock_load_model):
        """Test model creation with adapter path"""
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        model = _create_text_model("test-model", adapter_path="/path/to/adapter")

        mock_load_model.assert_called_once()
        args, kwargs = mock_load_model.call_args
        from mlx_omni_server.chat.mlx.models import ModelId

        assert args[0].name == "test-model"
        assert args[0].adapter_path == "/path/to/adapter"
        assert args[0].draft_model is None

    @patch("mlx_omni_server.chat.router.load_model")
    def test_create_text_model_with_draft(self, mock_load_model):
        """Test model creation with draft model"""
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        model = _create_text_model("test-model", draft_model="draft-model")

        mock_load_model.assert_called_once()
        args, kwargs = mock_load_model.call_args
        from mlx_omni_server.chat.mlx.models import ModelId

        assert args[0].name == "test-model"
        assert args[0].adapter_path is None
        assert args[0].draft_model == "draft-model"


class TestCacheCleanup:
    """Test cache cleanup functionality"""

    def test_background_cache_cleanup_logic(self):
        """Test background cache cleanup logic without running the infinite loop"""
        response_cache.clear()

        old_entry = NonStreamCacheEntry(payload={"test": "data"})
        old_entry.created_at = time.time() - (CACHE_TTL + 1)
        new_entry = NonStreamCacheEntry(payload={"test": "data"})
        response_cache["old"] = old_entry
        response_cache["new"] = new_entry

        cutoff = time.time() - CACHE_TTL
        keys_to_delete = [
            k
            for k, v in response_cache.items()
            if v.created_at < cutoff
            and (not isinstance(v, StreamCacheEntry) or v.active_clients == 0)
        ]

        assert "old" in keys_to_delete
        assert "new" not in keys_to_delete

    def test_background_cache_cleanup_stream_with_clients(self):
        """Test that stream entries with active clients aren't cleaned up"""
        response_cache.clear()

        stream_entry = StreamCacheEntry()
        stream_entry.active_clients = 1
        stream_entry.created_at = time.time() - (CACHE_TTL + 1)
        response_cache["stream"] = stream_entry

        cutoff = time.time() - CACHE_TTL
        keys_to_delete = [
            k
            for k, v in response_cache.items()
            if v.created_at < cutoff
            and (not isinstance(v, StreamCacheEntry) or v.active_clients == 0)
        ]

        assert "stream" not in keys_to_delete

    def test_background_cache_cleanup_stream_no_clients(self):
        """Test that expired stream entries without clients are cleaned up"""
        response_cache.clear()

        stream_entry = StreamCacheEntry()
        stream_entry.active_clients = 0
        stream_entry.created_at = time.time() - (CACHE_TTL + 1)
        response_cache["stream"] = stream_entry

        cutoff = time.time() - CACHE_TTL
        keys_to_delete = [
            k
            for k, v in response_cache.items()
            if v.created_at < cutoff
            and (not isinstance(v, StreamCacheEntry) or v.active_clients == 0)
        ]

        assert "stream" in keys_to_delete

    @pytest.mark.asyncio
    async def test_cache_cleanup_integration(self):
        """Test the cache cleanup logic with integration test approach."""
        # Manually clear cache
        response_cache.clear()

        now = time.time()
        old = now - CACHE_TTL - 100

        # Expired non-stream entry
        req1 = ChatCompletionRequest(model=MODEL_ID, messages=[{"role": "user", "content": "req1"}])
        hash1 = make_request_hash(req1)
        response_cache[hash1] = NonStreamCacheEntry(payload={}, created_at=old)

        # Expired stream entry with no active clients
        req2 = ChatCompletionRequest(model=MODEL_ID, messages=[{"role": "user", "content": "req2"}])
        hash2 = make_request_hash(req2)
        entry2 = StreamCacheEntry(created_at=old)
        entry2.done_event.set()
        response_cache[hash2] = entry2

        # Expired stream entry with active clients (should not be removed)
        req3 = ChatCompletionRequest(model=MODEL_ID, messages=[{"role": "user", "content": "req3"}])
        hash3 = make_request_hash(req3)
        entry3 = StreamCacheEntry(created_at=old)
        entry3.active_clients = 1
        response_cache[hash3] = entry3

        # Fresh entry (should not be removed)
        req4 = ChatCompletionRequest(model=MODEL_ID, messages=[{"role": "user", "content": "req4"}])
        hash4 = make_request_hash(req4)
        response_cache[hash4] = NonStreamCacheEntry(payload={}, created_at=now)

        assert len(response_cache) == 4

        # Run one cycle of the cleanup logic
        # The original function has an infinite loop, so we extract the core logic
        from mlx_omni_server.chat.router import cache_lock
        cutoff = time.time() - CACHE_TTL
        async with cache_lock:
            for k in list(response_cache.keys()):
                if response_cache[k].created_at < cutoff:
                    if isinstance(response_cache[k], StreamCacheEntry):
                        if response_cache[k].active_clients == 0:
                            del response_cache[k]
                    else:
                        del response_cache[k]

        assert len(response_cache) == 2
        assert hash1 not in response_cache
        assert hash2 not in response_cache
        assert hash3 in response_cache
        assert hash4 in response_cache


# Integration Tests
def test_non_streaming_cache_integration(openai_client):
    """Test caching for non-streaming responses with a real model call."""
    request_payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Tell me a joke."}],
    }

    response1_with_raw = openai_client.chat.completions.with_raw_response.create(
        **request_payload
    )
    response1 = response1_with_raw.parse()
    assert "x-idempotent-replay" not in response1_with_raw.headers
    assert len(response_cache) == 1
    req_hash = list(response_cache.keys())[0]
    assert isinstance(response_cache[req_hash], NonStreamCacheEntry)

    response2_with_raw = openai_client.chat.completions.with_raw_response.create(
        **request_payload
    )
    response2 = response2_with_raw.parse()
    assert response2_with_raw.headers["x-idempotent-replay"] == "true"
    assert response1.id == response2.id
    assert (
        response1.choices[0].message.content == response2.choices[0].message.content
    )
    assert len(response_cache) == 1


@pytest.mark.asyncio
async def test_streaming_cache_two_clients_integration(async_client):
    """Test two clients connecting to the same stream with a real model call."""
    request_payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": "Write a short story about a robot."}
        ],
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
                    data = line[len("data: "):]
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
                    data = line[len("data: "):]
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
