import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mlx_omni_server.main import app

from .mock_models import MockModel, MockTokenizer

client = TestClient(app)


@pytest.fixture
def mock_load_model():
    with patch("mlx_omni_server.services.chat.models.load") as mock_load:
        mock_load.return_value = (MockModel(), MockTokenizer())
        yield mock_load


def test_chat_completion(mock_load_model):
    """Test the chat completion endpoint with a basic request."""
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "temperature": 0.7,
        "stream": False,
    }

    response = client.post("/chat/completions", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert "id" in data
    assert "created" in data
    assert "model" in data
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]
    assert "content" in data["choices"][0]["message"]
    assert "role" in data["choices"][0]["message"]
    assert data["choices"][0]["message"]["role"] == "assistant"


def test_chat_completion_stream(mock_load_model):
    """Test the chat completion endpoint with streaming enabled."""
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Count from 1 to 5"}],
        "temperature": 0.7,
        "stream": True,
    }

    response = client.post("/chat/completions", json=request_data)
    assert response.status_code == 200

    # Check if we're getting a stream of server-sent events
    assert "text/event-stream" in response.headers["content-type"]

    # Process the streaming response
    for line in response.iter_lines():
        if line:
            # TestClient returns strings, not bytes
            line = line if isinstance(line, str) else line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]  # Remove "data: " prefix
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                assert "id" in chunk
                assert "choices" in chunk
                assert len(chunk["choices"]) > 0
                assert "delta" in chunk["choices"][0]


def test_chat_completion_invalid_request(mock_load_model):
    """Test the chat completion endpoint with invalid request data."""
    # Missing required field 'messages'
    request_data = {"model": "test-model", "temperature": 0.7}

    response = client.post("/chat/completions", json=request_data)
    assert response.status_code == 422  # Validation error


def test_chat_completion_with_system_message(mock_load_model):
    """Test chat completion with a system message."""
    request_data = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What can you do?"},
        ],
        "temperature": 0.7,
        "stream": False,
    }

    response = client.post("/chat/completions", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]
    assert "content" in data["choices"][0]["message"]
    assert "role" in data["choices"][0]["message"]
