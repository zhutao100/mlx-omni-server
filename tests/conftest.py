import asyncio
import os
import sys
from unittest.mock import AsyncMock, Mock

# Add project root to path before importing the package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))

import pytest
import pytest_asyncio
from fastapi import Request
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from openai import OpenAI

from mlx_omni_server.chat.generation_service import response_cache
from mlx_omni_server.chat.models.models import model_cache_manager
from mlx_omni_server.main import app


# Global fixtures that can be used across all tests
@pytest.fixture(autouse=True)
def cleanup_caches():
    """Fixture to automatically clean up caches after each test."""
    # Clear caches before each test
    response_cache.clear()
    model_cache_manager.clear()
    yield
    # Clear caches after each test
    response_cache.clear()
    model_cache_manager.clear()


@pytest.fixture
def client():
    """Create a test client for synchronous tests."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def openai_client(client):
    """Create an OpenAI client configured with the test server."""
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


# Utility functions that can be used in tests
def create_mock_request():
    """Create a FastAPI Request mock."""
    request = Mock(spec=Request)
    request.is_disconnected = AsyncMock(return_value=False)
    return request


# Event loop fixture for asyncio tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
