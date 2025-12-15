# Analysis: tests/conftest.py

## Overview
Root configuration for pytest, providing essential fixtures for both synchronous and asynchronous testing.

## Fixtures
- **cleanup_caches (autouse)**: Ensures isolation between tests by clearing `response_cache` and `model_cache_manager` before and after each test.
- **client**: Provides a synchronous `TestClient` for the FastAPI app.
- **openai_client**: Wraps the `TestClient` in an `OpenAI` SDK client, enabling tests to use the standard OpenAI API interface against the local server.
- **async_client**: Provides an `httpx.AsyncClient` for testing async endpoints and concurrency.
- **event_loop**: Session-scoped event loop management.

## Observations
- The setup mimics a real OpenAI environment, which is excellent for integration testing.
- Automatic cache cleanup is crucial for preventing state leakage between tests.
