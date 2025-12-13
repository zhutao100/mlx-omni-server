# Main Application and Routing Analysis

This document analyzes the server's entry point (`main.py`) and the main API routing structure (`routers.py`).

## Application Entry Point (`main.py`)

The server is a standard `FastAPI` application launched with `uvicorn`.

-   **Configuration:** A `start()` function, exposed as a command-line script `mlx-omni-server`, handles server configuration. It uses `argparse` to set the `host`, `port`, `workers`, and `log-level`.
-   **Workers:** The CLI exposes `--workers` and passes it to `uvicorn`. This creates multiple *processes* (not threads), so each worker has its own in-memory caches and its own “global” locks. For MLX-bound workloads, `workers=1` is typically the safest default unless you explicitly design for multi-process coordination and memory budgeting.
-   **Lifecycle Management:** The application uses a `lifespan` manager to run background tasks. It starts `background_cache_cleanup()` from the chat subsystem and a URL-mode image artifact cleanup task (TTL-based cleanup).
-   **Middleware:** Custom logging middleware (`RequestResponseLoggingMiddleware`) is registered, indicating that all requests and responses are logged.
-   **Root Router:** The application's API is consolidated into a single `APIRouter` instance imported from the `routers` module.

## API Routing Structure (`routers.py`)

The `routers.py` file acts as the central hub for defining the application's API structure. It aggregates all the individual routers from the different functional modules.

-   **Modular Design:** The API is highly modular. Each core feature of the server is encapsulated in its own subdirectory and has its own `APIRouter`.
-   **Combined Endpoints:** The main `api_router` includes the following routers, effectively creating a single, unified API surface:
    -   `stt_router`: Endpoints for Speech-to-Text.
    -   `tts_router`: Endpoints for Text-to-Speech.
    -   `models_router`: Endpoints for listing and describing available models.
    -   `images.router`: Endpoints for image generation.
    -   `chat_router`: Endpoints for chat completions.
    -   `embeddings_router`: Endpoints for generating embeddings.
    -   `responses_router`: OpenAI Responses API endpoint implemented as an adapter over the chat service.

## Architecture Summary

The top-level architecture is a classic `FastAPI` application with a modular, feature-based routing system. The application is designed to be run as a standalone server process and is configured via command-line arguments. Request logging and background cache cleanup indicate a production-oriented baseline. The `--workers` option is important operationally: multiple workers mean multiple processes with duplicated caches and independent locks, which can increase MLX memory pressure.
