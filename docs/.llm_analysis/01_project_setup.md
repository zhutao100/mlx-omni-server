# Project Setup Analysis (`pyproject.toml`)

This document summarizes the project's configuration based on `pyproject.toml`.

## Core Project Information

-   **Name:** `mlx-omni-server`
-   **Version:** 0.4.9
-   **Description:** An OpenAI-compatible API server using Apple's MLX framework.
-   **Python Version:** >=3.11

## Key Technologies and Dependencies

The project is a `FastAPI` web server designed to serve various AI models running on the `MLX` framework.

### Core Framework

-   **Web Server:** `fastapi`
-   **ASGI Server:** `uvicorn`
-   **Data Validation:** `pydantic`

### AI/ML Capabilities & Libraries

The server is structured around providing several types of AI services, each with its own set of dependencies:

-   **Chat & Vision:**
    -   `mlx-lm`: For running large language models on MLX.
    -   `mlx-vlm`: For running vision-language models on MLX.
    -   `sse-starlette`: Indicates support for streaming chat completions.
-   **Audio Processing:**
    -   `mlx-whisper`: For Speech-to-Text (STT).
    -   `f5-tts-mlx`: For Text-to-Speech (TTS).
    -   `mlx-audio`: Core audio processing capabilities for MLX.
-   **Image Generation:**
    -   `mflux`: A library for running diffusion models on MLX.
-   **Embeddings:**
    -   `mlx-embeddings`: For generating text embeddings.
-   **Model Management:**
    -   `huggingface-hub`: Used for downloading models.

## Entry Point

-   A console script `mlx-omni-server` is defined, which executes the `start` function in `mlx_omni_server.main`.

## Development & Tooling

-   **Testing:** `pytest`, `httpx`
-   **Code Quality:** `pre-commit`, `black`, `isort`
-   **Build System:** `hatchling`

This analysis provides a clear picture of a modular AI server built on a modern Python web stack, designed to leverage Apple's MLX for efficient machine learning inference on Apple Silicon. The project is well-structured, with clear separation of dependencies for different AI functionalities.
