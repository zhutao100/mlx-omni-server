# Embeddings Component Analysis

The `embeddings` component provides an OpenAI-compatible API for generating text embeddings.

## API and Schema

-   **Endpoints:** Exposes `/embeddings` and `/v1/embeddings`.
-   **Schema:** The Pydantic models in `schema.py` (`EmbeddingRequest`, `EmbeddingResponse`) are designed for compatibility with the OpenAI embeddings API. It accepts a string or list of strings as input.

## Core Logic (`embeddings_service.py`)

The `EmbeddingsService` class contains the core implementation.

-   **Model Caching:** The service maintains an in-memory cache of loaded models to avoid reloading them for subsequent requests. Models are loaded on-demand using the `mlx_embeddings.load` function.
-   **Token Counting:** It uses `tiktoken` to provide accurate token counts in the API response's `usage` field.
-   **Generation:**
    -   The service uses the `mlx-embeddings` library to perform the embedding generation.
    -   It contains logic to handle different types of model outputs, with specific handling for BERT-like models (mean pooling or CLS token extraction) and a generic fallback.
-   **Concurrency Model:** Embedding generation is executed via the shared inference runtime (`run_mlx`), which runs blocking work in a thread pool and serializes MLX-backed compute through a shared gate. This keeps the event loop responsive and prevents embeddings from contending unsafely with other MLX workloads.

## Summary

The embeddings component is a straightforward implementation of an embedding API. It uses a library (`mlx-embeddings`) to do the heavy lifting and wraps it in a service that provides model caching and an OpenAI-compatible interface. The synchronous design is a key difference compared to the chat component.
