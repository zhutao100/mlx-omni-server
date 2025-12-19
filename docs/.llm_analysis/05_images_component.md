# Images Component Analysis

The `images` component provides an OpenAI DALL-E compatible API for text-to-image generation.
It is shipped as an optional install extra; if `mflux` is not installed, the endpoints remain but return `501 Not Implemented` with an install hint.

## API and Schema

-   **Endpoints:** Exposes `/images/generations` and `/v1/images/generations`.
-   **Schema:** The `ImageGenerationRequest` schema is compatible with the OpenAI API, supporting parameters like `prompt`, `model`, `n`, `size`, and `response_format`. It defaults to a specific FLUX model on Hugging Face.

## Core Logic (`images_service.py`)

The logic is split between a service class and a generator class.

-   **`ImagesService`:**
    -   A shared service instance is created lazily on first use, allowing its generator cache to persist across requests while keeping the server importable without `mflux` installed.
    -   It maintains a cache of `MFluxImageGenerator` instances keyed by model name (enabling cross-request reuse).
    -   It handles file system operations: saving generated images to a temporary directory, encoding them to Base64, and optionally cleaning them up (Base64 mode deletes the temp file; URL mode returns a `file://` path).
    -   URL-mode artifacts use UUID filenames and are periodically cleaned up via a background TTL-based cleanup task.

-   **`MFluxImageGenerator`:**
    -   This class acts as a wrapper around the `mflux` library, specifically the `Flux1` model for text-to-image generation.
    -   It performs lazy loading of the MLX model; the model is only loaded into memory on the first generation request that uses it.
    -   It translates the API request parameters into the configuration objects required by `mflux`.

-   **Concurrency Model:**
    -   Image generation is executed via the shared inference runtime (`run_mlx`), which runs blocking work in a thread pool and serializes MLX-backed compute through a shared gate.
    -   This prevents event-loop blocking and makes contention policy explicit (one MLX-backed job at a time by default).

## Summary

The images component is a capable text-to-image API endpoint that relies on the `mflux` library. It uses a shared images service (cross-request generator caching), runs generation off the event loop under the shared MLX gate, and uses collision-safe filenames plus TTL cleanup for URL-mode artifacts.
