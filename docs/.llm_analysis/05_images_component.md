# Images Component Analysis

The `images` component provides an OpenAI DALL-E compatible API for text-to-image generation.

## API and Schema

-   **Endpoints:** Exposes `/images/generations` and `/v1/images/generations`.
-   **Schema:** The `ImageGenerationRequest` schema is compatible with the OpenAI API, supporting parameters like `prompt`, `model`, `n`, `size`, and `response_format`. It defaults to a specific FLUX model on Hugging Face.

## Core Logic (`images_service.py`)

The logic is split between a service class and a generator class.

-   **`ImagesService`:**
    -   This is a lightweight service class that is instantiated on each request.
    -   It defines a cache of `MFluxImageGenerator` instances keyed by model name, but because `ImagesService` is instantiated per request, this cache does not persist across requests (it only helps within a single request that generates multiple images).
    -   It handles file system operations: saving generated images to a temporary directory, encoding them to Base64, and optionally cleaning them up (Base64 mode deletes the temp file; URL mode returns a `file://` path and leaves the file on disk).
    -   Output filenames are derived from second-level timestamps, which can collide across concurrent requests.

-   **`MFluxImageGenerator`:**
    -   This class acts as a wrapper around the `mflux` library, specifically the `Flux1` model for text-to-image generation.
    -   It performs lazy loading of the MLX model; the model is only loaded into memory on the first generation request that uses it.
    -   It translates the API request parameters into the configuration objects required by `mflux`.

-   **Concurrency Model:**
    -   The FastAPI route is `async`, but image generation is implemented synchronously and is called directly from the route (no thread pool), so long generations will block the server's event loop.
    -   There is **no locking/gating mechanism** equivalent to the `mlx_lock` used by chat. Multiple concurrent image generations may contend for unified memory and trigger GPU OOM.
    -   The shared temp output directory and collision-prone filenames introduce additional request-safety hazards under concurrency.

## Summary

The images component is a capable text-to-image API endpoint that relies on the `mflux` library. However, its current implementation blocks the event loop for generation, lacks an MLX concurrency gate, and has collision-prone on-disk artifact naming. It also does not currently achieve cross-request model/generator reuse because `ImagesService` is instantiated per request.
