# Text-to-Speech (TTS) Component Analysis

The `tts` component provides an OpenAI-compatible API for generating speech from text.
It is shipped as an optional install extra; if the TTS backend dependencies are not installed, the endpoints remain but return `501 Not Implemented` with an install hint.

## API and Schema

-   **Endpoints:** Exposes `/audio/speech` and `/v1/audio/speech`.
-   **Schema:** The `TTSRequest` schema is compatible with the OpenAI TTS API, supporting parameters like `model`, `input`, `voice`, and `response_format`.

## Core Logic (`tts_service.py`)

-   **Adapter Pattern:** The service uses a clean adapter pattern (`TTSModelAdapter`) to support multiple underlying TTS libraries with different APIs. It has specific implementations for `f5-tts-mlx` and a more general one for `mlx-audio`. The correct adapter is chosen based on the requested model name.
-   **File Generation:** The service generates the audio and saves it to a file before reading the bytes to send in the response.
-   **Instantiation:** The router creates a new `TTSService` per request (there is no shared, long-lived service instance).

## Concurrency and File Handling Model

The component now follows the server’s shared concurrency contract:

-   **Threadpool Offload:** Audio generation and file I/O are executed off the event loop.
-   **Shared MLX Gate:** Generation is executed via the shared inference runtime (`run_mlx`), serializing MLX-backed compute across endpoints.
-   **Request-Scoped Artifacts:** Each request generates into its own temporary directory, eliminating filename collisions. (The `f5-tts-mlx` backend is currently constrained to WAV output.)

## Summary

The TTS component features a good software design pattern (adapter) for handling different backends. It now executes generation off the event loop under a shared MLX gate and uses request-scoped output paths, making it safe under concurrent load.
