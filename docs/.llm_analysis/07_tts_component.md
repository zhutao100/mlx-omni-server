# Text-to-Speech (TTS) Component Analysis

The `tts` component provides an OpenAI-compatible API for generating speech from text.

## API and Schema

-   **Endpoints:** Exposes `/audio/speech` and `/v1/audio/speech`.
-   **Schema:** The `TTSRequest` schema is compatible with the OpenAI TTS API, supporting parameters like `model`, `input`, `voice`, and `response_format`.

## Core Logic (`tts_service.py`)

-   **Adapter Pattern:** The service uses a clean adapter pattern (`TTSModelAdapter`) to support multiple underlying TTS libraries with different APIs. It has specific implementations for `f5-tts-mlx` and a more general one for `mlx-audio`. The correct adapter is chosen based on the requested model name.
-   **File Generation:** The service generates the audio and saves it to a file before reading the bytes to send in the response.
-   **Instantiation:** The router creates a new `TTSService` per request (there is no shared, long-lived service instance).

## Concurrency and File Handling Model

The concurrency model of this component has several significant flaws:

-   **Event Loop Blocking:** The core audio generation is a synchronous, blocking function. It is called directly from an `async` method, which will **block the server's event loop** and prevent it from handling other requests until the audio generation is complete.
-   **No Locking:** There is no lock to serialize access to the MLX backend.
-   **Race Conditions:** The service uses a **hardcoded output filename** (`sample.wav`, relative to the working directory) for all generations. If two requests are processed concurrently, they will attempt to write to, read from, and delete the same file, leading to incorrect output and errors. This makes the component unsafe for concurrent use.

## Summary

The TTS component features a good software design pattern (adapter) for handling different backends. However, its implementation of concurrency and file handling is flawed. The combination of event loop blocking and race conditions on the output file makes it unreliable under concurrent load.
