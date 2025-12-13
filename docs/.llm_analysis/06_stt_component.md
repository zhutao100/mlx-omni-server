# Speech-to-Text (STT) Component Analysis

The `stt` component provides an OpenAI-compatible API for audio transcription, based on the Whisper model.

## API and Schema

-   **Endpoints:** Exposes `/audio/transcriptions` and `/v1/audio/transcriptions`.
-   **Schema:** The API uses a `multipart/form-data` request format, defined in `STTRequestForm`, to handle file uploads. This matches the OpenAI Whisper API. It supports various response formats (`json`, `text`, `srt`, etc.) and timestamp granularities.

## Core Logic (`whisper_model.py`)

The implementation is split across `STTService` and `WhisperModel` classes.

-   **`STTService`:** This service is instantiated on every request. It manages the process of saving the uploaded audio file to a temporary location, calling the model for transcription, and cleaning up the file.
-   **`WhisperModel`:** This class is a direct wrapper around the `mlx_whisper.transcribe` function. It handles:
    -   Passing API parameters to the `mlx-whisper` library.
    -   Formatting the raw output from the library into the various response formats required by the API (e.g., generating SRT/VTT files or detailed JSON).
-   **Model Management:** Model loading is handled implicitly by the `mlx_whisper.transcribe` function. There is no explicit model caching layer within the service, so performance relies on the caching behavior of the underlying `mlx-whisper` library.

## Concurrency Model

-   **Threadpool Offload:** Upload persistence, transcription, and response formatting are executed off the event loop, keeping the server responsive during long transcriptions.
-   **Shared MLX Gate:** The blocking MLX-backed transcription call is executed via the shared inference runtime (`run_mlx`), which serializes MLX-backed compute across endpoints to reduce unified-memory contention.

## Summary

The STT component provides a functional Whisper API endpoint with an async-safe execution model: blocking work runs in a thread pool and the MLX-backed transcription is gated through the shared MLX gate.
