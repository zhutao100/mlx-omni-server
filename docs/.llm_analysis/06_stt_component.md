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

-   **Event Loop Blocking:** The core transcription logic (`WhisperModel.generate`) is a synchronous, blocking function. It is called directly from an `async` method in the `STTService` **without** being run in a thread pool. This is a significant performance flaw that will cause the server's main event loop to freeze during transcription.
-   **No Locking:** Similar to the `images` component, there is no lock to serialize access to the MLX backend. Concurrent requests will run in parallel and will also block the event loop in parallel, likely leading to very poor performance and potential GPU memory issues.

## Summary

The STT component provides a functional Whisper API endpoint. However, its concurrency model is its biggest weakness. The direct call to a blocking function from an async context will severely limit server throughput and responsiveness.
