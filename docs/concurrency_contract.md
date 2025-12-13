# Concurrency Contract

This document defines the operational rules for running MLX-backed work inside MLX Omni Server. The goal is to keep the server responsive under low concurrency, avoid unified-memory contention/OOM, and prevent cross-request correctness bugs.

## Contract

1. **Never block the event loop**

   Any potentially long-running or blocking work (ML inference, model loading, large file I/O, CPU-heavy preprocessing) must not run on the FastAPI event loop thread. Run it via `fastapi.concurrency.run_in_threadpool` (or `anyio.to_thread.run_sync`) from `async` endpoints.

2. **All MLX-backed compute goes through a shared gate**

   All model execution that can pressure MLX/unified memory must acquire the same concurrency gate (initially serialized, e.g. `asyncio.Lock` or `asyncio.Semaphore(1)`). This gate must be shared across chat, embeddings, images, STT, and TTS to avoid unpredictable contention.

3. **Request-scoped filesystem artifacts are unique**

   Any on-disk artifact written during request handling (audio files, image files, intermediate outputs) must be unique per request. Do not use fixed filenames. Prefer `tempfile` (or UUID-based names) and clean up in `finally` blocks.

   If an endpoint returns a `file://` URL, the artifact lifetime must be explicitly defined (temporary vs. cached), and cleanup behavior must match that definition.

4. **Disconnect/cancellation behavior is uniform**

   - For streaming responses: stop streaming immediately when clients disconnect; cancel queued work when feasible; release the MLX gate promptly.
   - For non-streaming responses: cancellation is best-effort, but the server must remain responsive and must not leak resources.

5. **Multi-worker mode is opt-in and explicit**

   `uvicorn --workers N` creates **N processes** with independent caches and independent “global” locks. Unless cross-process coordination and memory budgeting are implemented, the safe default for MLX-bound workloads is `workers=1`.

## Current known violations (as of this codebase)

- Chat/responses already follow the contract (threadpool + shared MLX gate).
- Embeddings/images/STT/TTS currently call synchronous ML work from `async` routes without a shared gate.
- TTS uses a shared output filename (`sample.wav`).
- Images use collision-prone output naming (second-based IDs) and a shared temp directory.
