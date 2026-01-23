# Stack Assessment

This document summarizes why the current stack fits the project’s target constraints and highlights the main operational risk areas.

## Target constraints (product reality)

- macOS + Apple Silicon, optimized for MLX and unified memory.
- Trusted clients on localhost/LAN (robust to malformed inputs, not hardened for hostile traffic).
- Low concurrency (typically 1–5); prioritize predictable latency over maximum throughput.
- OpenAI-compatible APIs to reuse existing client libraries and tooling.

## Why the current stack fits

- **FastAPI + Uvicorn + Pydantic**: a productive, well-supported foundation for typed REST APIs with streaming.
- **MLX ecosystem** (`mlx-lm`, `mlx-vlm`, `mlx-embeddings`): aligned with Apple Silicon acceleration and local inference.
- **Shared runtime contract** (threadpool offload + global MLX gate): matches unified-memory contention realities and keeps the event loop responsive (see `docs/concurrency_contract.md`).
- **Optional extras for long-tail modalities**:
  - `images` (`mflux`)
  - `stt` (`mlx-whisper` + `python-multipart`)
  - `tts` (`f5-tts-mlx` + `mlx-audio` + `numba`)

This keeps the base install focused on chat/responses/embeddings while allowing heavier modality stacks to be opt-in.

## Main risk areas (operational, not “security”)

- **Overload behavior**: without bounded backpressure around the MLX gate, requests can queue indefinitely (tail-latency spikes). See Phase 1 in `docs/architecture_evaluation.md`.
- **Model lifecycle and memory budgeting**: caches exist, but there is no single place enforcing admission control/eviction across modalities.
- **Multi-worker configuration**: `--workers > 1` creates multiple processes with independent caches and independent gates; this is easy to misconfigure into unified-memory spikes/OOM.
- **Unreliable clients**: robustness also requires explicit payload/time limits and consistent OpenAI-style error shapes (beyond schema validation).

## Practical maintenance policy (recommended)

- Keep modality stacks behind install extras; routes should return `501` with a clear install hint when missing.
- Treat MLX ecosystem upgrades as coordinated changes: bump versions intentionally, run smoke tests per endpoint, and watch for template/tool-call regressions.
- Prefer a small set of “known-good” models for CI/manual verification (documented in `docs/supported_models.md`).
