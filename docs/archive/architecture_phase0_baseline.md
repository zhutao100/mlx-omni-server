# Phase 0 Baseline (Implemented)

This document captures the **baseline checklist** that is already implemented in the server, and is archived to keep `docs/architecture_evaluation.md` focused on future iterations.

For the current behavioral contract, read `docs/concurrency_contract.md`.

## Baseline checklist

- **No event-loop blocking**: all blocking ML/backends run via threadpool helpers from async endpoints.
- **Shared MLX gate**: MLX-backed compute is serialized through the shared runtime helpers.
- **Request-scoped artifacts**: any on-disk artifacts are unique per request (no shared filenames) and cleaned up with TTL where applicable.

## Code pointers (spot-check)

- Shared runtime helpers (MLX gate + threadpool): `src/mlx_omni_server/inference/runtime.py`
- Embeddings runtime integration: `src/mlx_omni_server/embeddings/router.py`
- Images runtime integration and artifact lifecycle: `src/mlx_omni_server/images/images.py`, `src/mlx_omni_server/images/images_service.py`
- STT runtime integration: `src/mlx_omni_server/stt/whisper_model.py`
- TTS request-scoped outputs: `src/mlx_omni_server/tts/tts_service.py`
- Startup background tasks (chat cache + image cleanup): `src/mlx_omni_server/main.py`
