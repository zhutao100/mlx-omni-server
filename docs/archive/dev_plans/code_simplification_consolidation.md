# Code Simplification / Consolidation (Completed)

This document captures a completed “code simplification and consolidation” pass. It lives in `docs/archive/` because it is no longer an active plan.

## Status

- Completed (implementation updated; unit tests pass).

## Summary of implemented changes

- **Streaming-safe request/response logging**
  - Response body logging is capped and does not buffer SSE or binary/attachment responses.
  - Request IDs are UUID-based.
  - See `src/mlx_omni_server/middleware/logging.py`.
- **Removed debug `print()` and dead/unreachable code paths**
  - Replaced request-path `print()` calls with logger usage (or removed entirely).
  - Removed unreachable `raise` after `handle_model_error()`.
- **Consolidated duplicated tool tokenizer logic**
  - HF/Llama3 tool-prefill logic is centralized and prefill state is reset per call.
  - See `src/mlx_omni_server/chat/tools/json_prefill.py`.
- **Extracted shared prompt-cache primitives**
  - Token hashing + prefix helpers are centralized.
  - Compatibility wrappers (`tokens_key`) are preserved for callers/tests.
  - See `src/mlx_omni_server/chat/prompt_cache_utils.py`.
- **Unified optional-extra gating**
  - Central `ensure_extra_available()` helper standardizes `501` errors and install hints.
  - STT/TTS/images routes remain registered and validate extra availability at request time.
  - See `src/mlx_omni_server/optional_features.py`.
- **Centralized “extra params” extraction + alias normalization**
  - Shared `extract_extra_params()` helper used by chat/images/tts/embeddings request models.
  - `draft-model` is normalized to `draft_model` in chat requests.
  - See `src/mlx_omni_server/schema_utils.py` and `src/mlx_omni_server/chat/schema.py`.
- **Eliminated busy-polling in response registry waits**
  - `ResponseRegistry.wait_until_finished()` uses condition waiting instead of 50ms polling.
  - See `src/mlx_omni_server/responses/registry.py`.
- **Embeddings cleanup**
  - Removed unused imports/types and bare `except:`.
  - Removed an unused schema enum.
  - See `src/mlx_omni_server/embeddings/`.

## Remaining follow-ups (optional)

- Add explicit request size limits/timeouts and consistent OpenAI-style error envelopes across all endpoints.
- Consider configurable path exclusions/redaction/sampling for request/response logging.
