# Simplification / Consolidation Report (ranked)

## 1) SIMP-001 — Logging middleware buffers streaming/binary responses (can break SSE + StreamingResponse)
- **ID**: `SIMP-001`
- **Location**: `src/mlx_omni_server/middleware/logging.py:45`, `src/mlx_omni_server/middleware/logging.py:79`, `src/mlx_omni_server/tts/tts.py:49`, `src/mlx_omni_server/responses/router.py:92`
- **Type**: `structure`
- **Description**: `RequestResponseLoggingMiddleware` decides “streaming” only by parsing JSON request bodies and checking `\"stream\"`. Everything else falls through to the “non-streaming” path which **consumes `response.body_iterator` into memory** and then re-wraps it. This can:
  - defeat/serialize streaming for endpoints that stream without a JSON `\"stream\"` field (e.g. TTS audio),
  - hang or massively buffer SSE streams triggered by query params (e.g. `GET /responses/{id}?stream=true`),
  - add large memory spikes for big responses.
- **Impact**: correctness risk (streaming behavior), latency, memory pressure, harder debugging (middleware side-effects).
- **Suggested change**:
  - Treat streaming as a **property of the response**, not just request JSON:
    - if `isinstance(response, StreamingResponse)` (or content-type starts with `text/event-stream` / `audio/` / `application/octet-stream`), **don’t iterate the body**; just log status/headers/timing.
  - Put a size cap on body logging (e.g. first N bytes, only for JSON/text).
  - Use a real request id (e.g. `uuid4().hex`) instead of `str(time.time())`.
- **Risk/effort**: **medium risk, medium effort** (logging output changes), but very high leverage and likely fixes real streaming issues.

## 2) SIMP-002 — Debug `print()` calls and dead/unreachable code in request paths
- **ID**: `SIMP-002`
- **Location**: `src/mlx_omni_server/stt/schema.py:127`, `src/mlx_omni_server/stt/whisper_model.py:38`, `src/mlx_omni_server/chat/models/router.py:21`, `src/mlx_omni_server/chat/models/router.py:54`, `src/mlx_omni_server/chat/tools/mistral.py:90`
- **Type**: `non-idiomatic`
- **Description**:
  - Multiple `print(...)` statements run during normal requests (STT form validation, whisper generation, model router error handling, tool parsing). They bypass the configured logging system and can leak user content or flood stdout.
  - `delete_model()` calls `handle_model_error(e)` (which always raises) and then does `raise` anyway (`src/mlx_omni_server/chat/models/router.py:54`) — unreachable.
- **Impact**: noisy logs, inconsistent observability, potential leakage, small maintainability debt.
- **Suggested change**:
  - Replace `print()` with `logger.debug/info/warning/exception` (or remove entirely if not needed).
  - Remove unreachable `raise` after `handle_model_error(e)`.
- **Risk/effort**: **low risk, small change**.

## 3) SIMP-003 — Stateful `pre_fill_tools_prompt` + duplicated tokenizer code (bug risk + duplication)
- **ID**: `SIMP-003`
- **Location**: `src/mlx_omni_server/chat/tools/hugging_face.py:33`, `src/mlx_omni_server/chat/tools/hugging_face.py:98`, `src/mlx_omni_server/chat/tools/llama3.py:35`, `src/mlx_omni_server/chat/tools/chat_tokenizer.py:145`
- **Type**: `duplication`
- **Description**:
  - `HuggingFaceChatTokenizer` and `Llama3ChatTokenizer` are near-copies (encode/decode/strict parsing) differing mainly by `tool_call_start_token`.
  - Both **append** to `self.pre_fill_tools_prompt` during `encode()` and never reset it. If the tokenizer instance is reused across requests (likely), this can accumulate stale prefixes and corrupt prompts/decoding.
  - Logger imports are inconsistent (`llama3.py` imports `mlx_omni_server.utils.logger`, others use relative imports).
- **Impact**: subtle correctness bugs over time, harder to reason about per-request behavior, extra code to maintain.
- **Suggested change**:
  - Make prefill **per-call**, not mutable instance state (at minimum: reset `self.pre_fill_tools_prompt = \"\"` at the start of `encode()`).
  - Consolidate both classes into one parameterized implementation (e.g. `UnifiedJsonToolChatTokenizer(tool_call_start_token=...)`).
  - Normalize imports (`from ...utils.logger import logger`) everywhere.
- **Risk/effort**: **medium risk, medium effort** (tool-call formatting is sensitive), but high value.

## 4) SIMP-004 — Duplicate prompt-cache primitives across LM/VLM (drift risk)
- **ID**: `SIMP-004`
- **Location**: `src/mlx_omni_server/chat/mlx_lm/prompt_cache.py:20`, `src/mlx_omni_server/chat/mlx_lm/prompt_cache.py:28`, `src/mlx_omni_server/chat/mlx_vlm/prompt_cache.py:17`, `src/mlx_omni_server/chat/mlx_vlm/prompt_cache.py:25`
- **Type**: `duplication`
- **Description**: Both prompt-cache implementations re-define `common_prefix_len()` and very similar token hashing (`struct.pack` + `sha256`), with only small variation for VLM media hashes.
- **Impact**: future bug risk from near-identical logic diverging, harder refactors.
- **Suggested change**:
  - Extract shared helpers into a small module (e.g. `chat/prompt_cache_utils.py`):
    - `common_prefix_len(a: list[int], b: list[int]) -> int`
    - `hash_tokens(tokens: list[int]) -> str`
    - `hash_tokens_with_media(tokens: list[int], media_hashes: list[str] | None) -> str`
- **Risk/effort**: **low risk, small-to-medium change**.

## 5) SIMP-005 — Optional extras gating patterns vary (images/stt/tts/main) and repeat boilerplate
- **ID**: `SIMP-005`
- **Location**: `src/mlx_omni_server/optional_features.py:58`, `src/mlx_omni_server/stt/stt.py:9`, `src/mlx_omni_server/images/images.py:30`, `src/mlx_omni_server/tts/tts.py:20`, `src/mlx_omni_server/main.py:24`
- **Type**: `structure`
- **Description**:
  - STT computes `_MISSING_DEPS` at import time and conditionally defines handlers.
  - Images/TTS do per-request checks.
  - `main.py` checks `is_available(\"images\")` and then uses a broad `except Exception: pass` around imports/background tasks.
- **Impact**: inconsistent conventions, duplicated `501` logic, edge-case confusion (caching missing deps, swallowed startup errors).
- **Suggested change**:
  - Introduce a single helper (or decorator) to enforce extras consistently, e.g. `ensure_extra_available(\"tts\") -> None` that raises `HTTPException(501, detail=...)`.
  - Prefer runtime checks (consistent semantics) and log startup failures instead of silent `pass`.
- **Risk/effort**: **low risk, small change**.

## 6) SIMP-006 — `get_extra_params()` duplicated across multiple schema models + key normalization mismatch
- **ID**: `SIMP-006`
- **Location**: `src/mlx_omni_server/chat/schema.py:398`, `src/mlx_omni_server/chat/schema.py:426`, `src/mlx_omni_server/images/schema.py:47`, `src/mlx_omni_server/tts/schema.py:30`, `src/mlx_omni_server/embeddings/schema.py:26`
- **Type**: `duplication`
- **Description**:
  - Four request models implement the same “standard_fields minus dump” pattern.
  - `ChatCompletionRequest.get_extra_params()` lists `\"draft-model\"` as a standard field, while generation reads `\"draft_model\"` (`src/mlx_omni_server/chat/generation_service.py:147`). This invites silent incompatibilities and extra-key drift.
- **Impact**: needless repetition, inconsistent API surface, harder to extend safely.
- **Suggested change**:
  - Factor `get_extra_params()` into a shared helper/mixin.
  - Normalize known aliases in one place (e.g. map `\"draft-model\"` → `\"draft_model\"` during `_normalize_openai_compat`).
- **Risk/effort**: **low-to-medium risk, medium effort** (touches request parsing).

## 7) SIMP-007 — `ResponseRegistry.wait_until_finished` busy-polls (could be condition-based)
- **ID**: `SIMP-007`
- **Location**: `src/mlx_omni_server/responses/registry.py:141`, `src/mlx_omni_server/responses/registry.py:187`
- **Type**: `structure`
- **Description**: `wait_until_finished()` polls every 50ms and repeatedly calls `get()` (global lock + `touch()`), while `stream_events()` already has the right condition-wait pattern.
- **Impact**: unnecessary CPU wakeups, more lock contention, TTL “touch” side effects during waits.
- **Suggested change**:
  - Rework `wait_until_finished()` to wait on `record.condition` with a timeout and return once `record.finished` is set (mirroring `stream_events`).
- **Risk/effort**: **low risk, small change**.

## 8) SIMP-008 — Embeddings service: unused imports, legacy typing, and bare `except:`
- **ID**: `SIMP-008`
- **Location**: `src/mlx_omni_server/embeddings/embeddings_service.py:1`, `src/mlx_omni_server/embeddings/embeddings_service.py:17`, `src/mlx_omni_server/embeddings/schema.py:7`
- **Type**: `non-idiomatic`
- **Description**:
  - Unused imports (`re`, `Path`, some typing aliases) and a lot of pre-3.9 typing (`Dict`, `List`, `Union`) in a repo targeting modern Python.
  - `except:` blocks in tokenizer initialization hide unexpected issues.
  - `ModelType` enum in embeddings schema appears unused.
- **Impact**: noisy code, harder debugging, inconsistent modern-Python style.
- **Suggested change**:
  - Remove unused imports/types, switch to `dict[str, ...]` / `list[...]`, use `except Exception as exc:`.
  - Drop unused schema enums/types if truly unused.
- **Risk/effort**: **low risk, small change**."
