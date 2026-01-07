# Phased Implementation Notes

## Phase 0 Implementation

### Reference

- Source plan: `docs/dev_plans/interleaved_thinking/Phased Plan.md` → “Phase 0 — Fix `/chat/completions` correctness for DeepSeek/GLM interleaved thinking (tools + replay)”.

### What Phase 0 Required (from the plan) vs What Was Implemented

#### 1) Always attach reasoning to the tool-call assistant step

- **Non-stream**
  - **Plan:** After tool parsing (`decode(..., tools)`), re-attach extracted reasoning onto the returned `ChatMessage`.
  - **Implemented:**
    - LM: `src/mlx_omni_server/chat/mlx_lm/mlx_lm_model.py:385` sets `message.reasoning = reasoning` after `self._chat_tokenizer.decode(...)`.
    - VLM: `src/mlx_omni_server/chat/mlx_vlm/mlx_vlm_model.py:535` sets `message.reasoning = reasoning` after `self._chat_tokenizer.decode(...)`.

- **Streaming**
  - **Plan:** Accumulate reasoning deltas during streaming; before yielding the final `parse_buffer(...)` tool-call message, attach `reasoning_so_far`.
  - **Implemented (equivalent outcome, different mechanism):**
    - LM: `src/mlx_omni_server/chat/mlx_lm/mlx_lm_model.py:451` accumulates `raw_completion` (the unmodified streamed text), and on the final `parse_buffer(...)` chunk (`src/mlx_omni_server/chat/mlx_lm/mlx_lm_model.py:502`) derives `final_reasoning` via `ReasoningDecoder.decode(raw_completion)` and attaches it to the final tool-call delta.
    - VLM: same pattern (`src/mlx_omni_server/chat/mlx_vlm/mlx_vlm_model.py:223`, `src/mlx_omni_server/chat/mlx_vlm/mlx_vlm_model.py:279`).
  - **Notes:** Streaming still emits `delta.reasoning` chunks as before; Phase 0 additionally ensures the *final tool_calls chunk* includes full reasoning for replay.

#### 2) Server-side tool-loop reasoning cache (defensive replay)

- **Plan:** Store thinking keyed by `(conversation_fingerprint, tool_call_id)` with TTL/LRU; if clients omit unknown fields, reinject reasoning internally when tool results arrive.
- **Implemented:**
  - Cache: `src/mlx_omni_server/chat/tool_loop_reasoning_cache.py:15` implements a small in-memory TTL+LRU cache keyed by `tool_call_id` (defaults: 10 minutes, 1024 entries).
  - Write path:
    - LM non-stream: `src/mlx_omni_server/chat/mlx_lm/mlx_lm_model.py:405`
    - LM streaming final tool_calls chunk: `src/mlx_omni_server/chat/mlx_lm/mlx_lm_model.py:504`
    - VLM non-stream: `src/mlx_omni_server/chat/mlx_vlm/mlx_vlm_model.py:544`
    - VLM streaming final tool_calls chunk: `src/mlx_omni_server/chat/mlx_vlm/mlx_vlm_model.py:282`
  - Read/inject path:
    - `src/mlx_omni_server/chat/generation_service.py:63` restores cached reasoning into the assistant tool-call message when a subsequent request contains `role="tool"` messages referencing `tool_call_id` and the assistant message is missing reasoning.
    - Called before request hashing in both non-stream and stream entrypoints (`src/mlx_omni_server/chat/generation_service.py:115`, `src/mlx_omni_server/chat/generation_service.py:192`), so it applies to `/chat/completions` and `/responses` callers.

#### 3) Compatibility aliasing (`reasoning` + `reasoning_content`)

- **Plan:** Emit both `reasoning` and `reasoning_content` to maximize client compatibility.
- **Implemented:**
  - `src/mlx_omni_server/chat/schema.py:119` accepts inbound `reasoning_content` and maps it into internal `reasoning`.
  - `src/mlx_omni_server/chat/schema.py:131` emits `reasoning_content` as a computed alias of `reasoning`, ensuring:
    - HTTP responses include it automatically (`model_dump(exclude_none=True)` paths)
    - prompt construction can see it (e.g., GLM templates read `m.reasoning_content`)

### Deviations / Clarifications vs the Phase 0 Plan

- **Cache key simplification:** Implemented cache key is `tool_call_id` only (not `(conversation_fingerprint, tool_call_id)`).
  - Rationale: tool call IDs are server-generated and effectively unique; simplifying avoids underspecified fingerprinting logic and reduces complexity.
  - Tradeoff: if a client forges/reuses a `tool_call_id`, it could collide within TTL; acceptable for the repo’s trusted-localhost threat model.

- **Streaming “reasoning accumulation” strategy:** Implemented via `raw_completion` + `ReasoningDecoder.decode(...)` instead of concatenating `delta_reasoning`.
  - Rationale: avoids edge cases with stream chunk boundaries and ensures the final reasoning equals what the model emitted, regardless of how deltas were split.

- **VLM coverage:** Phase 0 work was applied to both LM and VLM codepaths to keep `/chat/completions` consistent for multimodal models.

### Files Added / Changed

- Added: `src/mlx_omni_server/chat/tool_loop_reasoning_cache.py`
- Changed:
  - `src/mlx_omni_server/chat/schema.py`
  - `src/mlx_omni_server/chat/mlx_lm/mlx_lm_model.py`
  - `src/mlx_omni_server/chat/mlx_vlm/mlx_vlm_model.py`
  - `src/mlx_omni_server/chat/generation_service.py`
  - `tests/conftest.py`
  - Added tests: `tests/unit/chat/test_phase0_interleaved_thinking.py`

### Verification

- Unit tests added to cover:
  - aliasing (`reasoning_content` ↔ `reasoning`)
  - cache reinjection on tool-loop continuation
  - non-stream tool-call reasoning survival
  - streamed final tool-call chunk reasoning survival
- Ran: `PYENV_VERSION=venv313 pyenv exec python3 -m pytest -q tests/unit` (pass).
