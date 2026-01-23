> Archived planning document (historical). May be stale; start from `docs/README.md` for current docs.

# Phased Implementation Notes

## Phase 0 Implementation

### Reference

- Source plan: `docs/archive/dev_plans/interleaved_thinking/Phased Plan.md` → “Phase 0 — Fix `/chat/completions` correctness for DeepSeek/GLM interleaved thinking (tools + replay)”.

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

## Phase 1 Implementation

### Reference

- Source plan: `docs/archive/dev_plans/interleaved_thinking/Phased Plan.md` → “Phase 1 — Make `/responses` work with Codex CLI: `include=[\"reasoning.encrypted_content\"]` + reasoning replay”.

### What Phase 1 Required (from the plan) vs What Was Implemented

#### 1) Accept and validate `include`

- **Plan:** Accept `include`, support `reasoning.encrypted_content`, and ensure it does not affect chat request hashing/caching.
- **Implemented:**
  - `ResponseRequest` now defines `include: list[str] | None` (`src/mlx_omni_server/responses/schema.py:143`).
  - `/responses` validates `include` with an allowlist and rejects unknown values (`src/mlx_omni_server/responses/router.py:234`).
  - `include` is removed before translating to `ChatCompletionRequest` so it does not affect request hashing/caching (`src/mlx_omni_server/responses/adapter.py:299`).

#### 2) Add a Responses `reasoning` output item type

- **Plan:** Extend the Responses schema with a discriminated `type="reasoning"` output item.
- **Implemented:**
  - Added `ResponseOutputReasoning` and extended `ResponseOutputItem` union (`src/mlx_omni_server/responses/schema.py:75`).

#### 3) Implement seal/unseal for `reasoning.encrypted_content`

- **Plan:** Provide an opaque replay token for reasoning with integrity validation.
- **Implemented:**
  - Added `src/mlx_omni_server/responses/reasoning_envelope.py` implementing:
    - `seal`: JSON → zlib → base64url + HMAC-SHA256 signature.
    - `unseal`: signature verification + decompress + JSON parse.
  - Key behavior:
    - Uses `MLX_OMNI_SERVER_REASONING_HMAC_KEY` when set; otherwise generates an **ephemeral process-local key** (tokens cannot be unsealed after restart without a configured key).

#### 4) Emit non-stream reasoning output items

- **Plan:** When chat completions contain reasoning, emit a `type="reasoning"` output item; include `encrypted_content` only when requested by `include`.
- **Implemented:**
  - `chat_response_to_response(...)` emits `type="reasoning"` when `choice.message.reasoning` exists (`src/mlx_omni_server/responses/adapter.py:609`).
  - `encrypted_content` is only included when `include` contains `reasoning.encrypted_content` (`src/mlx_omni_server/responses/adapter.py:617`).

#### 5) Capture `delta.reasoning` during streaming and emit reasoning output item events

- **Plan:** Accumulate reasoning during streaming; emit a `type="reasoning"` output item in events and in the final `response.completed` payload.
- **Implemented:**
  - `ResponseStreamAdapter` captures `delta.reasoning` and accumulates it per choice (`src/mlx_omni_server/responses/adapter.py:969`).
  - Tracks tool call IDs per choice (for envelope metadata) (`src/mlx_omni_server/responses/adapter.py:871`).
  - On `on_done()`, emits `response.output_item.added` + `response.output_item.done` for `type="reasoning"` and includes it in `response.completed` (`src/mlx_omni_server/responses/adapter.py:1195`).

#### 6) Parse reasoning input items and preserve reasoning via `previous_response_id` chaining

- **Plan:** Accept `{"type":"reasoning","encrypted_content":"..."}` input items and preserve reasoning when rebuilding history from stored output items.
- **Implemented:**
  - Input parsing:
    - `_convert_input_to_chat_messages(...)` recognizes `type="reasoning"` items, unseals them, and attaches reasoning onto the matching assistant tool-call step; also hydrates Phase 0 cache (`src/mlx_omni_server/responses/adapter.py:248`).
  - History preservation:
    - `response_output_items_to_chat_messages(...)` consumes reasoning items when reconstructing history, and reattaches reasoning to assistant tool-call messages; also hydrates Phase 0 cache (`src/mlx_omni_server/responses/adapter.py:545`).

### Deviations / Clarifications vs the Phase 1 Plan

- **`include` strictness:** The Phase 1 plan recommended “ignore unknown includes (warn)”. The implementation is stricter:
  - Unknown include values return a 400 `invalid_request_error` (`src/mlx_omni_server/responses/router.py:237`).
  - Rationale: fail fast on malformed clients; simplifies compatibility surface. If needed, this can be relaxed later to “ignore unknowns”.

### Files Added / Changed

- Added:
  - `src/mlx_omni_server/responses/reasoning_envelope.py`
- Changed:
  - `src/mlx_omni_server/responses/router.py`
  - `src/mlx_omni_server/responses/schema.py`
  - `src/mlx_omni_server/responses/adapter.py`
  - `tests/unit/responses/test_responses_router.py`

### Verification

- Added unit tests covering:
  - non-stream reasoning item + envelope roundtrip
  - streaming reasoning item events + envelope roundtrip
  - parsing reasoning input items and attaching to assistant tool-call steps
  - `previous_response_id` history reconstruction preserving tool-call reasoning
- Ran: `PYENV_VERSION=venv313 pyenv exec python3 -m pytest -q tests/unit` (pass).

## Phase 2 Implementation

### Reference

- Source plan: `docs/archive/dev_plans/interleaved_thinking/Phased Plan.md` → “Phase 2 — Deterministic backend replay semantics (DeepSeek v3.2 + GLM 4.7)”.

### What Phase 2 Required (from the plan) vs Current Implementation

#### 1) ThinkingAdapter interface

- **Plan:** Introduce a backend-aware `ThinkingAdapter` (extract + inject) and use it as the deterministic IR boundary.
- **Current status:** **Not implemented.**
  - There is no `ThinkingState` / adapter abstraction; reasoning remains a string field on `ChatMessage` (`src/mlx_omni_server/chat/schema.py:112`).

#### 2) Deterministic prompt injection

- **Plan:** If a tokenizer/template ignores `reasoning`, inject backend-specific `<think>...</think>` (or equivalent) into the assistant message *content* before `apply_chat_template`.
- **Current status:** **Not implemented (still template-dependent).**
  - GLM templates explicitly render `m.reasoning_content` inside `<think>...</think>` (`src/mlx_omni_server/chat/templates/glm4_chat_template.jinja:52`), so replay works there.
  - For non-GLM templates/backends, there is no guarantee that `ChatMessage.reasoning` will reach the model unless the template explicitly references it.

#### 3) Backend policy semantics (DeepSeek strictness, GLM preserve/clear)

- **Plan:** Make preserve/clear semantics explicit and testable; optionally enforce DeepSeek strict behavior when tool-loop reasoning is missing.
- **Current status:** **Not implemented.**
  - Phase 0 provides best-effort reasoning reinjection via `tool_loop_reasoning_cache`, but no explicit “strict vs best-effort” policy exists yet (`src/mlx_omni_server/chat/generation_service.py:63`).
  - GLM preserve/clear semantics are currently driven only by template parameters (where supported) and are not modeled as a first-class policy.

### What Phase 0 + Phase 1 Already Unblocked (Relevant to Phase 2)

- Reasoning now survives tool-call steps across `/chat/completions` and `/responses` (Phase 0 + Phase 1).
- `/responses` now has a replay mechanism (`reasoning.encrypted_content`) and history reconstruction that preserves tool-call reasoning (Phase 1).
- This provides the necessary transport and caching foundation for Phase 2, but **does not yet make replay deterministic across backends/templates**.

## Phase 3 (partial) Implementation

### Reference

- Source plan: `docs/archive/dev_plans/interleaved_thinking/Phased Plan.md` → “Phase 3 — Long-session performance and stability (Codex-oriented)”.

### What Was Implemented (Option B1: namespaced prompt cache)

- `prompt_cache_key` is now a first-class request field:
  - Chat: `src/mlx_omni_server/chat/schema.py` (`ChatCompletionRequest.prompt_cache_key`)
  - Responses: `src/mlx_omni_server/responses/schema.py` (`ResponseRequest.prompt_cache_key`)
- Prompt KV cache reuse is scoped by `prompt_cache_key`:
  - LM: `src/mlx_omni_server/chat/mlx_lm/prompt_cache.py` namespaces cache keys and filters reuse/fork by session key.
  - VLM: `src/mlx_omni_server/chat/mlx_vlm/prompt_cache.py` namespaces cache keys and filters reuse by session key.
- Model wrappers pass `prompt_cache_key` into the prompt cache managers:
  - LM: `src/mlx_omni_server/chat/mlx_lm/mlx_lm_model.py`
  - VLM: `src/mlx_omni_server/chat/mlx_vlm/mlx_vlm_model.py`
- Added tests:
  - `tests/unit/chat/test_prompt_cache_session_key.py`

### Notes / Tradeoffs

- This intentionally keeps a single global prompt-cache manager with a small `max_caches` (global LRU eviction).
- The plan’s “per-session prompt cache manager dict” remains a future optimization if concurrent sessions become common.
