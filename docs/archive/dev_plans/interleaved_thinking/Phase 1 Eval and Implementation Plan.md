> Archived planning document (historical). May be stale; start from `docs/README.md` for current docs.

# Phase 1 Eval and Implementation Plan

## Context: What Phase 0 already unlocked

Phase 0 fixed the underlying chat stack so tool-call steps retain replayable reasoning, and added a best-effort server cache keyed by `tool_call_id`:

- `/chat/completions` (LM + VLM) now preserves reasoning alongside tool calls in non-stream and the final streamed tool-call chunk.
- `ChatMessage` now aliases `reasoning_content` ↔ `reasoning`, so templates (e.g., GLM) and clients can use either field.
- `chat_generation_service` now reinjects missing tool-loop reasoning into incoming requests when tool outputs reference `tool_call_id`.

This means Phase 1 is primarily about **Responses API compatibility** (schemas + adapters + streaming) and **explicit replay transport** for Codex CLI (`reasoning.encrypted_content`), not about fixing core chat parsing.

## Evaluation of the current Phase 1 plan (from `Phased Plan.md`)

The Phase 1 deliverables are still correct and necessary. The repo currently:

- Hard-rejects `include` on `/responses` (`src/mlx_omni_server/responses/router.py:242`).
- Does not model `type="reasoning"` output items (`src/mlx_omni_server/responses/schema.py:75`).
- Does not emit reasoning items in non-stream conversion (`src/mlx_omni_server/responses/adapter.py:607`).
- Does not capture `delta.reasoning` during Responses streaming (`src/mlx_omni_server/responses/adapter.py:866` + `_extract_text_from_delta` is content-only).
- Drops reasoning when reconstructing history from `previous_response_id` because `response_output_items_to_chat_messages(...)` ignores reasoning-like items (`src/mlx_omni_server/responses/adapter.py:470`).

Phase 0’s server cache helps tool-loop continuity even if reasoning items are dropped, but Phase 1 must still implement reasoning items and replay semantics to meet Codex CLI expectations.

## Implementation plan (Phase 1)

### 0) Decide compatibility policy (explicit choices)

1) **`include` strictness**
   - Recommended: accept a list; support `reasoning.encrypted_content`; ignore unknown includes (warn) to avoid breaking Codex CLI/client variants.
   - Alternative strict mode: 400 unknown include values.

2) **When to emit reasoning output items**
   - Recommended: emit `type="reasoning"` output items whenever underlying chat has reasoning; include `encrypted_content` only if requested via `include`.
   - This keeps schema stable and matches “encrypted_content is opt-in”.

3) **Encrypted-content transport format**
   - Recommended for Phase 1: **integrity-first envelope**, not full cryptography.
     - `seal`: JSON → zlib → base64url, plus HMAC-SHA256 signature.
     - `unseal`: verify signature, decompress, parse.
   - Rationale: local/trusted server, but clients can be unreliable; integrity validation + robust errors matter more than secrecy.

### 1) Accept and validate `include`

**Files**
- `src/mlx_omni_server/responses/schema.py`
- `src/mlx_omni_server/responses/router.py`
- `src/mlx_omni_server/responses/adapter.py` (request→chat conversion)

**Steps**
- Add an explicit field to `ResponseRequest`:
  - `include: list[str] | None = None`
- In `create_response(...)` (`responses/router.py`):
  - Remove the hard reject.
  - Normalize to `include_list: list[str]`.
  - Validate/allowlist `reasoning.encrypted_content` (and any future supported entries).
  - Keep `include` in `request_dump` (for request echo and adapters), but **do not forward it into the chat generation request**.
- In `response_request_to_chat_request(...)` (`responses/adapter.py`):
  - `payload.pop("include", None)` so chat caching/hashing is not affected by output-format-only params, and the model layer doesn’t warn/drop it.

**Tests**
- Update `tests/unit/responses/test_responses_router.py:428`:
  - Replace “reject include” with:
    - accept `include=["reasoning.encrypted_content"]`
    - assert unknown includes are ignored or rejected depending on strictness choice.

### 2) Add a Responses `reasoning` output item type

**Files**
- `src/mlx_omni_server/responses/schema.py`

**Steps**
- Add:
  - `class ResponseOutputReasoning(BaseModel):`
    - `id: str`
    - `type: Literal["reasoning"] = "reasoning"`
    - `status: ResponseOutputItemStatus = COMPLETED`
    - `encrypted_content: str | None = None` (only present when include requests it)
    - (optional future): `summary: Any | None = None`
- Extend `ResponseOutputItem` discriminated union to include `ResponseOutputReasoning`.

### 3) Implement seal/unseal module for `reasoning.encrypted_content`

**Files**
- Add `src/mlx_omni_server/responses/reasoning_envelope.py`

**Envelope payload v1 (minimal)**
- `v: 1`
- `model: str` (Responses `model`)
- `created_at: int`
- `tool_call_ids: list[str]` (may be empty for non-tool reasoning)
- `reasoning: str` (the hidden reasoning string as extracted in Phase 0)

**API**
- `seal(payload: ReasoningEnvelope) -> str`
- `unseal(token: str) -> ReasoningEnvelope`

**Key management**
- Use an env var (e.g. `MLX_OMNI_SERVER_REASONING_HMAC_KEY`); if unset, generate a process-local key at startup (document that replay across restarts won’t work in that mode).

**Error handling**
- Invalid token / signature / decode failures → 400 `invalid_request_error` with a stable `code` (e.g., `invalid_reasoning_encrypted_content`).

### 4) Emit reasoning output items in non-stream `/responses`

**Files**
- `src/mlx_omni_server/responses/adapter.py` (`chat_response_to_response`)

**Steps**
- Detect `include_reasoning_encrypted` via `request_echo.get("include")`.
- For each `choice.message`:
  - If `message.reasoning` (or `message.reasoning_content`) exists:
    - Create a `type="reasoning"` output item:
      - `id`: deterministic (e.g., `{response_id}-reasoning-{choice.index}`)
      - `encrypted_content`: `seal(...)` only if requested by `include`.
      - `tool_call_ids`: derive from `message.tool_calls[*].id` (if present) and embed in envelope.
    - Append to `output_items` (order can be “reasoning then function_call(s) then message”).

**Tests**
- Add unit tests that mock a `ChatCompletionResponse` with:
  - reasoning-only response (no tools)
  - tool-call response with reasoning + tool_calls
  - verify `encrypted_content` is present only when include requests it
  - verify `unseal(encrypted_content)` round-trips payload and includes `tool_call_ids`.

### 5) Capture `delta.reasoning` in Responses streaming and emit reasoning output items

**Files**
- `src/mlx_omni_server/responses/adapter.py` (`OutputItemState`, `ResponseStreamAdapter`)

**Steps**
- Extend `OutputItemState.kind` to include `"reasoning"` and support `encrypted_content`:
  - Update `to_output_dict()` to return a `type="reasoning"` item.
  - Reuse `response.output_item.added` + `response.output_item.done` events (no `output_text.*` events for reasoning).
- In `ResponseStreamAdapter`:
  - Track per-choice:
    - `reasoning_text[choice_index]` (accumulator)
    - `tool_call_ids[choice_index]` (order-preserving list)
  - On each chunk:
    - If `delta.reasoning` is not `None`, append using prefix-safe logic (handle both true deltas and the Phase 0 “final chunk may carry full reasoning” case).
    - When tool calls appear, collect `call_id` as they are allocated/updated.
  - On `on_done()`:
    - For each choice index with non-empty reasoning:
      - Allocate a reasoning output item state.
      - If include requests encrypted reasoning:
        - `encrypted_content = seal({reasoning, tool_call_ids, ...})`
      - Emit `response.output_item.added` and `response.output_item.done` for it before `response.completed`.
    - Ensure `response.completed.response.output` contains the reasoning item(s).

**Tests**
- Streaming unit test that mocks streamed chat chunks containing:
  - reasoning deltas (`delta.reasoning`)
  - tool_call deltas/final tool_call chunk
  - ensures no reasoning is emitted as `output_text.delta`
  - ensures `response.completed` includes a reasoning output item and that `encrypted_content` unseals correctly when requested.

### 6) Parse reasoning input items (`type="reasoning"`) and replay into chat prompts

**Files**
- `src/mlx_omni_server/responses/adapter.py`
  - `_convert_input_to_chat_messages(...)`
  - `response_output_items_to_chat_messages(...)` (for `previous_response_id` chaining)

**Steps**
- Support input items:
  - `{"type":"reasoning","encrypted_content":"..."}`
- Bridge rule:
  - Never convert reasoning items into standalone visible `ChatMessage`s.
  - Instead:
    - `unseal(encrypted_content)` to get `{tool_call_ids, reasoning, ...}`
    - Attach `reasoning` to the next assistant tool-call step that matches those `tool_call_ids` (or, if missing, the next assistant tool-call step in sequence).
    - Also populate Phase 0 cache (`tool_loop_reasoning_cache.set(call_id, reasoning)`) for each `tool_call_id` in the envelope to keep best-effort reinjection working.
- Apply the same logic when reconstructing history from stored output items for `previous_response_id`:
  - Extend `response_output_items_to_chat_messages(...)` to process reasoning items and attach reasoning to the appropriate assistant tool-call messages.

**Tests**
- Input parsing test: `ResponseRequest.input=[reasoning_item, function_call_item, function_call_output_item, ...]` produces `ChatCompletionRequest.messages` where the assistant tool-call message has `.reasoning`.
- `previous_response_id` test: output items containing reasoning + function_call rebuild into `history_messages` that preserve reasoning on tool-call steps.

### 7) Exit criteria (Phase 1)

- `/responses` accepts `include=["reasoning.encrypted_content"]` and returns at least one `type="reasoning"` output item with `encrypted_content` when reasoning exists.
- A follow-up request that includes those reasoning items (manual `input` replay) succeeds and preserves tool-loop thinking continuity (DeepSeek/GLM).
- Streaming `/responses` captures `delta.reasoning` and includes the reasoning item in the final `response.completed` payload without leaking it as visible text deltas.

### Suggested PR sequencing

1) PR1: accept+validate `include`, add reasoning output schema, non-stream reasoning output item + seal/unseal.
2) PR2: streaming capture + reasoning output item events.
3) PR3: input parsing for reasoning items + `previous_response_id` history preservation.
