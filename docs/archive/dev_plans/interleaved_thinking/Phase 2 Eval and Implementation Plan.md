> Archived planning document (historical). May be stale; start from `docs/README.md` for current docs.

# Phase 2 Eval and Implementation Plan

## Context: What Phase 0 + Phase 1 already unlocked

Phase 0 + Phase 1 together established **transport + replay** for interleaved thinking:

- `/chat/completions` tool-call steps retain replayable reasoning (non-stream + final streamed tool-call chunk), and missing reasoning can be best-effort restored via `tool_loop_reasoning_cache` before hashing/generation.
- `/responses` now supports:
  - `include=["reasoning.encrypted_content"]`
  - `type="reasoning"` output items (non-stream + streaming events)
  - replay via input `type="reasoning"` items and via `previous_response_id` history reconstruction.

This means Phase 2 is no longer about “how to transport reasoning”, but about **ensuring the backend model deterministically receives the replayed reasoning**, regardless of chat template behavior.

## Evaluation of the Phase 2 plan (from `Phased Plan.md`)

The Phase 2 goal (“replay does not depend on whether a template happens to reference `reasoning`”) is correct and still required.

### Why Phase 2 is still needed

- Today, replayed reasoning is stored on `ChatMessage.reasoning` (and aliased as `reasoning_content`), but prompt construction is still **template-dependent**:
  - GLM templates explicitly render `m.reasoning_content` into `<think>...</think>` (`src/mlx_omni_server/chat/templates/glm4_chat_template.jinja:52`), so GLM can work.
  - The Qwen3 template does **not** reference `reasoning`/`reasoning_content` at all (`src/mlx_omni_server/chat/templates/qwen3_chat_template.jinja:88`), so replayed reasoning can be silently dropped for any backend using that template family.
- The current request preprocessor (`src/mlx_omni_server/chat/generation_service.py:63`) restores missing reasoning onto assistant tool-call messages, but it does not guarantee the tokenizer/template will serialize it into the final prompt.

### Phase 2 plan adjustments (to fit the current codebase)

The Phase 2 plan is directionally right, but can be implemented with a smaller surface area than a full new “IR”:

- A minimal “ThinkingAdapter” is still useful, but it can be **narrowly scoped to prompt normalization** (inject + policy enforcement) rather than a large shared IR.
- Deterministic injection should hook into the single choke points where prompts are built:
  - LM: `ChatTokenizer.encode(...)` → `tokenizer.apply_chat_template(...)` (`src/mlx_omni_server/chat/tools/chat_tokenizer.py:49`)
  - VLM: `apply_chat_template(...)`/`get_chat_template(...)` path (`src/mlx_omni_server/chat/mlx_vlm/mlx_vlm_model.py:761`)

## Implementation plan (Phase 2)

### 0) Decide “policy knobs” (explicit, minimal)

Introduce an internal policy model with defaults chosen for this repo’s expectations (trusted clients, low concurrency, avoid over-engineering):

1) **Injection scope (default: tool-loop only)**
   - Default behavior should inject `<think>...</think>` only onto assistant messages that are part of the active tool loop window (typically “since the last user message”), minimizing context bloat.
2) **Strictness (default: best-effort)**
   - “DeepSeek-strict” mode can fail fast (400) when a tool-loop continuation is missing required prior reasoning.
   - Default remains best-effort (continue without raising) to match existing behavior.
3) **Preserve vs clear thinking (default: clear on new user turn)**
   - Default clears historical thinking at new user turns by only injecting within the “since last user” window.
   - GLM-style preservation can be enabled via an explicit flag (mapped to template/config later).

### 1) Add a minimal `ThinkingAdapter` layer (prompt-focused)

**New module suggestion**
- `src/mlx_omni_server/chat/thinking_adapter.py`

**Recommended API (minimal)**
- `ThinkingPolicy` (dataclass or Pydantic model):
  - `mode: Literal["best_effort","deepseek_strict"]`
  - `inject_scope: Literal["tool_loop_window","all_assistant_messages"]`
  - `thinking_tag: str` (default from `ChatTokenizer.thinking_tag`)
- `inject_reasoning_into_conversation(conversation: list[dict[str, Any]], *, policy: ThinkingPolicy) -> list[dict[str, Any]]`
  - Operates on the conversation dicts *only* (internal), does not mutate `ChatMessage`.
  - For eligible assistant messages:
    - Convert `reasoning` / `reasoning_content` into a `<think>...</think>\n` prefix inside `content`.
    - Drop `reasoning`/`reasoning_content` keys from the dict to avoid duplicate rendering in templates that already use them.
    - Avoid double-injection if content already contains `<think>`/`</think>`.
- `validate_tool_loop_reasoning(messages: list[ChatMessage], *, policy: ThinkingPolicy) -> None`
  - Optional helper for strict mode: raise a structured error if a `role="tool"` message references a `tool_call_id` whose preceding assistant tool-call message has no reasoning after best-effort restoration.

### 2) Wire deterministic injection into prompt construction (LM + VLM)

#### 2.1 LM: `ChatTokenizer.encode(...)`

Update `src/mlx_omni_server/chat/tools/chat_tokenizer.py:69`:

- After `msg_dict = message.model_dump(exclude_none=True)` normalization but before `conversation.append(msg_dict)`:
  - Identify:
    - assistant message kind
    - last user index (for default tool-loop window behavior)
    - whether message is within the tool-loop window and has tool calls (default scope).
  - Apply `ThinkingAdapter.inject_reasoning_into_conversation(...)` (or inline helper) to ensure `<think>...</think>` ends up in `content` for the assistant tool-call step.

This is the key Phase 2 guarantee: **even templates that ignore `reasoning_content` will still receive the thinking via `content`.**

#### 2.2 VLM: template message conversion path

Update `src/mlx_omni_server/chat/mlx_vlm/mlx_vlm_model.py:776` path:

- Today it copies `reasoning_content` fields from `chat_messages` into the template message dicts.
- Instead (or additionally), apply the same “inject into content + drop reasoning fields” normalization to the template message dicts so VLM replay is also template-independent.

### 3) Implement backend policy semantics (Phase 2’s “behavior knobs”)

#### 3.1 DeepSeek strictness mode (optional, but planned)

- Add a request-mappable knob, e.g.:
  - extra param `thinking_mode="deepseek_strict"` or `deepseek_strict_reasoning=true`
- In `src/mlx_omni_server/chat/generation_service.py:63` preprocessing flow:
  - Run best-effort restoration from cache first.
  - If strict mode and any tool-loop-required assistant reasoning is still missing, return 400 `invalid_request_error`.

#### 3.2 GLM preserve/clear semantics

- Provide a single knob (default off) to “preserve thinking across turns”:
  - If enabled, inject reasoning for all eligible assistant messages (or keep window but extend definition).
  - If disabled (default), inject only within the “since last user” window.
- Keep this policy separate from the Jinja templates: templates become a rendering detail, not the source of truth for replay semantics.

### 4) Tests (make Phase 2 regression-proof)

Add unit tests that demonstrate Phase 2’s deterministic behavior:

1) **Template-independent injection**
   - Construct a request where an assistant tool-call message has `reasoning` + `tool_calls` + empty content.
   - Ensure the prompt built through `ChatTokenizer.encode(...)` contains `<think>...</think>` even when using a template that does not reference `reasoning_content` (Qwen3).
2) **No duplication for GLM templates**
   - Ensure injected content does not cause duplicated `<think>` blocks (drop `reasoning_content` keys after injection).
3) **Strict mode behavior**
   - With strict enabled, verify missing reasoning for a tool-loop continuation produces a 400 (and best-effort mode continues).

### 5) Exit criteria (Phase 2)

- For both LM and VLM prompt building paths, replayed reasoning reliably reaches the backend via prompt text (not just as out-of-band fields).
- Tool-loop continuation does not depend on chat template support for `reasoning`/`reasoning_content`.
- Preserve-vs-clear semantics and (optional) DeepSeek strictness are explicit, testable, and do not rely on template quirks.

## Suggested PR sequencing (Phase 2)

1) PR1: Add ThinkingAdapter + LM injection in `ChatTokenizer.encode(...)` + unit tests.
2) PR2: Apply same injection to VLM prompt path + add VLM-focused tests.
3) PR3: Add strictness + preserve/clear policy wiring + error envelope tests.
