[Sub-Research 1: DeepSeek And GLM Interleaved Thinking](./Context%20-%20DeepSeek%20and%20GLM%20Interleaved%20Thinking.md)

[Sub-Research 2: Codex CLI And OpenAI Responses API](./Context%20-%20OpenAI%20Responses%20reasoning.encrypted_content.md)

[Task]

Now, based on the project integration plans from both of the sub-researches, propose a fused and phased integration plan that

- defines a well phased roadmap
- satisfies the functional requirements in both plans, so that
  - the `/chat/completions` endpoint works correctly with the interleaved thinking of "DeepSeek v3.2" and "GLM 4.7", for standard clients that expects and interacts with OpenAI-compatible `/chat/completions` endpoint
  - the `/responses` endpoint works optimistically with the Codex CLI, and functions correctly with the interleaved thinking of "DeepSeek v3.2" and "GLM 4.7"

---

## North-star behavior and invariants

### A. One internal “thinking state” model, two external APIs

Implement a shared internal representation (IR) that is the only place where you *truly* model “interleaved thinking + tool loops”:

* **AssistantTurn**

  * `visible_text: str | None`
  * `tool_calls: list[ToolCall]`
  * `thinking_blocks: list[str]` (ordered, unmodified)
  * `backend: Literal["deepseek-v3.2","glm-4.7",...]`
  * `policy: ThinkingPolicy` (preserve vs clear, strictness)

Everything else is translation:

* `/chat/completions`: IR → OpenAI-compatible chat payload + (optional) nonstandard `reasoning`/`reasoning_content`
* `/responses`: IR → Responses output items + **`reasoning.encrypted_content`** when requested ([OpenAI Platform][1])

### B. “Reasoning + tool_calls must travel together”

This is the specific gap to close immediately in `mlx-omni-server`:

* In **non-stream** chat generation, the current code drops extracted reasoning whenever tools are present (it returns only the tool-parsed message)
* In **streaming**, reasoning deltas can be emitted, but the final tool-call message is produced via `parse_buffer(...)` which returns a `ChatMessage` without reasoning  and the tokenizer’s `parse_buffer` only constructs `{content, tool_calls}`

Your roadmap should treat this as Phase 0 because both `/chat/completions` and `/responses` build on the same underlying chat stack.

---

## Fused phased roadmap

### Phase 0 — Fix `/chat/completions` correctness for DeepSeek/GLM interleaved thinking (tools + replay)

**Goal:** Standard OpenAI-compatible chat clients can run tool loops against DeepSeek v3.2 / GLM 4.7 without losing the model’s preserved thinking continuity.

**Deliverables**

1. **Always attach reasoning to the tool-call assistant step**

   * **Non-stream:** After tool parsing (`decode(..., tools)`), re-attach the extracted reasoning onto the returned `ChatMessage` before returning it. This directly addresses the current drop path .
   * **Streaming:** Accumulate `reasoning_so_far` from per-chunk deltas and, before yielding the final tool-call message constructed by `parse_buffer(...)`, attach `reasoning_so_far` onto it. This closes the “final tool_calls chunk has no reasoning” bug .

2. **Introduce a server-side “tool-loop reasoning cache” (defensive for standard clients)**
   Many “OpenAI-compatible” clients will not reliably echo unknown fields (e.g., `reasoning`) back in subsequent requests. To ensure DeepSeek/GLM continuity even when the client drops it:

   * Store `thinking_blocks` keyed by `(conversation_fingerprint, tool_call_id)` with TTL/LRU.
   * When the next request arrives with `role="tool"` messages referencing `tool_call_id`, inject the cached thinking back into the prompt reconstruction (internal-only), so replay does not depend on client echoing.

3. **Add compatibility aliasing**

   * Emit both `reasoning` and `reasoning_content` (same value) on assistant messages when thinking is enabled, because DeepSeek/GLM ecosystems commonly standardize on `reasoning_content` even if your internal field is `reasoning`.

**Key touched modules**

* `src/mlx_omni_server/chat/mlx_lm/mlx_lm_model.py` (generate + stream_generate)
* `src/mlx_omni_server/chat/tool_parsing_chat_tokenizer.py` (no required structural change if you attach reasoning post-parse; `parse_buffer` itself currently omits reasoning)

**Exit criteria**

* Tool-call assistant messages (stream and non-stream) always include reasoning alongside `tool_calls`.
* A tool loop continues correctly even if the client omits reasoning fields in the next request (server cache reinjects).

---

### Phase 1 — Make `/responses` work with Codex CLI: `include=["reasoning.encrypted_content"]` + reasoning replay

**Goal:** Codex CLI (Responses API) can do long tool-heavy sessions using **encrypted reasoning transport** and/or `previous_response_id` chaining, while your backend models (DeepSeek/GLM) receive deterministic replayed thinking.

**Current repo gaps to close**

* `/responses` currently rejects any `include` value with a 400 “include is not supported”
* The Responses schema union does not include a `reasoning` output item type (only message + function_call)
* `chat_response_to_response(...)` emits only `function_call` and `message` output items—no reasoning item at all
* Streaming adapter processes deltas for text and tool calls, but there is no reasoning capture/serialization pathway , and the helper that extracts text from deltas is content-focused

**Deliverables**

1. **Accept and validate `include`**

   * Remove the hard reject in `responses/router.py` .
   * Implement allowlist validation for `reasoning.encrypted_content` (and ignore or 400 unknowns—choose strictness explicitly).
   * This aligns with the Responses API contract: `reasoning.encrypted_content` exists specifically to enable stateless multi-turn reasoning continuity ([OpenAI Platform][1]).

2. **Add a Responses `reasoning` output item**

   * Extend `responses/schema.py` to include `ResponseOutputReasoning` and add it to the discriminated union .
   * Populate `encrypted_content` only when requested via `include` (matching the documented include behavior) ([OpenAI Platform][1]).

3. **Implement seal/unseal for encrypted reasoning**

   * Implement a small “reasoning envelope” module (AEAD or HMAC+compression) that serializes:

     * backend id, ordered thinking blocks, and minimal continuity metadata
   * This becomes the payload for `reasoning.encrypted_content` output, and the input you accept back in subsequent calls.

4. **Capture `delta.reasoning` during Responses streaming and emit reasoning output**

   * Update `ResponseStreamAdapter` to accumulate per-output-item `thinking_blocks` from streaming chunks (not user-visible text).
   * At completion, emit a reasoning output item and (if requested) its `encrypted_content`.
   * Keep Responses streaming event names aligned with the official spec (e.g., `response.created`, `response.output_item.added`, etc.) ([OpenAI Platform][2]).

5. **Parse reasoning input items**

   * Extend `_convert_input_item_to_chat_messages` to accept `{type:"reasoning", encrypted_content:"..."}` (unseal; attach to IR state), in addition to existing message/function_call/function_call_output handling .

**Exit criteria**

* A `/responses` request with `include=["reasoning.encrypted_content"]` returns a reasoning output item with `encrypted_content` ([OpenAI Platform][1]).
* A follow-up request that includes that reasoning item succeeds and preserves tool-loop thinking continuity.

---

### Phase 2 — Deterministic backend replay semantics (DeepSeek v3.2 + GLM 4.7)

**Goal:** Replay does not depend on whether a specific chat template “happens” to reference your `reasoning` field.

**Deliverables**

1. **ThinkingAdapter interface**

   * `extract_state(...) -> ThinkingState`
   * `inject_state(messages, ThinkingState) -> messages` (internal prompt normalization)

2. **Deterministic prompt injection**

   * If your tokenizer/template ignores `reasoning`, synthesize an internal-only `<think>...</think>` prefix (or the backend’s expected tag) into the assistant message content before `apply_chat_template`.
   * This prevents the “reasoning exists but never reaches the model” failure mode implied by your current flow (reasoning is extracted out-of-band, then tool parsing returns a message without it) .

3. **Backend policy semantics**

   * **DeepSeek v3.2:** enforce “tool-loop requires immediate prior reasoning” when strict mode is enabled; clear historical thinking at new user turns per policy.
   * **GLM 4.7:** implement preserve/clear semantics (e.g., `clear_thinking`) by maintaining ordered blocks unchanged.

**Exit criteria**

* For both backends, tool continuation works even if the client drops reasoning fields (server reinjects deterministically).
* Preserve-vs-clear behavior is explicit and testable.

---

### Phase 3 — Long-session performance and stability (Codex-oriented)

**Goal:** Long Codex CLI sessions remain fast and memory-stable.

**Deliverables**

1. **Plumb `prompt_cache_key` end-to-end**

   * Implement `prompt_cache_key` end-to-end and use it to select/cache prompt KV state (Codex CLI sends it per session).
   * Align with Responses API support for `prompt_cache_key` ([OpenAI Platform][1]).

2. **Session-scoped prompt cache managers**

   * Future optimization: instead of a single global prompt-cache manager, consider `dict[prompt_cache_key] -> PromptCacheManager` with LRU eviction across sessions.

3. **Compaction**

   * Implement `/responses/compact` (or an equivalent transparent server-side compaction policy). The official guidance explicitly describes compaction replacing prior assistant/tool/encrypted reasoning with a single encrypted compaction item ([OpenAI Platform][3]).

4. **SSE + registry hardening**

   * Cap stored SSE events per response; `ResponseRegistry.append_events` currently appends to an unbounded list .
   * Add keepalives and robust disconnect cancellation.

**Exit criteria**

* Stable latency across long conversations (cache hits preserved per session).
* Bounded memory growth for stored events and reasoning state.

---

### Phase 4 — Conformance matrix and “compat modes”

**Goal:** Make behavior predictable across clients (standard OpenAI chat clients vs Codex CLI) and backends.

**Deliverables**

* A test matrix covering:

  * endpoints: `/chat/completions` vs `/responses`
  * modes: streaming vs non-stream, tools vs no-tools
  * state: `previous_response_id` chaining vs encrypted reasoning stateless replay ([OpenAI Platform][1])
  * backends: DeepSeek v3.2 vs GLM 4.7
* Strictness knobs:

  * “DeepSeek-strict” (400 if tool-loop reasoning missing)
  * “Best-effort” (continue with warnings and/or cleared thinking)

---

## Minimal PR sequencing that unlocks end-to-end value quickly

1. **PR1 (Phase 0):** Fix chat tool-call reasoning survival (non-stream + stream) + add tool-loop reasoning cache.
2. **PR2 (Phase 1):** Allow `include`; add reasoning output item + seal/unseal; capture reasoning in Responses streaming; parse reasoning input items.  ([OpenAI Platform][1])
3. **PR3 (Phase 2):** ThinkingAdapter + deterministic injection for DeepSeek/GLM semantics.
4. **PR4 (Phase 3):** `prompt_cache_key` plumbing + per-session cache; compaction; registry bounds.  ([OpenAI Platform][3])

This fused plan satisfies both of your functional requirements:

* `/chat/completions` becomes correct for DeepSeek/GLM tool+thinking loops (reasoning preserved and replayed, even with standard clients).
* `/responses` becomes Codex CLI–friendly by implementing `reasoning.encrypted_content`, streaming events alignment, and long-session performance primitives (`prompt_cache_key`, compaction). ([OpenAI Platform][1])

[1]: https://platform.openai.com/docs/api-reference/responses "Responses | OpenAI API Reference"
[2]: https://platform.openai.com/docs/api-reference/responses-streaming "Streaming events | OpenAI API Reference"
[3]: https://platform.openai.com/docs/guides/conversation-state "Conversation state | OpenAI API"
