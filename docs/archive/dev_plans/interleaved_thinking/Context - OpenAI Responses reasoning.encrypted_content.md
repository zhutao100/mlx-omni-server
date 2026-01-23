> Archived planning document (historical). May be stale; start from `docs/README.md` for current docs.

Research on how OpenAI utilizes the `reasoning.encrypted_content` field in their Responses API endpoint.

````markdown
I’ll pull the latest OpenAI Responses API docs that mention `reasoning.encrypted_content`, extract the precise contract (when it appears, how clients should store/forward it, and constraints), then summarize the practical implications for proxying and inference-engine compatibility.


OpenAI’s `reasoning.encrypted_content` is a **stateless “carry-forward” mechanism for hidden reasoning tokens** in the **Responses API**, designed primarily for **`store: false` workflows** and **Zero Data Retention (ZDR)** organizations.

### What it is

* In the Responses API, reasoning-capable models can emit **reasoning items** in the `output` array. If you opt in, those reasoning items include an **`encrypted_content`** payload containing an encrypted form of the model’s internal reasoning tokens. ([OpenAI Platform][1])
* You **cannot see** raw reasoning tokens; the point is to let OpenAI re-use them **without exposing them**. (If you want a human-readable view, OpenAI instead offers *reasoning summaries* via `summary`, which is separate.) ([OpenAI Platform][1])

### How you enable it

* You must add `"reasoning.encrypted_content"` to the `include` array on each request where you want encrypted reasoning to be returned. ([OpenAI Platform][1])

### How OpenAI “uses” it (the core behavior)

When you send the encrypted reasoning items back on a subsequent Responses request, OpenAI:

1. **decrypts** the `encrypted_content` **in-memory**,
2. uses the recovered reasoning tokens as part of the model’s context for the next generation, and
3. **discards** them (and then returns newly generated reasoning as newly encrypted content). ([OpenAI Cookbook][2])

This is explicitly framed as enabling “stateless but still high-quality reasoning,” especially for ZDR / `store=false` scenarios. ([OpenAI Platform][1])

### How you’re expected to carry it forward

OpenAI recommends that with reasoning models—especially when doing function/tool calling—you pass back **the reasoning items** produced around the tool call(s), not just the tool outputs:

* “Pass back any reasoning items returned with the last function call” (and if multiple calls, pass back all reasoning items + function call items + tool outputs since the last user message). ([OpenAI Platform][1])
* Practically, you can do this either by:

  * using `previous_response_id`, **or**
  * manually passing prior `output` items into the next request’s `input`. ([OpenAI Platform][1])

Also: the platform may ignore reasoning items that aren’t relevant to the current step; including extras is “harmless” in their described behavior. ([OpenAI Platform][1])

### Where it fits in the broader Responses “conversation state” story

* If you don’t want server-side state (`store=false`) but still want multi-step reasoning/tool continuity, encrypted reasoning items are one approach. ([OpenAI Platform][1])
* For long conversations, OpenAI also provides a **compaction** flow (`/responses/compact`) that replaces prior assistant/tool content **and encrypted reasoning** with a single **encrypted compaction item**, preserving latent context while remaining opaque and compatible with ZDR-style constraints. ([OpenAI Platform][3])

### Implementation implications (for clients/proxies/engines)

* Treat `reasoning.encrypted_content` as an **opaque blob**: store it exactly as returned and replay it exactly (no parsing, no truncation, no mutation).
* If you’re building a proxy between “OpenAI-like” APIs, be aware this is **OpenAI-specific** semantics: other providers’ “reasoning replay” (e.g., DeepSeek `reasoning_content`) is not cryptographically equivalent and not interchangeable.

If you want, I can translate this into a concrete “de facto” inference-server contract (validation rules, input/output schemas, streaming deltas, and conformance tests) that supports:

* OpenAI Responses (`reasoning.encrypted_content`)
* DeepSeek-style `reasoning_content` replay
* GLM-style preserved thinking replay rules

[1]: https://platform.openai.com/docs/guides/reasoning "Reasoning models | OpenAI API"
[2]: https://cookbook.openai.com/examples/responses_api/reasoning_items "Better performance from reasoning models using the Responses API"
[3]: https://platform.openai.com/docs/guides/conversation-state "Conversation state | OpenAI API"
````

------

Based on the context above regarding
- how the DeepSeek and GLM interleaved thinking and preserved reasoning works with the OpenAI `Chat Completions` API endpoints.
- how OpenAI utilizes the `reasoning.encrypted_content` field in their `Responses` API endpoint

For the scenario quoted below,
```markdown
- The client is OpenAI Codex CLI, a standard first-party OpenAI client that leverages the `Responses` API and `reasoning.encrypted_content`.
- The client runs with customized model provider configurations, so that it connects to the localhost server, and the inference is powered by the open-weight LLM models in the server, e.g. the `DeepSeek v3.2` and `GLM 4.7`.
- The localhost server is an LLM inference engine that
  - natively supports `Chat Completions` API endpoints
  - provides translation/adapter layer, to accept `Responses` endpoint requests, and internally translates and processes using the `Chat Completions` stack, and translates the responses back to the `Responses` endpoint.
  - The engine can be powered by multiple open-weight LLM models, including the `DeepSeek v3.2` and `GLM 4.7`
```

Propose server integration options, so that the server and the client can work end-to-end to support the interleaved thinking and preserved reasoning when running with `DeepSeek v3.2` and `GLM 4.7` models under the hood.

------

````markdown
I’ll map the end-to-end state requirements of (a) Codex CLI + Responses + `reasoning.encrypted_content` and (b) DeepSeek/GLM Chat-Completions-style reasoning replay, then propose concrete adapter architectures (stateless replay, server-side session cache, and hybrid) with exact request/response translation rules and failure modes.


Below are server-side integration options that make **Codex CLI (Responses API + `reasoning.encrypted_content`)** work end-to-end against a localhost engine that **natively runs Chat Completions** with **DeepSeek v3.2** and **GLM-4.7** underneath.

---

## What must be reconciled (hard requirements)

### 1) Codex CLI expects Responses semantics (including encrypted reasoning in stateless flows)

OpenAI’s guidance for stateless reasoning is: set `store: false`, request `include: ["reasoning.encrypted_content"]`, and then pass returned reasoning items forward; `encrypted_content` is decrypted in-memory and discarded on the server side. ([OpenAI Platform][1])

### 2) DeepSeek + GLM require “reasoning replay” for tool loops (and GLM can require it across turns)

* DeepSeek “thinking mode” tool invocation requires the client to **pass back `reasoning_content`**; otherwise it returns **400**; and when a new user question starts, old `reasoning_content` should be removed (or it’s ignored). ([DeepSeek API Docs][2])
* GLM-4.7 “preserved thinking” (enabled via `clear_thinking: false`) requires forwarding the **complete, unmodified, correctly ordered** `reasoning_content` blocks; missing/truncated/reordered blocks degrade or prevent the feature. ([Z.AI][3])

### 3) Responses streaming and tool-call events differ from Chat Completions

If Codex CLI uses streaming, you must emit standard Responses SSE events such as `response.created`, `response.output_item.added`, `response.output_text.delta`, and function-call argument streaming events like `response.function_call_arguments.delta/done`. ([OpenAI Platform][4])

---

## Common adapter strategy (works for all options)

### A. Normalize to an internal “canonical conversation IR”

Represent the conversation internally as a list of items:

* `UserMessage(text)`
* `AssistantMessage(text, reasoning_blocks[], tool_calls[])`
* `ToolResult(tool_call_id, output_text)`
* Optional: `CompactionItem(opaque_blob)` (for long contexts)

This IR is the *only* thing your engine persists/derives; everything else is translation.

### B. Translate: Responses → IR → Chat Completions

1. Parse Responses `input` items and build/extend IR.
2. **If reasoning items contain `reasoning.encrypted_content`, decode it (see options below) into `reasoning_blocks[]` and attach them to the correct assistant turns.**
3. Convert IR to Chat Completions `messages` (including `role:"tool"` messages for tool results).
4. For DeepSeek/GLM backends, apply backend knobs:

   * DeepSeek: `thinking.type = enabled/disabled` as needed. ([DeepSeek API Docs][5])
   * GLM-4.7: when preserved thinking is desired, set `clear_thinking: false` (and ensure exact replay). ([Z.AI][3])

### C. Translate: Chat Completions → IR → Responses

1. Run Chat Completions, obtain `content`, `tool_calls`, and any backend-exposed `reasoning_content` (if available).
2. Update IR with `AssistantMessage(...)`.
3. Emit Responses `output` items:

   * A `message` output item (and stream `response.output_text.delta/done` if streaming). ([OpenAI Platform][4])
   * For tool calls, emit the appropriate function-call items (and stream arguments via `response.function_call_arguments.delta/done`). ([OpenAI Platform][4])
   * Emit a `reasoning` output item containing `encrypted_content` **if the request’s `include` contains `reasoning.encrypted_content`.** ([OpenAI Platform][6])

> Key point: for DeepSeek/GLM, “reasoning replay” is required to make interleaved tool use stable. OpenAI’s encrypted reasoning mechanism is *exactly* a transport for that kind of hidden state—your server just needs to implement an equivalent opaque replay channel. ([OpenAI Platform][7])

---

## Option 1 (closest to OpenAI/ZDR semantics): **True stateless “encrypted envelope”**

### What `reasoning.encrypted_content` becomes

Make `encrypted_content` an **AEAD-encrypted payload** containing enough information to reconstruct reasoning replay for DeepSeek/GLM:

Suggested plaintext schema:

* `v`: version
* `backend`: `"deepseek-v3.2"` | `"glm-4.7"`
* `turn_id` / `parent_response_id`
* `reasoning_blocks`: array of strings (exact blocks, in order)
* `tool_call_snapshot`: optional (to help validate tool-loop continuity)
* `hash_chain`: optional (see below)

Encrypt with a local key (e.g., AES-GCM). Treat it as opaque everywhere else.

### How requests work

* Client sends `include: ["reasoning.encrypted_content"]` → you return reasoning items containing encrypted envelopes. ([OpenAI Platform][6])
* On the next request, Codex CLI passes prior reasoning items back (either via `previous_response_id` or by including past output items in `input`). OpenAI recommends passing reasoning items through untouched between the last user message and function outputs; your adapter should follow the same invariant. ([OpenAI Platform][7])
* You decrypt envelopes **in memory**, reconstruct `reasoning_blocks`, inject them into Chat Completions messages for DeepSeek/GLM, then discard plaintext—mirroring OpenAI’s stated handling. ([OpenAI Platform][1])

### Why this supports DeepSeek + GLM

* DeepSeek tool loop: you can always re-inject the exact `reasoning_content` needed; missing replay should be treated as a 400 to match DeepSeek’s contract (your adapter can enforce this). ([DeepSeek API Docs][2])
* GLM preserved thinking: you preserve exact block ordering and content in the envelope; when `clear_thinking:false`, you can enforce integrity strictly. ([Z.AI][3])

### Pros / Cons

* Pros: faithful to OpenAI stateless + encrypted replay approach. ([OpenAI Platform][1])
* Cons: payload bloat; client bandwidth; more careful key management.

**When to choose:** you want “OpenAI-like” behavior, minimal server state, and predictable compatibility with `store:false` flows. ([OpenAI Platform][1])

---

## Option 2 (simplest operationally): **Stateful server cache; `encrypted_content` is a capability token**

### What `reasoning.encrypted_content` becomes

Instead of embedding reasoning, make it a **signed reference** (opaque handle):

* `encrypted_content = base64url(session_id || item_id || HMAC)`
* The server stores `(session_id, item_id) → reasoning_blocks[] (+ tool snapshot)` in an in-memory cache with TTL.

### How requests work

* On response, store the reasoning blocks server-side and return a token handle.
* On the next request, if the client echoes the token, you resolve it from cache and inject the reasoning blocks into the Chat Completions prompt.

### Why this still works end-to-end

Codex CLI only needs the field to be opaque and replayable. OpenAI describes it as a way to carry forward hidden reasoning in stateless modes; using it as a handle preserves the replay contract even if your “encryption” is effectively indirection. ([OpenAI Platform][7])

### Pros / Cons

* Pros: small payloads; easy to implement; supports long tool-heavy sessions.
* Cons: not truly stateless; cache loss breaks continuity; you must implement eviction and maybe persistence if desired.

**When to choose:** you control both sides (localhost), want robustness and performance, and don’t need strict “stateless/ZDR-like” semantics.

---

## Option 3 (recommended for long Codex sessions): **Hybrid + `/responses/compact` compatibility**

OpenAI’s conversation-state guidance notes that for long-running sessions you can call `/responses/compact`, which replaces prior assistant/tool/reasoning with a single **encrypted compaction item** that remains opaque and is compatible with encrypted reasoning workflows. ([OpenAI Platform][8])

### Hybrid design

* Use **Option 1 envelopes** (or Option 2 handles) for normal operation.
* Implement `/responses/compact`:

  * Input: full window (must still fit).
  * Output: a reduced window containing:

    * all user messages verbatim
    * one `compaction` item that encapsulates everything else (assistant/tool/reasoning state). ([OpenAI Platform][8])
* After compaction, your adapter injects a single “state blob” back into prompts (for DeepSeek/GLM), reducing context growth.

### Why it matters for GLM preserved thinking

GLM’s preserved thinking is explicitly about continuity and cache-hit rates; compaction gives you a practical escape hatch when the strict replay window gets too large. ([Z.AI][3])

---

## Backend-specific enforcement (what your adapter should do)

### DeepSeek v3.2 (thinking + tools)

* During a tool loop, require that the immediately preceding assistant turn’s reasoning is available (via decrypted envelope/handle). If missing, fail fast (400) to match DeepSeek’s stated tool-loop requirement. ([DeepSeek API Docs][2])
* When a *new user question* begins, you may drop previous reasoning blocks from the prompt (DeepSeek notes they should be removed / are ignored). ([DeepSeek API Docs][2])

### GLM-4.7 preserved thinking

* If the upstream request indicates preserved thinking (your policy, or a mapped param akin to `clear_thinking:false`), you must:

  * replay *all* prior `reasoning_blocks` in order,
  * reject or warn on mismatch (I recommend strict reject in “compat mode”), because GLM explicitly cautions about missing/truncated/reordered blocks. ([Z.AI][3])

---

## Practical recommendation

* If your goal is “behaves like OpenAI Responses + Codex CLI expects,” start with **Option 1 (stateless encrypted envelope)**, because it most closely matches OpenAI’s described encrypted reasoning lifecycle for stateless mode. ([OpenAI Platform][1])
* If your goal is “ship quickly and it works locally,” start with **Option 2 (server cache token)**, then add **Option 3 compaction** once you hit long-session scaling issues. ([OpenAI Platform][8])

If you want, I can turn the above into a concrete spec: exact IR item types, deterministic mapping rules for tool calls + streaming events, and a conformance test matrix specifically for DeepSeek v3.2 tool loops and GLM-4.7 preserved thinking.

[1]: https://platform.openai.com/docs/guides/migrate-to-responses "Migrate to the Responses API | OpenAI API"
[2]: https://api-docs.deepseek.com/guides/thinking_mode?utm_source=chatgpt.com "Thinking Mode | DeepSeek API Docs"
[3]: https://docs.z.ai/guides/capabilities/thinking-mode?utm_source=chatgpt.com "Thinking Mode - Overview - Z.AI DEVELOPER DOCUMENT"
[4]: https://platform.openai.com/docs/api-reference/responses-streaming "Streaming events | OpenAI API Reference"
[5]: https://api-docs.deepseek.com/api/create-chat-completion?utm_source=chatgpt.com "Create Chat Completion | DeepSeek API Docs"
[6]: https://platform.openai.com/docs/api-reference/responses?utm_source=chatgpt.com "Responses | OpenAI API Reference"
[7]: https://platform.openai.com/docs/guides/reasoning "Reasoning models | OpenAI API"
[8]: https://platform.openai.com/docs/guides/conversation-state?utm_source=chatgpt.com "Conversation state | OpenAI API"

````

---

Below is a concrete implementation plan for **`zhutao100/mlx-omni-server`** to “behave like OpenAI Responses + what Codex CLI expects”, while optimizing for **long-running Codex sessions** (low concurrency, low latency, robust to malformed inputs, streaming).

---

## What the repo already has (good foundations)

1. **Responses endpoint + stateful chaining (`previous_response_id`)**

   * `/v1/responses` exists and already supports merging prior history when `previous_response_id` is provided.
   * Responses can be stored in a registry and replayed via events (good for reconnect/resume).

2. **Tool loop support in Responses `input`**

   * The adapter already converts `function_call` and `function_call_output` items into chat-style tool calls/messages.

3. **Local model already produces streaming “reasoning deltas”**

   * The MLX chat wrapper can emit `delta.reasoning` separately from `delta.content` when thinking is enabled.

---

## Current gaps vs “OpenAI Responses + Codex CLI expectations”

### A) `include` support is narrow (strict allowlist)

The router supports `include=["reasoning.encrypted_content"]` and rejects unknown include values with a `400 invalid_request_error`.
This is enough for Codex CLI’s encrypted reasoning flow, but it’s not a general “accept anything” pass-through.

### B) Responses streaming reasoning is captured (implemented)

`ResponseStreamAdapter` captures `delta.reasoning` and emits a `type="reasoning"` output item (with optional `encrypted_content` when requested via `include`).

### C) `prompt_cache_key` is plumbed and used to namespace prompt cache (implemented)

`prompt_cache_key` is accepted on `/responses` and `/chat/completions` and is used as a **namespace** for local prompt KV cache reuse.
This prevents accidental cache mixing across concurrent sessions while keeping a small global cache size.

### D) Remaining: deterministic preserved-thinking injection across backends/templates

* DeepSeek v3.2 “thinking mode” tool calling requires preserving `reasoning_content` in the assistant tool-call message across turns (otherwise tool continuation can fail).
* GLM “preserved thinking” similarly requires returning unmodified thinking blocks when configured.
  Your server needs a **portable state mechanism** that works for:
* OpenAI-style encrypted reasoning items (Codex CLI),
* DeepSeek/GLM preserved-thinking requirements (local model backends).

---

## Target behavior (what we implement)

### 1) Two compatible session modes (both useful)

**Mode 1 — Stateful (best for long Codex sessions, default):**

* Use `previous_response_id` chaining (already implemented)
* Server stores history + reasoning state internally (TTL/LRU).
* Client does *not* need to resend huge histories.

**Mode 2 — Stateless (OpenAI-compatible):**

* Client passes back a `type="reasoning"` input item containing `encrypted_content` (from `include=["reasoning.encrypted_content"]`).
* Server decrypts/unseals it and reconstructs the backend-preserved-thinking requirement.

Both modes can coexist; Codex CLI typically benefits from Mode 1, but Mode 2 is important for strict OpenAI semantics and portability.

---

## Implementation plan (file-by-file, minimal but production-shaped)

### Phase 0 — Compatibility plumbing (fast wins)

#### 0.1 Allow `include` (don’t error)

**File:** `src/mlx_omni_server/responses/router.py`
Replace the current hard error:
with:

* Accept `include: list[str] | None`
* Validate only the values you support (initially: `"reasoning.encrypted_content"`)

Behavior:

* Unknown include entries → 400 invalid_request (or ignore if you prefer forward-compat; Codex usually expects strictness).

#### 0.2 Introduce `prompt_cache_key` as a first-class request field

**Files:**

* `responses/schema.py` (Responses request model)
* `chat/schema.py` (ChatCompletionRequest model)
* `chat/mlx_lm/mlx_lm_model.py`

Goal: stop treating it as “extra param” and stop dropping it. Today it’s dropped.

Implementation:

* Add `prompt_cache_key: str | None` to `ChatCompletionRequest`
* Update `get_extra_params()` to exclude it
* In `MlxLmModel`, pass the key into `PromptCacheManager` selection (see Phase 3)

---

### Phase 1 — Encrypted reasoning state (OpenAI-style) + preserved-thinking bridge

#### 1.1 Add a new Responses output item type: `reasoning`

**File:** `responses/schema.py`
Add:

* `ResponseOutputReasoning` with fields like:

  * `type: Literal["reasoning"]`
  * `id: str`
  * `status: ...`
  * `summary: Optional[...]` (optional, can be empty)
  * `encrypted_content: Optional[str]` (only present when requested via `include`)

This matches the OpenAI pattern where encrypted reasoning is returned only when included.

#### 1.2 Capture `delta.reasoning` in Responses streaming adapter (but don’t stream it to clients)

**File:** `responses/adapter.py` (`ResponseStreamAdapter.on_chunk`)
Currently it ignores reasoning deltas.

Change:

* Maintain per-choice `reasoning_buffer[choice_index] += delta.reasoning`
* Do **not** emit `output_text.delta` for reasoning
* At completion:

  * If `include` requests encrypted reasoning, emit a `reasoning` output item (either via final `response.completed` payload or via a late `response.output_item.added` + `done`)

#### 1.3 Implement “seal/unseal” for encrypted_content

**New file:** `responses/reasoning_crypto.py`

Minimal, reliable (security isn’t the priority here, integrity is):

* `seal(obj) -> str`: `json -> zlib -> base64url` + `HMAC-SHA256`
* `unseal(str) -> obj`: validate HMAC, decompress, parse

Errors:

* invalid/expired/corrupted token → 400 invalid_request

#### 1.4 Wire encrypted reasoning into request input parsing

**File:** `responses/adapter.py` conversion path (`_convert_input_item_to_chat_messages`)
Today it supports `function_call`, `function_call_output`, and `message`.
Add support for:

* `{"type":"reasoning","encrypted_content":"..."}`

Bridge logic (important for DeepSeek/GLM):

* When a reasoning item is received, **do not** turn it into a standalone visible chat message.
* Instead store it as `pending_reasoning` and attach it to the **next assistant message that has tool_calls**, or to the next assistant turn (depending on backend needs).

This is the key mapping:

* OpenAI/Codex: reasoning is a separate item
* DeepSeek/GLM: preserved thinking is typically associated with the assistant/tool-call turn

---

### Phase 2 — Make stateful chaining include reasoning state automatically

**Goal:** When `previous_response_id` is used, the server should automatically include preserved-thinking info in the backend prompt, so the client doesn’t have to.

You already merge history messages for `previous_response_id`.

Extend stored record to include:

* `reasoning_state_by_choice` (or per output item id)
* `sealed_reasoning` (optional) so you can regenerate encrypted_content if requested later

When building the next prompt:

* Merge history_messages (already)
* Also inject/attach the preserved thinking to the right assistant tool-call turns (DeepSeek/GLM requirement)

---

### Phase 3 — Long Codex session optimizations (high leverage)

#### 3.1 Prompt cache keyed per session (`prompt_cache_key`)

Today the model wrapper drops `prompt_cache_key`.

Implement:

* `PromptCacheManager` becomes effectively `dict[prompt_cache_key] -> PromptCacheManagerInstance`
* Default key when not provided:

  * if `previous_response_id`: use the *root* response id (stable across the chain)
  * else: use current response id
* Add LRU eviction across session keys (since concurrency is low, small LRU is enough)

This makes long sessions much faster: repeated prefix → maximal cache hits.

#### 3.2 Automatic compaction policy (server-side, transparent to Codex)

For long sessions, you’ll eventually hit context length / latency cliffs.

Implement a policy:

* If prompt tokens exceed threshold (e.g. 70% of `max_position_embeddings`):

  * Summarize older turns into a system “memory” message
  * Drop older tool transcripts and (optionally) drop older thinking blocks (DeepSeek docs explicitly suggest clearing to reduce bandwidth).
* Store a compacted representation in the response record.

This keeps Codex sessions stable without requiring Codex CLI to call a special endpoint.

*(If you want strict OpenAI parity later, you can also add a `/v1/responses/compact` endpoint that returns an opaque compaction item, but server-side compaction is the best ROI for a local trusted setup.)*

#### 3.3 Streaming robustness for CLI realities

Add SSE hardening:

* periodic keepalive comment/event if no tokens for N seconds
* graceful client disconnect detection + cancellation propagation
* cap stored events per response id (ring buffer) to avoid memory creep in long sessions

---

### Phase 4 — Backend-specific preserved-thinking adapters (DeepSeek v3.2 + GLM 4.7)

Implement a small backend “thinking adapter” interface used by the prompt builder:

```python
class ThinkingAdapter(Protocol):
    def attach_preserved_thinking(self, messages: list[ChatMessage], state: ThinkingState) -> list[ChatMessage]
    def extract_thinking_state(self, delta_reasoning: str | None, final_reasoning: str | None) -> ThinkingState
```

Concrete behaviors:

* **DeepSeek v3.2 tool-use thinking**

  * Ensure assistant tool-call turns retain `reasoning_content` across the tool loop
  * Optionally clear thinking after tool resolution to reduce prompt size (per DeepSeek guidance)
* **GLM preserved thinking**

  * If configured (clear_thinking=false), preserve exact blocks for continuation

This adapter layer keeps the rest of the Responses/OpenAI-facing surface stable.

---

## Acceptance tests (what “done” looks like)

1. **Codex-style tool loop works**

   * Turn 1: `/v1/responses` → emits `function_call`
   * Turn 2: client sends `function_call_output` input item → model continues
   * Already supported at input conversion level , you’re extending it to include preserved thinking state.

2. **Encrypted reasoning round-trip**

   * Request `include=["reasoning.encrypted_content"]` → response contains reasoning item with `encrypted_content`
   * Next request includes `{type:"reasoning", encrypted_content:"..."}` → server accepts and continues.

3. **Long session stays fast**

   * With stable `prompt_cache_key`, sequential turns reuse prompt cache (no repeated full-prefill).
   * Automatic compaction prevents context blowups.

---

## The smallest “first PR” that unlocks everything

If you want the minimal change-set that immediately moves you toward Codex compatibility:

1. Allow `include` (don’t 400)
2. Capture `delta.reasoning` from MLX stream  and store it (no streaming exposure)
3. Add `reasoning` output item with optional `encrypted_content`
4. Stop dropping `prompt_cache_key` and use it to key prompt cache

That combination gets you “Responses-ish + Codex-friendly” quickly, and everything else layers cleanly on top.
