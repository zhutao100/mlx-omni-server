Below is (1) what the docs establish for OpenAI and DeepSeek, then (2) concrete plan options for a **Responses-shaped ⇄ Chat Completions upstream** proxy that needs to preserve “reasoning/thinking” behavior and artifacts **round-trip**, including **SSE streaming** and **multi-round conversations**.

---

## 1) What the docs say

### OpenAI (Responses API): `reasoning={"effort": ...}` and `reasoning.encrypted_content`

**Reasoning effort**

* OpenAI’s reasoning configuration supports an “effort” dial. For recent GPT-5.x reasoning models, supported values include **`none`, `minimal`, `low`, `medium`, `high`, `xhigh`**, with **defaults varying by model family**. ([OpenAI Platform][1])
* For earlier o-series reasoning models, effort levels are commonly described as **`low | medium | high`** (again with model-specific defaults). ([OpenAI Platform][2])

**Encrypted reasoning items (`reasoning.encrypted_content`)**

* The Responses API can return **encrypted reasoning items** when you opt in via the `include` parameter (e.g. `include: ["reasoning.encrypted_content"]`). ([OpenAI Platform][3])
* The **`encrypted_content`** is an **opaque blob** attached to a reasoning item, intended for **clients to store and replay** so the model can continue reasoning across tool loops / turns without revealing raw chain-of-thought. ([OpenAI Platform][2])

**Conversation state**

* Responses supports multi-turn flows via `previous_response_id` (and related “conversation state” concepts), with constraints like “cannot be used with `conversation`” noted in the API reference. ([OpenAI Platform][4])

**SSE streaming: dedicated reasoning events exist**

* Responses streaming defines explicit event types for reasoning text and reasoning summaries (e.g. `response.reasoning_text.delta/done`, `response.reasoning_summary_text.delta/done`). ([OpenAI Platform][5])

---

### DeepSeek (Chat Completions–style): `extra_body={"thinking": {"type": "enabled"}}` and `reasoning_content`

**Thinking toggle**

* DeepSeek’s “thinking mode” is enabled via an extra request body field conceptually equivalent to sending top-level JSON like `thinking: { type: "enabled" }` (the docs illustrate this using SDK `extra_body=...`). ([DeepSeek API Docs][6])

**Reasoning output channel**

* When enabled, DeepSeek returns the model’s reasoning in a separate field: **`reasoning_content`**, distinct from the user-visible `content`. ([DeepSeek API Docs][6])

**Critical multi-round rule (tool loops / continued reasoning)**

* During a tool-loop / multi-call “same user question” sequence, DeepSeek requires you to **send back `reasoning_content`** to continue reasoning; but when the **next user question** begins, the prior `reasoning_content` should be removed (and if included, the API ignores it). ([DeepSeek API Docs][6])

---

## 2) Proxy bridging: plan options

Your proxy shape is:

**Client ⇄ (local HTTP proxy speaking “Responses”) ⇄ Upstream Chat Completions**

Key design decision: **how the proxy transports and replays “reasoning state”** across:

* **single request tool-loops** (multiple upstream calls under one client request),
* **multi-round conversations** (multiple client requests),
* **streaming** (SSE deltas).

Below are workable options, ordered by increasing fidelity and complexity.

---

## Option A — Capability-based passthrough with standardized Responses reasoning events (recommended baseline)

**Goal:** For upstreams that expose reasoning separately (DeepSeek’s `reasoning_content`), map it into **Responses-native reasoning stream/events**, while keeping the proxy mostly stateless across user turns.

### Request mapping (Responses → Chat Completions)

1. **Infer “thinking enabled?”**

   * If incoming Responses request has `reasoning.effort != "none"` (or is present), enable upstream thinking:

     * DeepSeek: send `thinking: { type: "enabled" }`. ([DeepSeek API Docs][6])
   * If `reasoning.effort == "none"` (or absent), disable thinking (or omit the field).
2. **Effort level mapping (lossy but controllable)**

   * DeepSeek’s public control is primarily enable/disable; the proxy can map effort to configurable budgets:

     * `minimal/low` → thinking enabled + lower token budget
     * `medium/high/xhigh` → thinking enabled + higher token budget
       This is necessarily heuristic because the DeepSeek knob is not a 1:1 “effort enum” as in OpenAI. ([DeepSeek API Docs][6])

### Response mapping (Chat Completions → Responses)

* If upstream returns **`reasoning_content`**, map it into a **Responses “reasoning” output item** and stream it using:

  * `response.reasoning_text.delta` / `response.reasoning_text.done` ([OpenAI Platform][5])
* Map normal assistant text (`content`) into the usual Responses text stream (your proxy already does this).

### SSE streaming behavior

* When upstream chunk contains `delta.reasoning_content`: emit `response.reasoning_text.delta`.
* When upstream chunk contains `delta.content`: emit `response.output_text.delta` (and associated item/content indices per Responses streaming schema).
* Close with the appropriate `*.done` events.

### Multi-round / tool-loop replay rules (DeepSeek correctness)

* **Within a single client request** (where your proxy does tool calls and then re-queries the model), **replay the just-produced `reasoning_content` back to DeepSeek** exactly as required by their docs. ([DeepSeek API Docs][6])
* **Across client requests (next user question)**: do **not** include prior `reasoning_content` in the upstream prompt; treat it as ephemeral. ([DeepSeek API Docs][6])

**Pros**

* Preserves reasoning as a distinct channel end-to-end (clients can subscribe to it).
* Aligns with Responses’ own streaming event taxonomy for reasoning. ([OpenAI Platform][5])

**Cons**

* If a client expects OpenAI-style *encrypted* reasoning items, this option provides plaintext reasoning (unless you add Option C).

---

## Option B — Stateful proxy keyed by `response.id` / `previous_response_id` (best UX for clients)

**Goal:** Make the client’s life easy: let them use Responses-style conversation state while the proxy stores upstream state.

### How it works

* When the proxy returns a Responses `id`, it also stores (locally):

  * upstream message transcript,
  * last-turn DeepSeek `reasoning_content` (only for in-flight tool loops),
  * any vendor-specific state needed to continue.
* When the client later sends `previous_response_id`, the proxy reconstructs the upstream chat context without requiring the client to resend everything. ([OpenAI Platform][4])

### Reasoning preservation

* You can still stream reasoning to the client via `response.reasoning_text.*` events. ([OpenAI Platform][5])
* For DeepSeek, enforce “do not preserve old reasoning_content into the next user question” by construction (only keep it for the duration of a tool loop). ([DeepSeek API Docs][6])

**Pros**

* Most “OpenAI-like” client experience for multi-round conversations (`previous_response_id` flows).
* Easy to implement correct DeepSeek tool-loop replay without involving the client.

**Cons**

* Requires local persistence (even if only in-memory), plus eviction policy and concurrency safety.

---

## Option C — Proxy-issued “encrypted reasoning items” (stateless client + privacy boundary)

**Goal:** Emulate OpenAI’s `reasoning.encrypted_content` semantics even when the upstream is Chat Completions (e.g., DeepSeek), so reasoning state can be replayed **without exposing plaintext** and without proxy persistence.

### Mechanism

1. If client requests `include: ["reasoning.encrypted_content"]`, the proxy returns a reasoning output item with `encrypted_content` populated. ([OpenAI Platform][3])
2. The proxy encrypts a compact state payload (AEAD) containing only what’s needed for continuation, e.g.:

   * last-turn DeepSeek `reasoning_content` (for tool-loop continuation),
   * upstream provider identifier,
   * a short expiry timestamp.
3. On the next request, the client replays that reasoning item in the Responses `input` array (mirroring OpenAI’s “replay reasoning items” concept). ([OpenAI Platform][7])
4. The proxy decrypts and uses it **only where allowed**:

   * **allowed:** within the same user question’s tool-loop continuation (DeepSeek requirement). ([DeepSeek API Docs][6])
   * **not allowed:** carry-over into the next user question for DeepSeek (discard / ignore on new turn). ([DeepSeek API Docs][6])

### Streaming

* You typically **cannot stream `encrypted_content` meaningfully**; you buffer reasoning deltas and finalize one encrypted blob at end-of-response.
* Optionally, you can provide *either*:

  * plaintext reasoning stream (`response.reasoning_text.delta`) **or**
  * encrypted-only (no plaintext reasoning events), controlled by proxy config.

**Pros**

* Matches the *shape* and replay pattern of OpenAI encrypted reasoning items. ([OpenAI Platform][2])
* Avoids proxy persistence and avoids exposing plaintext reasoning.

**Cons**

* This is **proxy-defined encryption**, not OpenAI’s model-native encrypted reasoning format—so it’s interoperable only with your proxy (still often fine for “local proxy” goals).

---

## Recommended implementation shape (works across A/B/C)

Implement a per-upstream **“ReasoningBridge”** with explicit capabilities:

* `supports_thinking_toggle` (DeepSeek: yes) ([DeepSeek API Docs][6])
* `supports_reasoning_effort_enum` (OpenAI Responses: yes; DeepSeek: not 1:1) ([OpenAI Platform][1])
* `emits_reasoning_channel` (DeepSeek: `reasoning_content`) ([DeepSeek API Docs][6])
* `requires_reasoning_replay_within_turn` (DeepSeek: yes, tool loops) ([DeepSeek API Docs][6])
* `supports_encrypted_reasoning_items` (OpenAI Responses: yes; proxy can emulate via Option C) ([OpenAI Platform][8])
* `stream_event_mapping` → map upstream deltas to Responses events (`response.reasoning_text.*`, `response.reasoning_summary_text.*`). ([OpenAI Platform][5])

This keeps your main translator clean: the bridge decides how to interpret incoming `reasoning.effort`, whether to ask the upstream for thinking, how to replay state, and how to stream it back out.

---

If you tell me which upstream(s) you need to support first (OpenAI Chat Completions, DeepSeek, both, others), I can give you a concrete event-by-event SSE mapping sequence (including `output_index`/`content_index` bookkeeping) and the exact “where to store/replay” logic for tool-loop turns versus new user turns.

[1]: https://platform.openai.com/docs/guides/latest-model "Using GPT-5.2 | OpenAI API"
[2]: https://platform.openai.com/docs/guides/reasoning "Reasoning models | OpenAI API"
[3]: https://platform.openai.com/docs/api-reference/responses "Responses | OpenAI API Reference"
[4]: https://platform.openai.com/docs/api-reference/responses?utm_source=chatgpt.com "Responses | OpenAI API Reference"
[5]: https://platform.openai.com/docs/api-reference/responses-streaming "Streaming events | OpenAI API Reference"
[6]: https://api-docs.deepseek.com/guides/thinking_mode "Thinking Mode | DeepSeek API Docs"
[7]: https://platform.openai.com/docs/api-reference/responses-streaming/response/in_progress?lang=curl&utm_source=chatgpt.com "Streaming events | OpenAI API Reference"
[8]: https://platform.openai.com/docs/api-reference/conversations/object?utm_source=chatgpt.com "Conversations | OpenAI API Reference"
