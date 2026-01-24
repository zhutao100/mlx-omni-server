Below is an evaluation of the **current Responses ↔ Chat Completions bridging** implementation in the `mlx-omni-server` project, focusing specifically on **reasoning/thinking fields**, **multi-round tool loops**, and **SSE streaming**, and then mapping it to the Option **A/B/C** families you outlined earlier.

---

## 1) Where the “bridge” lives in this repo

### Core adapter (Responses → Chat Completions + back)

* `src/mlx_omni_server/responses/adapter.py`

  * Request translation: `response_request_to_chat_request()` (lines ~418+)
  * Input item → chat messages (incl. tool calls + reasoning envelopes): `_convert_input_to_chat_messages()` and `_parse_reasoning_envelope()` (lines ~155+ and ~256+)
  * Non-stream response translation: `chat_response_to_response()` (lines ~794+)
  * SSE response translation: `ResponseStreamAdapter` (notably `on_chunk()` ~1129+, `on_done()` ~1375+)

### “Encrypted reasoning” token implementation (proxy-side)

* `src/mlx_omni_server/responses/reasoning_envelope.py` (entire file)

### Tool-loop reasoning continuity (server-side cache)

* `src/mlx_omni_server/chat/tool_loop_reasoning_cache.py`
* `src/mlx_omni_server/chat/generation_service.py` (restoration logic)
* `src/mlx_omni_server/chat/schema.py` (accepts `reasoning_content` and normalizes into `.reasoning`)

### Stateful “previous_response_id” support (server-side conversation state)

* `src/mlx_omni_server/responses/router.py` (merging previous history)
* `src/mlx_omni_server/responses/registry.py` (stores history + outputs)

---

## 2) What it currently does for reasoning/thinking fields

### A) Incoming request: Responses-shaped → Chat Completions-shaped

1. **`reasoning={...}` is passed through, not specially mapped**

* `ResponseRequest` allows extra fields, and `ChatCompletionRequest` allows extra fields.
* The adapter does *not* translate `reasoning.effort` into any upstream-vendor-specific knob; it effectively “passes it along” and relies on the downstream (or local model stack) to interpret it.

2. **It supports “encrypted reasoning items” as *input items*** (Responses input replay)

* `_parse_reasoning_envelope()` detects an input item like:

  ```json
  {"type":"reasoning","encrypted_content":"..."}
  ```

  and calls `unseal(...)` to recover a `ReasoningEnvelope`. (adapter.py ~155–176)

3. **Recovered reasoning is re-attached to tool-call context**

* In `_convert_input_to_chat_messages()`, it tries to associate recovered reasoning with subsequent `function_call` items (by `call_id`), and also writes it into an in-memory cache keyed by `tool_call_id`. (adapter.py ~256–418)

This aligns with the OpenAI guidance that tool-call continuations need reasoning items preserved between the last user message and tool outputs. ([OpenAI Platform][4])

### B) Outgoing response: Chat Completions-shaped → Responses-shaped

#### Non-streaming

* `chat_response_to_response()`:

  * If `choice.message.reasoning` exists, it emits a Responses `output` item:

    ```json
    {"type":"reasoning","status":"completed", ...}
    ```
  * If the client requested `include: ["reasoning.encrypted_content"]`, it adds `encrypted_content = seal(ReasoningEnvelope(...))` and includes `tool_call_ids`. (adapter.py ~812–833)

This matches the OpenAI “include reasoning.encrypted_content to enable stateless multi-turn” concept. ([OpenAI Platform][4])

#### SSE streaming

* `ResponseStreamAdapter.on_chunk()` **accumulates** `delta.reasoning` text (chat-chunk deltas) into an internal buffer (`_reasoning_by_choice`). (adapter.py ~1140–1147)
* It **does not emit** incremental Responses “reasoning text delta” events. Instead:

  * At `on_done()`, it creates/finishes a single Responses `reasoning` output item, and (optionally) adds `encrypted_content` there. (adapter.py ~1379–1395)

So: reasoning is “preserved for continuation,” but not “streamed out” as reasoning deltas.

---

## 3) “Encrypted content” in this repo is not actually encrypted

This is the single most important gap relative to OpenAI’s documented semantics.

`reasoning_envelope.py` implements:

* `seal()` = JSON → zlib compress → **HMAC-SHA256 sign** → token `v1.<payload_b64>.<sig_b64>`
* `unseal()` verifies signature, decompresses, parses JSON.

There is **no encryption** (no confidentiality). Anyone who receives the token can decode the compressed payload and recover the reasoning string if they choose; the HMAC only prevents tampering. (reasoning_envelope.py ~47–63, ~66–91)

OpenAI’s `reasoning.encrypted_content` is explicitly described as an **encrypted version of reasoning tokens** intended to be replayed for stateless continuity. ([OpenAI Platform][4])
This repo’s token is closer to “signed envelope” than “encrypted content.”

A second practical issue: the HMAC key is **ephemeral by default** unless `MLX_OMNI_SERVER_REASONING_HMAC_KEY` is configured (reasoning_envelope.py ~14–35). A restart breaks verification, so replay across proxy restarts can fail.

---

## 4) How it compares to Option A / B / C

Because I do not have the literal text of your earlier Option A/B/C write-up in this chat, I’m mapping by the standard interpretation you were driving toward:

### Option A — “Stateless, transparent pass-through of thinking/reasoning content (incl. SSE deltas)”

**Current implementation:** *partial, and conservative.*

* ✅ It can ingest `reasoning_content` (DeepSeek-style) because `ChatMessage` normalizes `reasoning_content → reasoning`. (chat/schema.py validator)
* ✅ It accumulates reasoning deltas during streaming (`delta.reasoning`).
* ❌ It does **not** emit reasoning deltas as Responses stream events; it only finalizes a reasoning item at the end.
* ❌ It does **not** preserve/display raw reasoning in Responses output unless you treat the sealed envelope as “content,” which OpenAI does not.

This is consistent with OpenAI’s stance that raw reasoning tokens are not exposed, while summaries are opt-in. ([OpenAI Platform][4])
But it diverges from DeepSeek’s “reasoning_content is exposed alongside content” model. ([DeepSeek API Docs][5])

### Option B — “Stateful proxy that stores conversation state and replays it (previous_response_id)”

**Current implementation:** *yes, strongly.*

* ✅ Implements `previous_response_id` by fetching a stored record and prepending `history_messages`. (responses/router.py ~257–280)
* ✅ Keeps a registry of responses including `history_messages` for next turn. (responses/registry.py)
* ✅ Additionally maintains a short-lived **tool-loop reasoning cache** keyed by tool_call_id. (tool_loop_reasoning_cache.py)

Caveat: this is still “soft state” (in-memory TTL), not durable persistence.

### Option C — “Encrypted reasoning items for stateless continuation (OpenAI-style encrypted_content)”

**Current implementation:** *conceptually aligned, but cryptographically incompatible.*

* ✅ Supports `include=["reasoning.encrypted_content"]` and emits an `encrypted_content` field. (responses/router.py ~234–245, adapter.py ~818–831, stream on_done ~1385–1395)
* ✅ Accepts a reasoning item with `encrypted_content` as input and uses it to restore reasoning for tool loops. (adapter.py ~155–176, ~256–418)
* ❌ Token is **not encrypted**, only signed/compressed (reasoning_envelope.py).
* ❌ Token format is custom, so it is not interoperable with OpenAI’s real encrypted reasoning items.

---

## 5) Practical implications for “correct preservation roundtrip” (multi-round + tool loops)

### With OpenAI-style Responses semantics

OpenAI’s docs emphasize that for tool calling with reasoning models you should pass back reasoning items, and for stateless mode you must request `reasoning.encrypted_content` and replay it. ([OpenAI Platform][4])
This repo’s architecture matches that flow structurally, but the “encrypted” property is not equivalent.

### With DeepSeek-style tool-loop semantics

DeepSeek explicitly requires replaying `reasoning_content` during tool-call subturns, and dropping it when the next user question begins. ([DeepSeek API Docs][5])
This repo’s approach approximates that by:

* caching reasoning per tool_call_id server-side, and/or
* embedding reasoning in a replayable “reasoning item” (envelope)

…but it does **not** provide a first-class mapping to `extra_body={"thinking":{"type":"enabled"}}` (DeepSeek enablement), nor does it surface `reasoning_content` as a peer of `content` in the Responses output schema.

---

## 6) Recommended deltas to fully match the intent of A/B/C

If your target is “preserve correctly roundtrip” across vendors and match the OpenAI contract more tightly:

1. **Make `encrypted_content` actually encrypted**

* Use an AEAD scheme (e.g., AES-GCM) with:

  * stable secret configured by env/keystore
  * key id/version for rotation
  * authenticated additional data (model, created_at)
* Keep the payload opaque to clients (as OpenAI intends). ([OpenAI Platform][4])

2. **Decouple “continuation payload” from “display payload”**

* For OpenAI-like behavior:

  * `encrypted_content` is for replay only
  * “reasoning summaries” should be opt-in (`reasoning.summary`) and appear in output if requested. ([OpenAI Platform][4])
* For DeepSeek-like behavior:

  * optionally map vendor reasoning into a Responses “reasoning_text delta” channel (if you want to display it), but that’s a product decision.

3. **Add explicit request-side mapping for enablement knobs**

* OpenAI Responses: `reasoning.effort = low|medium|high` ([OpenAI Platform][4])
* DeepSeek Chat Completions: `extra_body={"thinking":{"type":"enabled"}}` ([DeepSeek API Docs][5])
  Your adapter currently does not implement a vendor-targeted mapping layer; it mostly passes through.

4. **Streaming: consider emitting a reasoning output item earlier**
   Even if you keep “reasoning content” hidden, emitting `response.output_item.added` for the reasoning item when you first detect reasoning (then finalizing at done) can improve client UX alignment with the Responses event model.

---

If you want, I can also produce a concrete “diff-level” plan for evolving **this repo’s** adapter into a cleaner A/B/C switch (per-upstream capability detection + strict OpenAI-compat mode), but the evaluation above reflects what is implemented today and where it matches or diverges from the intended contracts.

[1]: https://chatgpt.com/c/695e0795-3dac-8326-83ee-b72fa0cbe45b "omni-server Interleaved Thinking"
[2]: https://chatgpt.com/c/6973e08f-6d8c-832d-8cba-4b12c67ca885 "Responses<->Chat Reasoning Bridge"
[3]: https://chatgpt.com/c/6951a15c-3724-832a-b06b-6cd4c96ba8df "MLX Cancellation Strategy"
[4]: https://platform.openai.com/docs/guides/reasoning "Reasoning models | OpenAI API"
[5]: https://api-docs.deepseek.com/guides/thinking_mode "Thinking Mode | DeepSeek API Docs"
