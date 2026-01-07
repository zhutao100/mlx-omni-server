I’ll (1) pull the key protocol details from GLM-4.7 “Thinking Mode” docs, (2) map them against DeepSeek’s reasoning_content + replay-for-tools contract, and (3) propose a unified, implementable “de facto” spec with compatibility modes and conformance tests.


## What GLM-4.7 “Thinking Mode” actually specifies

From Z.AI’s GLM-4.7 documentation:

* **Thinking is on by default** for GLM-4.7; you disable it via `thinking: { type: "disabled" }`. ([Z.AI][1])
* **Interleaved thinking** (think → tool → think → tool …) is supported by default, and when tools are used, **thinking blocks must be preserved and returned with tool results**. ([Z.AI][1])
* **Preserved thinking** is an opt-in behavior (default on a “Coding Plan” endpoint; default off on the standard endpoint). You enable it on the standard endpoint by setting `thinking.clear_thinking: false`, and you must return **complete, unmodified `reasoning_content`**, with **consecutive blocks exactly matching the original sequence** (no edits/reordering). ([Z.AI][1])
* **Turn-level control**: you can toggle thinking per request within the same session. ([Z.AI][1])
* Their example explicitly accumulates `delta.reasoning_content` during streaming and appends `reasoning_content` back into `messages` along with `tool_calls` before sending tool results. ([Z.AI][1])

## DeepSeek-style reasoning recap (the pieces that matter for engines)

DeepSeek’s docs establish two distinct “reasoning behaviors” that an engine must treat differently:

1. **Thinking + tools loop (DeepSeek Chat w/ thinking enabled)**

* During a tool loop, the client must **echo `reasoning_content` back** so the model can continue reasoning; otherwise the API returns **400**. ([DeepSeek API Docs][2])
* When a *new user question* begins, prior `reasoning_content` should be removed; if retained, it is **ignored**. ([DeepSeek API Docs][2])

2. **DeepSeek “reasoning model” multi-round chat (deepseek-reasoner)**

* Outputs `reasoning_content` alongside `content`, but **including `reasoning_content` in the next request causes a 400**. ([DeepSeek API Docs][3])
* Also: function calling is not supported for `deepseek-reasoner`. ([DeepSeek API Docs][3])

These two modes are easy to conflate; inference engines should not.

---

## What the DeepSeek + GLM “de facto” guide actually requires

For inference engines that want to be *functionally compatible* with the recent “reasoning + tools” conventions, the non-negotiables are:

1. **Two-channel assistant output**

   * Return *user-visible* `content`, and return *separately* the model’s hidden reasoning as `reasoning_content` (and/or a compatibility alias). This separation must work in both **non-stream** and **streaming delta** forms.

2. **Tool-loop continuation requires replayable reasoning**

   * When the assistant emits a **tool call**, the assistant message for that step must still carry the associated reasoning channel (so the client can “echo it back” on the next request).
   * DeepSeek’s spec is explicit that *for tool-calling continuation* the client must include prior `reasoning_content`, otherwise the request can be rejected (400) and/or the model cannot continue coherently.

3. **Cross-turn policy: clear vs preserve**

   * DeepSeek: for a *new user turn*, prior `reasoning_content` should be dropped (it may be ignored if sent, but it should not be treated as normal chat context).
   * GLM Thinking Mode: the API exposes an explicit **clear vs preserve** control (e.g., `clear_thinking`). If you preserve, you must return previous thinking blocks *unaltered and in order*, including in tool flows.

The key point: **“reasoning + tool_calls” must travel together**, and the system must support **round-tripping** the reasoning channel back into the next request.

---

## What `mlx-omni-server` does today

### A) It has a solid *parsing* foundation for tag-based thinking

* It implements a `ReasoningDecoder` that extracts `<think>…</think>` into a separate `"reasoning"` value and removes it from `"content"` in non-stream mode.
* It also supports streaming separation into `delta_content` vs `delta_reasoning`.

This is directionally compatible with the “two-channel” idea, albeit using the field name `reasoning` rather than `reasoning_content`.

### B) It supports “thinking knobs” *as template kwargs*, but not GLM’s clear/preserve semantics

The request-extra-param splitter explicitly recognizes a small set of template parameters, including:

* `enable_thinking` (Qwen3-style)
* `thinking` (Claude-style)
* `thinkingConfig` (Gemini-style)
* `reasoning_effort` (Grok-style)
* `reasoning` (misc)

However, there is **no explicit implementation of GLM’s `clear_thinking` preservation contract** (drop prior thinking vs preserve and replay), beyond whatever a particular HF chat template might do.

### C) Critical mismatch: reasoning is dropped whenever tools are enabled

This is the largest divergence from the DeepSeek/GLM “reasoning + tools” convention:

**Non-streaming**:

* The server extracts reasoning into `reasoning`, *but* if `request.tools` is present, it routes through tool decoding and **does not attach the extracted reasoning to the returned assistant message**.

**Streaming**:

* During streaming, reasoning deltas are emitted as separate chunks, but the **final tool_calls chunk** is constructed from `parse_buffer(...)`, which returns a `ChatMessage` containing only `content` and `tool_calls`—no reasoning field—so the final “tool_calls” message is missing the reasoning channel.
* The underlying tool parsing tokenizer also structurally omits reasoning in `decode(...)` / `parse_buffer(...)`.

This fails the most important “de facto” requirement: **the assistant tool-call step must carry the reasoning channel so it can be replayed**.

### D) Replayability into the *next request* is not guaranteed

Even if a client sends `reasoning` back, it is not clear the engine will feed it into the model context in a consistent way:

* `ChatTokenizer.encode(...)` forwards `ChatMessage.model_dump(exclude_none=True)` into `apply_chat_template(...)`. That means the `reasoning` key may exist in the message dict, but unless the model’s chat template explicitly references it, it will be ignored (i.e., the model never “sees” the replayed reasoning).

DeepSeek/GLM’s pattern assumes the reasoning channel is **semantically replayed** (whether via hidden state on their servers or via prompt/context on yours). Right now, `mlx-omni-server` only guarantees replay if the **content** itself still contains the thinking text (e.g., tags are preserved), not if thinking is returned separately.

---

## Alignment verdict

* **Two-channel reasoning output (no tools):** *Mostly aligned* (tag-based extraction + streaming deltas), but uses `reasoning` instead of `reasoning_content`.
* **Reasoning + tool_calls in the same assistant step:** *Not aligned* (reasoning dropped for tool paths; tool-call final chunk lacks reasoning).
* **Round-tripping reasoning back into the next request (DeepSeek/GLM replay):** *Partially aligned / template-dependent* (the engine does not robustly re-inject reasoning into prompt content).
* **GLM Thinking Mode clear/preserve contract:** *Not implemented as a first-class behavior* (no explicit `clear_thinking` handling).

Net: `mlx-omni-server` has the right primitives, but it does **not** currently satisfy the defining DeepSeek v3.2 / GLM Thinking Mode pattern: **“reasoning survives tool calling and can be replayed to continue thinking.”**

---

## Concrete changes to make `mlx-omni-server` conform (practical “engine guide” upgrades)

1. **Always attach reasoning alongside tool_calls**

   * In non-streaming `generate()`: after `message = self._chat_tokenizer.decode(...)`, set `message.reasoning = reasoning` (and/or `message.reasoning_content`) before returning. This directly fixes the current drop.
   * In streaming: maintain an accumulated `reasoning_so_far` (append every `delta_reasoning`), and when emitting the final `tool_calls` chunk (`final_message = parse_buffer(...)`), attach `final_message.reasoning = reasoning_so_far`.

2. **Support `reasoning_content` as a first-class alias**

   * Accept inbound `reasoning_content` and map it onto the internal `reasoning` field.
   * Optionally emit both fields for maximum ecosystem compatibility (OpenAI-compat stacks vary here).

3. **Make replay deterministic: re-inject reasoning into prompt context**

   * Before `apply_chat_template`, normalize assistant messages:

     * If a message has `reasoning(_content)` and the template is not known to consume it, synthesize `content := "<think>{reasoning}</think>\n{content or ''}"` (or the model’s configured thinking tag), then drop the separate field for the template call.
   * This makes the “echo reasoning back in request” pattern work regardless of the underlying template.

4. **Implement GLM-style clear/preserve explicitly**

   * Parse `thinking = {"type": "...", "clear_thinking": bool}` when present (GLM contract). If `clear_thinking` is true, strip historical reasoning before encoding; if false, preserve and re-inject it deterministically as above.

If you want, I can turn the above into a repo-specific patch plan (exact file/line targets, minimal diff strategy, and tests) using your current module boundaries and the existing `ReasoningDecoder` / `ToolParsingChatTokenizer` structure.
