Below are practical plan options for how your **Responses-shaped downstream** proxy should populate `usage` (and specifically `usage.output_tokens_details`) when the **upstream is Chat Completions** (OpenAI / DeepSeek / GLM), including **SSE streaming**.

## What the docs imply you can rely on

### Downstream (OpenAI Responses) expectations

* In the **Responses streaming event model**, `usage` is typically **`null` during in-progress events** and becomes available on **`response.completed`**. ([OpenAI Platform][1])
* The `response.completed` example includes `usage.output_tokens_details.reasoning_tokens`. ([OpenAI Platform][1])
* The Reasoning guide also shows `usage.input_tokens_details.cached_tokens` alongside `output_tokens_details.reasoning_tokens`. ([OpenAI Platform][2])

### Upstream (Chat Completions) shapes differ

* **OpenAI Chat Completions**: `usage.prompt_tokens`, `usage.completion_tokens`, `usage.total_tokens`, plus `prompt_tokens_details.cached_tokens` and `completion_tokens_details.reasoning_tokens` (and more). ([OpenAI Platform][3])
* **DeepSeek**: `prompt_tokens`, `completion_tokens`, `total_tokens`, plus `prompt_cache_hit_tokens`, `prompt_cache_miss_tokens`, and `completion_tokens_details.reasoning_tokens`. ([DeepSeek API Docs][4])
* **GLM (Z.AI)**: `prompt_tokens`, `completion_tokens`, `total_tokens`, plus `prompt_tokens_details.cached_tokens` (and `reasoning_content` as a field, but no token breakdown documented there). ([Z.AI][5])

### Streaming usage support (important for SSE proxies)

* **OpenAI** streaming: last chunk can be empty when `stream_options: {"include_usage": true}` is used. ([OpenAI Platform][6])
* **DeepSeek** streaming: documents `stream_options.include_usage`, with a final “usage chunk” before `[DONE]`. ([DeepSeek API Docs][4])

---

## Canonical mapping you almost always want

When upstream provides a Chat Completions `usage`:

**Responses `usage`**

* `input_tokens`  ← upstream `prompt_tokens`
* `output_tokens` ← upstream `completion_tokens`
* `total_tokens`  ← upstream `total_tokens`

**Responses `usage.input_tokens_details.cached_tokens`**

* OpenAI/GLM: upstream `prompt_tokens_details.cached_tokens` ([OpenAI Platform][3])
* DeepSeek: upstream `prompt_cache_hit_tokens` (since it is explicitly “hits the context cache”) ([DeepSeek API Docs][4])

**Responses `usage.output_tokens_details.reasoning_tokens`**

* OpenAI/DeepSeek: upstream `completion_tokens_details.reasoning_tokens` ([OpenAI Platform][3])
* GLM: not documented as available → treat as “unknown” unless you implement an estimation fallback. ([Z.AI][5])

**Invariants to enforce**

* If you emit `output_tokens_details.reasoning_tokens`, ensure `0 ≤ reasoning_tokens ≤ output_tokens` (clamp if needed).
* Prefer keeping totals internally consistent: `input_tokens + output_tokens == total_tokens` (when upstream gives totals, trust upstream totals).

---

## Plan options

### Option 1 — Upstream-authoritative usage only (recommended default)

**Goal:** Never guess. Populate `usage` only when the upstream provides it.

**Non-streaming upstream**

* Map upstream `usage` → Responses `usage` using the canonical mapping above.

**Streaming upstream**

* Always request streaming usage **when supported**:

  * OpenAI: set `stream_options: {"include_usage": true}` ([OpenAI Platform][6])
  * DeepSeek: set `stream_options: {"include_usage": true}` ([DeepSeek API Docs][4])
* Capture the “final usage chunk” (choices empty, usage populated) and map it.
* Emit Responses `usage` only on `response.completed` (keep `usage: null` earlier), matching the Responses streaming examples. ([OpenAI Platform][1])

**Pros**

* Correct w.r.t. billing semantics and provider tokenization.
* Simplest to reason about and test.

**Cons**

* For providers/endpoints that don’t return usage (or where you can’t enable include-usage), downstream `usage` may be `null`.

---

### Option 2 — Compatibility mode: always emit a full `usage` object, but “unknown” becomes 0

**Goal:** Maximize downstream client compatibility that expects keys to exist.

* If upstream usage is present: same as Option 1.
* If upstream usage is absent:

  * Emit `usage` with zeros:

    * `input_tokens: 0`, `output_tokens: 0`, `total_tokens: 0`
    * `output_tokens_details: { reasoning_tokens: 0 }`
    * optionally `input_tokens_details: { cached_tokens: 0 }` (consistent with the Reasoning guide example). ([OpenAI Platform][2])

**Pros**

* Predictable JSON shape for clients.
* Avoids `null` checks.

**Cons**

* “0” can be misinterpreted as real metering data.
* If you use this, it’s best paired with an explicit policy flag (e.g., `proxy_usage_mode=compat_zero`) so operators know they’re not getting true accounting.

---

### Option 3 — Hybrid: upstream totals + best-effort breakdown for `reasoning_tokens`

**Goal:** Preserve authoritative totals while giving a *useful* `reasoning_tokens` value when upstream doesn’t provide it.

* If upstream provides `completion_tokens_details.reasoning_tokens`: pass through (same as Option 1).
* Else, derive `reasoning_tokens` from content you already have:

  * If upstream provides a distinct `reasoning_content` field (DeepSeek, GLM), tokenize that string with a configured tokenizer and treat that as `reasoning_tokens`.
  * Keep `output_tokens` and `total_tokens` from upstream (authoritative); clamp derived `reasoning_tokens ≤ output_tokens`.

**Pros**

* Improves reasoning-UX telemetry even when upstream doesn’t provide token breakdown.
* Keeps billing-relevant totals authoritative.

**Cons**

* Requires tokenizer plumbing and model-specific tokenization accuracy; mismatch risk can be significant.

---

### Option 4 — Full local estimation fallback (only if you truly need non-null usage everywhere)

**Goal:** Provide usage even when upstream never returns it.

* Locally tokenize:

  * **Input**: the serialized upstream prompt/messages
  * **Output**: emitted assistant text + tool call arguments + (optionally) reasoning content
* Emit Responses `usage` using computed counts.

**Pros**

* Always produces numbers.

**Cons**

* Hard to make “correct” across vendors/models because chat formatting and tokenization are provider-specific.
* High engineering cost; easy to introduce subtle inaccuracies.

---

## Streaming-specific implementation notes (applies to Options 1–4)

1. **Accumulate usage out-of-band during upstream SSE**

   * For OpenAI/DeepSeek include-usage mode, treat the final “usage chunk” specially (choices empty). ([OpenAI Platform][6])

2. **Emit usage only when the downstream Response is terminal**

   * Conform to the Responses streaming examples: `usage: null` in-progress, populated on `response.completed`. ([OpenAI Platform][1])

3. **Provider capability gating**

   * Don’t blindly send `stream_options` to every upstream—some OpenAI-compatible servers reject unknown fields.
   * Maintain a provider capability matrix (OpenAI: yes; DeepSeek: yes; GLM: not documented on that page) and/or a configurable allowlist.

---

## Practical recommendation

Start with **Option 1** (authoritative mapping) and add **Option 2 as an opt-in “shape stability” mode** for clients that cannot tolerate `usage: null`. Keep Options 3–4 behind flags until you have a concrete product requirement for non-null usage and are willing to own tokenizer accuracy across upstreams.

[1]: https://platform.openai.com/docs/api-reference/responses-streaming "Streaming events | OpenAI API Reference"
[2]: https://platform.openai.com/docs/guides/reasoning "Reasoning models | OpenAI API"
[3]: https://platform.openai.com/docs/api-reference/chat/object "Chat Completions | OpenAI API Reference"
[4]: https://api-docs.deepseek.com/api/create-chat-completion "Create Chat Completion | DeepSeek API Docs"
[5]: https://docs.z.ai/api-reference/llm/chat-completion "Chat Completion - Overview - Z.AI DEVELOPER DOCUMENT"
[6]: https://platform.openai.com/docs/api-reference/chat-streaming "Streaming | OpenAI API Reference"
