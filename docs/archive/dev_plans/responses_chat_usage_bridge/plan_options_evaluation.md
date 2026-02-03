## What the upstream docs actually give you (and what you can safely map)

**OpenAI (Chat Completions, streaming chunk `usage`)**

* `usage.prompt_tokens`, `usage.completion_tokens`, `usage.total_tokens`
* `usage.prompt_tokens_details.cached_tokens`
* `usage.completion_tokens_details.reasoning_tokens` ([OpenAI Platform][1])
  Also: if `stream_options.include_usage` is set, the *last* chunk’s `choices` can be empty. ([OpenAI Platform][1])

**DeepSeek (Chat Completions)**

* Same base trio: `prompt_tokens`, `completion_tokens`, `total_tokens`
* Cache is **split**: `prompt_cache_hit_tokens` + `prompt_cache_miss_tokens` (and they state `prompt_tokens = hit + miss`)
* Reasoning is in `completion_tokens_details.reasoning_tokens` ([DeepSeek API Docs][2])
  Streaming semantics are explicit: with `stream_options.include_usage`, an extra chunk is sent before `[DONE]` where `choices` is **always empty** and `usage` is populated. ([DeepSeek API Docs][2])

**GLM/Z.AI (Chat Completions)**

* `usage.prompt_tokens`, `usage.completion_tokens`, `usage.total_tokens`
* `usage.prompt_tokens_details.cached_tokens`
* No `completion_tokens_details` in the usage schema shown. ([Z.AI][3])

### Minimal, robust mapping into **Responses usage**

For a Responses-shaped downstream, your “normalized” mapping can be:

* `usage.input_tokens  = upstream.prompt_tokens`
* `usage.output_tokens = upstream.completion_tokens`
* `usage.total_tokens  = upstream.total_tokens`
* `usage.input_tokens_details.cached_tokens =`

  * `upstream.prompt_tokens_details.cached_tokens` (OpenAI/GLM-style), else
  * `upstream.prompt_cache_hit_tokens` (DeepSeek-style) ([DeepSeek API Docs][2])
* `usage.output_tokens_details.reasoning_tokens =`

  * `upstream.completion_tokens_details.reasoning_tokens` when present (OpenAI/DeepSeek), else 0/omitted ([OpenAI Platform][1])

---

## What `mlx-omni-server` does now

### Current mapping behavior

In `src/mlx_omni_server/responses/adapter.py`, `_build_usage_dict()` produces a Responses `usage` object with:

* `input_tokens/output_tokens/total_tokens` from upstream usage if present, else **0**
* `input_tokens_details.cached_tokens` from `prompt_tokens_details.cached_tokens` if present, else **0**
* `output_tokens_details.reasoning_tokens` from upstream `completion_tokens_details.reasoning_tokens` when present, else **0**
  (clamped to `0 ≤ reasoning_tokens ≤ output_tokens`)

### Streaming behavior: usage is requested upstream

The streaming Responses path emits `response.completed` with `include_usage=True` (`on_done()` → `_build_response_dict(... include_usage=True ...)`). The adapter learns token counts from the Chat Completions “usage chunk” (`chunk.usage` stored in `self._usage`).

`response_request_to_chat_request()` now forces `stream_options.include_usage=true` for streaming so the upstream can emit the final usage chunk when supported.

This matches the documented semantics: you only get a populated usage chunk if `stream_options.include_usage` is enabled. ([DeepSeek API Docs][2])

---

## Best-effort plan options (and how they fit this repo)

### Option A — **Authoritative totals (recommended baseline)**

**Goal:** always return correct `usage` totals for Responses, both non-streaming and streaming.

**Plan**

1. In `response_request_to_chat_request()`:

   * if `response_request.stream == True`, ensure upstream payload has `stream_options: { include_usage: true }` (merge if already present).
2. Keep mapping:

   * `prompt_tokens → input_tokens`
   * `completion_tokens → output_tokens`
   * `prompt_tokens_details.cached_tokens → input_tokens_details.cached_tokens`
3. Keep `output_tokens_details.reasoning_tokens = 0` for now.

**Why it fits `mlx-omni-server`**

* Your local MLX streaming implementation already emits a final “usage” chunk when `include_usage` is true; it’s just not being requested from the Responses path.
* Minimal code change, and it stops lying with zeros.

**Nice-to-have alignment:** in the MLX chat streamer, the “usage chunk” currently has a dummy `choices=[{delta:{role:assistant}}]`. OpenAI/DeepSeek both document “choices can be empty” (and DeepSeek says it *will always be empty* for that chunk). ([DeepSeek API Docs][2])
Your adapter tolerates it, but if you’re chasing strict compatibility, consider emitting `choices=[]` for that usage chunk.

**Status:** implemented (the Responses→Chat bridge requests `stream_options.include_usage=true` for streaming).

---

### Option B — **Best-effort reasoning token accounting**

**Goal:** fill `usage.output_tokens_details.reasoning_tokens` in Responses in a way that’s directionally consistent with OpenAI/DeepSeek’s `completion_tokens_details.reasoning_tokens`. ([OpenAI Platform][1])

**Plan**

1. Extend `ChatCompletionUsage` in `src/mlx_omni_server/chat/schema.py` to optionally include:

   * `completion_tokens_details: { reasoning_tokens?: int, ... }`
   * (optionally) DeepSeek cache split fields, if you ever proxy to DeepSeek.
2. In the MLX generator (`mlx_lm_model.py`), compute a **best-effort** `reasoning_tokens`:

   * simplest: tokenize the final extracted reasoning string (or accumulate per reasoning delta) using the model tokenizer wrapper.
   * ensure `reasoning_tokens <= completion_tokens` (cap if needed).
3. Map it in `_build_usage_dict()`:

   * `output_tokens_details.reasoning_tokens = upstream.completion_tokens_details.reasoning_tokens`

**Fit to this repo**

* You already have a `ReasoningDecoder` and you already track reasoning text in the Responses adapter; the missing piece is a token-counting path.
* This is optional; don’t block correctness of totals on this.

**Status:** implemented (local MLX chat models populate `completion_tokens_details.reasoning_tokens`, and Responses usage maps it through).

---

### Option C — **Provider-superset normalization + raw passthrough**

**Goal:** be resilient if the upstream is OpenAI *or* DeepSeek *or* GLM, without losing provider-specific fields.

**Plan**

1. Parse upstream `usage` as a tolerant dict (or Pydantic with `extra="allow"`), preserving unknown keys.
2. Produce **normalized Responses usage** using the precedence rules above:

   * cached tokens from `prompt_tokens_details.cached_tokens` else `prompt_cache_hit_tokens`
   * reasoning tokens from `completion_tokens_details.reasoning_tokens` if present
3. Optionally stash the raw upstream usage under something like `response.metadata.upstream_usage` for debugging/analytics.

**Fit to this repo**

* Today you’re “local MLX upstream”, so you don’t *need* this—unless you’re explicitly trying to support running the proxy against remote DeepSeek/GLM as an upstream.

---

## Concrete recommendation for `mlx-omni-server`

1. **Option A (implemented)**: force `stream_options.include_usage=true` for upstream Chat Completions whenever serving Responses streaming. ([DeepSeek API Docs][2])
2. **Option B (implemented)**: populate and map `output_tokens_details.reasoning_tokens` when the upstream provides `completion_tokens_details.reasoning_tokens` (local MLX computes best-effort tokens).
3. If your roadmap includes “Responses proxy in front of DeepSeek/GLM”, jump straight to Option C (or at least add the DeepSeek cache-split fallback) so cached tokens don’t silently disappear. ([DeepSeek API Docs][2])

[1]: https://platform.openai.com/docs/api-reference/chat-streaming "Streaming | OpenAI API Reference"
[2]: https://api-docs.deepseek.com/api/create-chat-completion "Create Chat Completion | DeepSeek API Docs"
[3]: https://docs.z.ai/api-reference/llm/chat-completion "Chat Completion - Overview - Z.AI DEVELOPER DOCUMENT"
