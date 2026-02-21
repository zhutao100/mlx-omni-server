Below is an **implementation plan for Option B**: support **both penalty families** (OpenAI-style additive `presence_penalty`/`frequency_penalty` *and* HF-style multiplicative `repetition_penalty` + `repetition_context_size`) plus **`logit_bias`**, **consistently across mlx-lm and mlx-vlm**, and **harden extra-param handling** so unknown kwargs can’t crash generation.

---

## 0) Target behavior and invariants

### Semantics

* **presence_penalty / frequency_penalty** (OpenAI-style, additive-count):

  * presence: if token seen at least once in history, `logits[token] -= presence_penalty`
  * frequency: `logits[token] -= frequency_penalty * count[token]`
* **repetition_penalty** (HF/llama.cpp style, multiplicative/sign-aware):

  * apply to tokens in a configurable window of history (`repetition_context_size`)
* **logit_bias**:

  * additive: `logits[token] += bias`

### Defaults and disable meaning

* `presence_penalty = 0`, `frequency_penalty = 0` → disabled
* `repetition_penalty = 1.0` → disabled
* `repetition_context_size = 20` (only used if repetition_penalty != 1.0)
* `logit_bias = None` → disabled

### Application rule

* **Apply once** at the **sampling/logits-processing site** (via `logits_processors`) for both backends.

### Order (deterministic)

Recommended order in a single `logits_processors` list:

1. repetition penalty (if enabled)
2. presence/frequency penalties (if enabled)
3. logit_bias (if enabled)
4. structured constraints (`JsonLogitsProcessor`) **last** (if enabled)

(You can swap #2/#3; keep constraints last.)

---

## 1) API / schema changes

### 1.1 Chat schema (`src/mlx_omni_server/chat/schema.py`)

Add fields to `ChatCompletionRequest`:

* `repetition_penalty: Optional[float] = Field(1.0, gt=0.0)`

  * keep optional but default 1.0
* `repetition_context_size: Optional[int] = Field(20, ge=1)`

  * or plain `int` if you don’t want `None` at all

Update `get_extra_params()`’s `standard_fields` to include:

* `"repetition_penalty"`
* `"repetition_context_size"`

This prevents them from being treated as “extra” and forwarded into `generate_kwargs` (which is currently a crash vector).

### 1.2 Responses schema (`src/mlx_omni_server/responses/schema.py`)

Mirror the same fields in the Responses request object:

* `repetition_penalty`, `repetition_context_size`, and keep existing `presence_penalty`, `frequency_penalty`, `logit_bias` pass-through.

### 1.3 Responses adapter (`src/mlx_omni_server/responses/adapter.py`)

Ensure the conversion `response_request_to_chat_request()` passes these through unchanged so `/responses` behaves identically to `/chat/completions`.

---

## 2) Core implementation primitives (shared across backends)

Create a small module, e.g.:

* `src/mlx_omni_server/chat/logits_processors/penalties.py`

### 2.1 `normalize_logit_bias(...)`

Input: `Dict[str, float] | None` (OpenAI format: token-id-as-string → bias)
Output: `Dict[int, float] | None`

Rules:

* parse keys to int (drop + warn if not parseable)
* optionally clamp bias to a sane range (many servers use [-100, 100])
* if empty → `None`

### 2.2 `make_presence_frequency_processor(presence, frequency)`

Return a callable `(tokens: mx.array, logits: mx.array) -> mx.array` that:

* counts tokens in history (start with **full history**, including prompt)
* subtracts presence/frequency penalties per token
* supports **negative values** (encourage novelty vs repetition correctly)

Implementation note:

* simplest correct version can use a per-call `Counter(tokens.tolist())`.
* better version (recommended) is a **stateful processor** that incrementally updates counts based on newly appended tokens (safe because generation appends tokens monotonically).
  * Important: `mlx_lm.generate_step()` does **not** pass the full prompt token history into `logits_processors` (it starts from the final prompt token). To preserve “include prompt tokens” semantics (and to work correctly with prompt caching), seed the processor with the full prompt token list from the server wrapper, then incrementally update using the tokens observed during generation.

### 2.3 `build_logits_processors(request, tokenizer, *, prompt_tokens: list[int] | None = None)`

A single builder used by both mlx-lm and mlx-vlm server wrappers:

* Convert `logit_bias`
* Decide `rep = None if repetition_penalty is None or repetition_penalty == 1.0 else repetition_penalty`
* Create list (deterministic order):

  1. repetition penalty (if enabled)
     * use `mlx_lm.sample_utils.make_logits_processors(repetition_penalty=rep, repetition_context_size=...)`
  2. presence/frequency processor (if enabled)
     * seed with `prompt_tokens` when available
  3. `logit_bias` (if enabled)
     * use `mlx_lm.sample_utils.make_logits_processors(logit_bias=...)`
  4. structured constraints (`JsonLogitsProcessor`) **last** (if enabled)

Return `processors: list[callable] | None`.

---

## 3) Integrate into mlx-lm path (text-only)

File: `src/mlx_omni_server/chat/mlx_lm/mlx_lm_model.py`

### 3.1 Replace the incorrect mapping

Remove the current behavior that treats `presence_penalty` as repetition penalty.

Instead, in `_prepare_generation()`:

* always compute processors via `build_logits_processors(...)`
* set `generate_kwargs["logits_processors"] = processors` when non-empty
* ensure processors **compose** with JSON schema constraints (do not overwrite)

### 3.2 Harden extra-param handling (crash prevention)

In `_get_generation_params()`:

* replace the current `else: generate_kwargs[key] = value` with a **whitelist** approach.

Whitelist should match `mlx_lm.generate.generate_step()` and `speculative_generate_step()` kwargs union, e.g.:

* `max_kv_size`, `prefill_step_size`, `kv_bits`, `kv_group_size`, `quantized_kv_start`,
* `num_draft_tokens` (speculative)
* (and anything else you *explicitly* support)

Unknown keys:

* **drop + log warning** (“unsupported generation parameter … dropping”)

This prevents `stream_generate(..., **kwargs)` from passing unknowns into `generate_step()` and raising `TypeError`.

---

## 4) Integrate into mlx-vlm path (multimodal)

File: `src/mlx_omni_server/chat/mlx_vlm/mlx_vlm_model.py`

### 4.1 Stop passing no-op presence/frequency kwargs

Right now you construct `generate_kwargs` with `presence_penalty`/`frequency_penalty`, but mlx-vlm’s `generate_step` doesn’t implement them.

Instead:

* compute processors using the shared `build_logits_processors(...)`
* pass them as `logits_processors=processors`
* **do not** also pass `repetition_penalty` / `logit_bias` via mlx-vlm args if you already included them in processors (avoid double application). Easiest is:

  * call `mlx_vlm.generate_step(..., repetition_penalty=None, logit_bias=None, logits_processors=processors, ...)`

This gives identical behavior across backends.

---

## 5) Compatibility strategy (important because semantics change)

Your current behavior is effectively:

* `presence_penalty` → repetition penalty (wrong)
* `frequency_penalty` → ignored
* `logit_bias` → ignored

Option B fixes that, which may surprise anyone who relied on the buggy behavior.

Recommended approach:

1. **Correct behavior by default** (presence/frequency additive; repetition via new field).
2. Add a short **migration note**:

   * “If you previously used `presence_penalty` to reduce repetition, switch to `repetition_penalty`.”
3. Optional (if you want a softer rollout): a server config flag such as:

   * `MLX_OMNI_LEGACY_PRESENCE_AS_REPETITION=1`
   * when enabled, keep old mapping but emit a deprecation warning
   * plan to remove after 1–2 releases

---

## 6) Tests

Add a small test suite that doesn’t require full model inference.

### 6.1 Unit tests (fast)

* `normalize_logit_bias()` parsing and dropping invalid keys
* presence/frequency processor math on toy logits:

  * create a small vocab logits tensor, apply processor, verify expected deltas
* repetition penalty wiring:

  * verify that `repetition_penalty==1.0` produces no repetition processor
  * verify non-1.0 produces one

If MLX isn’t importable in CI, structure tests to:

* run “reference math” in pure Python/Numpy and skip MLX-specific parts, OR
* mark MLX-dependent tests as optional.

### 6.2 Integration tests (smoke)

* start server with a tiny model (if repo has a fixture) and validate:

  * penalties affect token distribution (coarse: output changes deterministically under fixed seed)

---

## 7) Docs and observability

### 7.1 Update docs

* README / API docs: document new fields:

  * `repetition_penalty`, `repetition_context_size`
* Mention “penalty families” and defaults (from context.md)
* Add examples for:

  * “reduce repetition” → repetition_penalty
  * “increase novelty” → presence_penalty

### 7.2 Logging

* when dropping unknown extra params: warn once per request with the key list
* when `repetition_penalty` in (0, 1): optionally warn (“encourages repetition”)

---

## 8) Deliverables checklist

* [x] schema: add repetition fields + standard_fields update
* [x] shared module: bias normalization + presence/frequency processor + builder
* [x] mlx-lm: replace mapping; compose processors; whitelist extra params
* [x] mlx-vlm: route via processors; remove no-op kwargs
* [x] responses: pass-through repetition fields
* [x] tests: unit (smoke test deferred; would require a tiny model fixture)
* [x] docs: new params + migration note
