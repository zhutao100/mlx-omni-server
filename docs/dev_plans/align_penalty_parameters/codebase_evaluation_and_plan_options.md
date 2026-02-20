## What `docs/dev_plans/align_penalty_parameters/context.md` is establishing

That context doc is essentially a taxonomy + “don’t mix semantics” warning:

* **OpenAI-style penalties** (`presence_penalty`, `frequency_penalty`) are **additive, count-based** adjustments to logits:

  * presence: apply once if a token has appeared at least once
  * frequency: scale by how many times a token has appeared
* **HF/vLLM/llama.cpp-style `repetition_penalty`** is a **multiplicative** (sign-aware) transform on logits, and is commonly **windowed** (e.g., last *N* tokens).
* **Best-practice boundary**: apply these **once**, at the **sampling/logits-processing** layer (engine-side), not in multiple layers; keep defaults “off”; and avoid silently accepting parameters that are not actually implemented.

That is exactly the set of pitfalls the current code falls into.

---

## Current code vs best practices: what it does today

### 1) The API surface advertises OpenAI-style penalties + logit bias

In Chat Completions request schema, the project explicitly models these fields:

```py
# src/mlx_omni_server/chat/schema.py
324 presence_penalty: Optional[float] = Field(0, ge=-2.0, le=2.0)
325 frequency_penalty: Optional[float] = Field(0, ge=-2.0, le=2.0)
326 logit_bias: Optional[Dict[str, float]] = None
```

So clients are *reasonably* going to assume these work.

Responses request schema also exposes presence/frequency penalties:

```py
# src/mlx_omni_server/responses/schema.py
162 presence_penalty: Optional[float] = None
163 frequency_penalty: Optional[float] = None
```

And `/responses` is converted into a ChatCompletionRequest (so it inherits whatever Chat does).

---

### 2) Text-only backend (`mlx-lm`) **misinterprets `presence_penalty` as `repetition_penalty`**

In the `mlx_lm` path, penalties are wired like this:

```py
# src/mlx_omni_server/chat/mlx_lm/mlx_lm_model.py
267 elif request.presence_penalty:
268     generate_kwargs["logits_processors"] = make_logits_processors(
269         repetition_penalty=request.presence_penalty
270     )
```

**Why this is a serious misalignment:**

* **Wrong semantic family** (additive “seen/count” vs multiplicative repetition penalty).
* **Negative values break**: OpenAI-style `presence_penalty` allows negative numbers; `mlx_lm.sample_utils.make_repetition_penalty()` rejects penalty `< 0`, so a valid OpenAI request can error out.
* **Values between 0 and 1 invert behavior**: `repetition_penalty < 1` *encourages* repetition in the HF-style transform; a small positive presence penalty like `0.2` becomes a strong *reward* for repeats (the opposite of what the caller intends).

Also:

* `frequency_penalty` is **ignored** in the text-only path.
* `logit_bias` is **ignored** in the text-only path.

---

### 3) Multimodal backend (`mlx-vlm`) passes `presence_penalty`/`frequency_penalty`… but **they are no-ops**

The VLM path constructs kwargs including presence/frequency:

```py
# src/mlx_omni_server/chat/mlx_vlm/mlx_vlm_model.py
923 "frequency_penalty": (...)
926 "presence_penalty": (...)
```

…but `mlx_vlm.generate.generate_step()` doesn’t implement those parameters. It *does* support:

```py
# mlx_vlm/generate.py
244 repetition_penalty: Optional[float] = None
245 repetition_context_size: Optional[int] = 20
247 logit_bias: Optional[Dict[int, float]] = None
255 logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None
```

So today:

* presence/frequency are **silently ignored** for VLM generations
* repetition/logit_bias could be supported, but the server doesn’t map to them

---

### 4) `/responses` inherits the same problems

`responses/adapter.py` converts the Responses request payload into a Chat request:

```py
# src/mlx_omni_server/responses/adapter.py
418 def response_request_to_chat_request(...):
...
458 return ChatCompletionRequest.model_validate(payload)
```

So:

* for text-only models: `presence_penalty` still becomes (incorrect) `repetition_penalty`
* for VLM: `presence_penalty`/`frequency_penalty` still no-op

---

### 5) Extra-params pass-through can crash on unknown generation kwargs (and blocks a clean `repetition_penalty` extension)

`ChatCompletionRequest` allows extra fields, and `mlx_lm_model._get_generation_params()` forwards unknown extras into `generate_kwargs`:

```py
# src/mlx_omni_server/chat/mlx_lm/mlx_lm_model.py
135 else:
136     generate_kwargs[key] = value
```

But `mlx_lm.generate.generate_step()` is **not** `**kwargs`-tolerant; it has a fixed signature. So if a client sends an extra field like `repetition_penalty` (as many OpenAI-compatible servers support), you’ll forward it into `stream_generate(..., **generate_kwargs)` → `generate_step(..., **kwargs)` and hit a `TypeError`.

This is directly at odds with the “align penalties across stacks” goal.

---

## Gap analysis (best practice → current state)

| Item                             | Best-practice expectation                   | Current state                                                                                             |
| -------------------------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `presence_penalty`               | additive “seen?” penalty, supports negative | mapped to multiplicative repetition penalty; negative can error; small positives can encourage repetition |
| `frequency_penalty`              | additive “count-based” penalty              | modeled in schema, **unused** in text-only, **ignored** in VLM                                            |
| `logit_bias`                     | additive per-token bias                     | modeled in schema, **unused** everywhere                                                                  |
| `repetition_penalty` (extension) | separate from OpenAI-style penalties        | not supported cleanly; extra param likely crashes in text-only path                                       |
| Consistency across backends      | same semantics for same request fields      | text-only misapplies; VLM no-ops                                                                          |

---

## Improvement plan options (pick based on compatibility vs scope)

### Option A — Minimal correctness fix (recommended baseline)

**Goal:** Make advertised OpenAI-style fields actually behave correctly, without redesigning the API surface.

1. **Stop mapping `presence_penalty` → `repetition_penalty`.**
2. Implement **presence/frequency penalties as a logits processor** in the server (shared utility), applied in:

   * `mlx-lm` path via `generate_kwargs["logits_processors"]`
   * `mlx-vlm` path via its `logits_processors` argument (it already supports this)
3. Implement **`logit_bias`** by converting OpenAI’s `{ "token_id": bias }` to `Dict[int, float]` and applying via:

   * `mlx_lm.sample_utils.make_logits_processors(logit_bias=...)` (LLM path)
   * `mlx_vlm.generate_step(logit_bias=...)` or also via logits processor (VLM path)

**Pros:** fixes real bugs immediately; aligns with context.md best practices; minimal API churn.

**Cons:** still no explicit `repetition_penalty` knob unless you add it as an extension.

---

### Option B — Full alignment + explicit extension knobs

**Goal:** Match the “surveyed ecosystem” pattern: OpenAI-style penalties *plus* distinct repetition penalty, consistently across engines.

Adds everything in Option A, plus:

1. Add explicit fields (or documented “extra params”) for:

   * `repetition_penalty` (default 1.0 = off)
   * `repetition_context_size` (default e.g. 20 or a server default)
2. Wire them correctly:

   * `mlx-lm`: translate `repetition_penalty` into `make_logits_processors(repetition_penalty=..., repetition_context_size=...)`
   * `mlx-vlm`: pass `repetition_penalty=` and `repetition_context_size=` directly (it supports both)
3. **Harden extra-param handling** in text-only path:

   * whitelist kwargs that `mlx_lm.generate_step()` actually accepts
   * drop/warn unknown keys instead of forwarding (prevents `TypeError` crashes)

**Pros:** best ecosystem compatibility; clean separation of penalty families; removes crash footgun.

**Cons:** slightly more surface area to document and test.

---

### Option C — Push penalties down into the underlying libraries

**Goal:** Make `mlx-lm`/`mlx-vlm` natively support OpenAI-style `presence_penalty`/`frequency_penalty` as first-class args.

* Add `presence_penalty`/`frequency_penalty` parameters to `mlx_lm.generate_step()` and implement them in `sample_utils` similarly to repetition penalty.
* Mirror that in `mlx_vlm.generate_step()`.

**Pros:** eliminates server duplication; other consumers benefit.

**Cons:** larger change surface; you’re now maintaining (or upstreaming) library semantics, plus version skew between forks.

---

## Concrete “next edits” checklist (for Options A/B)

### Shared utility (server-side)

Create something like `src/mlx_omni_server/chat/logits_penalties.py`:

* `normalize_logit_bias(logit_bias: dict[str, float], *, vocab_size: int) -> dict[int, float]`
* `make_presence_frequency_processor(presence: float, frequency: float, *, context_size: int | None = None, include_prompt: bool = True) -> Callable[[mx.array, mx.array], mx.array]`

Implementation detail that matters:

* For performance, prefer an **incremental Counter/deque** approach over recomputing counts from the full token history every step.
* Decide (and document) whether “text so far” includes **prompt tokens**. Whatever you choose, do it consistently across both backends.

### Text-only (`mlx_lm_model.py`)

* Build a single `logits_processors` list in a defined order (typical order):

  1. structured-output constraint processor (if enabled)
  2. logit bias
  3. repetition penalty (if supported via Option B)
  4. presence/frequency penalties
* Remove the current `presence_penalty → repetition_penalty` mapping.
* Add a whitelist filter for forwarded `generate_kwargs` so extra params can’t crash `generate_step()`.

### VLM (`mlx_vlm_model.py`)

* Stop passing `presence_penalty`/`frequency_penalty` as raw kwargs (they’re ignored).
* Instead pass them via `logits_processors` (your new presence/frequency processor).
* Add logit_bias support (and repetition_penalty support if doing Option B).

### Tests (currently missing in this repo)

Add targeted tests in `tests/` that validate:

* positive presence_penalty reduces probability of reusing a token (vs baseline)
* frequency_penalty scales with count
* negative presence/frequency do **not** error and behave as “encourage reuse”
* VLM and LLM backends both apply penalties (at least at the logits-processor level)

---

## Bottom line

The current implementation **does not** align with the dev-plan best practices:

* it **conflates** penalty families (presence → repetition)
* it **silently no-ops** penalties in the VLM backend
* it **models** parameters (`frequency_penalty`, `logit_bias`) without implementing them
* it leaves a **crash path** for common “extra” generation params, which blocks a clean `repetition_penalty` extension

If you want the fastest path to “correct + compatible,” do **Option A immediately**, then follow with **Option B** to support explicit `repetition_penalty` + safe kwarg filtering.
