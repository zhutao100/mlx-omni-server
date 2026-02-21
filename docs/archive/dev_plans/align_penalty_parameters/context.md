## Executive summary

* There are **two distinct “penalty families”** you must not conflate:

  1. **Count-based additive logit penalties** (OpenAI-style **presence/frequency penalty**) that subtract a term based on “seen?” and/or “how many times seen?” (common in hosted APIs and OpenAI-compatible servers). ([OpenAI Developers][1])
  2. **Repetition penalty (multiplicative / sign-aware)** popularized by HF/llama.cpp-style decoders, which rescales logits for tokens that appear in the history (often multiplying negative logits and dividing positive logits). ([GitHub][2])

* Defaults that are “safe” across many stacks:

  * `presence_penalty = 0`, `frequency_penalty = 0` (disabled) in OpenAI-style APIs. ([OpenAI Developers][1])
  * `repetition_penalty = 1.0` (disabled) in HF-style repetition-penalty implementations. ([GitHub][2])

* Operationally: **apply penalties exactly once** at the final sampling site (the engine), and treat UI/SDK/proxies as **pass-through** to avoid “double punishment” in layered systems.

Recommended “starter presets” (pragmatic, conservative):

* **Chat / assistant:** `presence=0.2`, `frequency=0.1`, `repetition_penalty=1.05`
* **Code completion / structured JSON:** `presence=0`, `frequency=0`, `repetition_penalty=1.0`
* **Summarization / RAG:** `presence=0.1`, `frequency=0.2`, `repetition_penalty=1.05` (watch for citation/quote repetition)
* **Creative writing:** `presence=0.4`, `frequency=0.2`, `repetition_penalty=1.1` (expect more drift if too high)

---

## Cross-engine mapping table (names, defaults, semantics)

> **Legend**:
> **Additive-count** = subtract a term based on history (“seen?” / count).
> **Multiplicative** = rescale logits for tokens in history (HF repetition penalty).

| Engine / API                        | Exposed params (as named)                                                                                                                          | Defaults / disabled meaning                                                     | Semantics family                         | Notes / gotchas                                                                                                                                                           |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OpenAI Chat Completions             | `presence_penalty`, `frequency_penalty`, `logit_bias` ([OpenAI Developers][1])                                                                     | penalties default `0` (off); `logit_bias` absent = off ([OpenAI Developers][1]) | Additive-count (+ per-token bias)        | Docs describe purpose; exact math/order is not standardized publicly (treat as “count-based additive” in ports). ([OpenAI Developers][1])                                 |
| OpenAI Completions                  | `presence_penalty`, `frequency_penalty`, `logit_bias` ([OpenAI Developers][3])                                                                     | penalties default `0` (off) ([OpenAI Developers][3])                            | Additive-count                           | Useful as a “reference semantics” many servers emulate. ([OpenAI Developers][3])                                                                                          |
| OpenAI Responses                    | (no presence/frequency penalties shown); has sampling like `temperature`, `top_p` ([OpenAI Platform][4])                                           | N/A                                                                             | N/A                                      | If you’re migrating Chat→Responses, don’t assume penalties exist—plan for alternates (logit bias, custom logits processor in engine, etc.). ([OpenAI Platform][4])        |
| vLLM                                | `presence_penalty`, `frequency_penalty`, `repetition_penalty`, plus `min_p/top_p/top_k/temp` ([vLLM][5])                                           | `presence=0`, `frequency=0`, `repetition=1.0` disable ([vLLM][5])               | Both (Additive-count + Multiplicative)   | vLLM explicitly tracks OpenAI-style parameters and exposes a separate `repetition_penalty`. ([vLLM][5])                                                                   |
| Hugging Face Transformers (library) | `repetition_penalty`, `no_repeat_ngram_size`, `bad_words_ids`, `sequence_bias` ([GitHub][2])                                                       | `repetition_penalty=1.0` disable ([GitHub][2])                                  | Multiplicative + hard constraints/bias   | Repetition penalty is **sign-aware** (multiply vs divide) and typically includes prompt tokens unless excluded. ([GitHub][2])                                             |
| llama.cpp server                    | `repeat_penalty`, `repeat_last_n`, `presence_penalty`, `frequency_penalty`, `logit_bias` (+ others like `penalize_nl`, “dry” params) ([GitHub][6]) | Defaults vary by build/version (see server README)                              | Both                                     | **Windowing** (`repeat_last_n`) is a major semantic difference vs “whole history” penalties. ([GitHub][6])                                                                |
| Google Gemini API                   | `presencePenalty`, `frequencyPenalty` ([Google AI for Developers][7])                                                                              | (not always shown in snippet; treat as optional/off when unset)                 | Additive-count                           | Google explicitly describes presence as **binary** (seen or not) vs frequency as increasing with reuse. ([Google AI for Developers][7])                                   |
| Cohere                              | `presence_penalty`, `frequency_penalty` ([Cohere Documentation][8])                                                                                | (ranges/defaults depend on endpoint; examples show optional)                    | Additive-count                           | Cohere docs: frequency scales with count; presence applies once token appeared; includes prompt/preceding text. ([Cohere Documentation][8])                               |
| Mistral AI                          | `presence_penalty` (+ sampling docs mention frequency penalty) ([Mistral AI][9])                                                                   | `presence_penalty` default `0` ([Mistral AI][9])                                | Additive-count                           | Verify per endpoint: docs show presence penalty prominently; frequency penalty is documented in sampling guidance. ([Mistral AI][10])                                     |
| NVIDIA TensorRT-LLM                 | `repetition_penalty` in runtime / backend config ([NVIDIA GitHub][11])                                                                             | runtime shows `repetition_penalty` default `1.0` ([NVIDIA GitHub][11])          | Multiplicative (at least for repetition) | If you need OpenAI-style presence/frequency, confirm support in the specific serving layer; the most clearly documented knob is repetition penalty. ([NVIDIA GitHub][11]) |

**Explicit unknowns / version sensitivity:** Some engines and chat UIs (e.g., SGLang/Aphrodite/Text-generation-webui/SillyTavern) often provide *either* OpenAI-style penalties *or* repetition-penalty knobs depending on which backend they target, but I’m not asserting exact names/defaults here without primary citations in this run.

---

## Definitions & mechanics (implementation-level)

### A. `repetition_penalty` (multiplicative, “HF/llama.cpp style”)

**Core idea:** for tokens that have appeared in a defined history set/window, rescale their logits before sampling.

A widely used implementation (Hugging Face) is **sign-aware**: for a penalized token’s logit `l` and penalty `p>0`:

* if `l < 0`: `l := l * p`
* else: `l := l / p` ([GitHub][2])

This matters because multiplying a negative logit makes it **more negative** (lower probability), while dividing a positive logit makes it **smaller** (also lower probability). Same end-goal, stable across signs. ([GitHub][2])

**Windowing:**

* HF’s processor by default considers “tokens include the prompt” unless you explicitly ignore prompt length. ([GitHub][2])
* llama.cpp exposes an explicit sliding window via `repeat_last_n`, which changes semantics materially (recent repetition vs global repetition). ([GitHub][6])

**Pseudo-code (sign-aware repetition penalty)**

```text
history = tokens_to_penalize(prompt, generated, window=repeat_last_n?)
for t in history_unique_tokens:
    l = logits[t]
    logits[t] = (l < 0) ? (l * p) : (l / p)
```

**Edge cases**

* **Tokenization artifacts:** “word repetition” might be multiple token IDs (`" the"` vs `"the"`, different whitespace/prefix tokens). Penalty is applied to token IDs, not words.
* **EOS interaction:** penalizing EOS can cause run-on outputs; some stacks exclude special tokens, others don’t (must verify per engine).
* **Very small `p` (<1):** you are *rewarding* repetition (HF explicitly supports 0–1 to encourage repetition). ([GitHub][2])

---

### B. `presence_penalty` (additive-count, “seen?”)

**Core idea:** subtract a fixed amount if a token has appeared at least once.

Many vendor docs describe it as *binary*—once a token has appeared, the penalty applies (independent of how many times). ([Cohere Documentation][8])

A canonical implementation looks like:

```text
if count[token] > 0:
    logits[token] -= presence_penalty
```

**Windowing:** often “generated so far,” sometimes includes prompt/preceding text (Cohere explicitly frames it over preceding text including prompt). ([Cohere Documentation][8])
(Some engines also apply this only to the last N tokens, similar to llama.cpp’s repeat windowing, but confirm per engine.)

**Typical tuning intuition**

* Positive values → encourage novelty / topic exploration.
* Negative values → encourage reuse of the same tokens (can help with strict phrasing, but increases loops).

---

### C. `frequency_penalty` (additive-count, “how many times?”)

**Core idea:** subtract proportionally to the count of prior occurrences.

Vendor docs describe it as scaling with frequency; e.g., a token seen 10 times is penalized more than a token seen once. ([Cohere Documentation][8])

Canonical implementation:

```text
logits[token] -= frequency_penalty * count[token]
```

**Operational nuance:** Frequency penalties without windowing can become very strong in long generations. That’s why some engines/front-ends prefer a recent-window count (or cap the count), but you must treat that as an engine-specific variant unless documented.

---

## How penalties compose with sampling controls (ordering matters)

A common decode pipeline:

```text
logits = model(x, kv_cache)

# 1) hard constraints / bias
logits += logit_bias[token]                 (OpenAI-style)        :contentReference[oaicite:48]{index=48}
logits = ban_sequences(logits, bad_words)   (HF bad_words_ids)    :contentReference[oaicite:49]{index=49}
logits = no_repeat_ngram(logits, n)         (HF no_repeat_ngram)  :contentReference[oaicite:50]{index=50}

# 2) penalties
logits = repetition_penalty(logits, p)      (multiplicative)      :contentReference[oaicite:51]{index=51}
logits = presence/frequency(logits, ...)    (additive-count)

# 3) sampling warpers
logits /= temperature                       (temp)               :contentReference[oaicite:52]{index=52}
logits = top_k/top_p/min_p_filter(logits)   (truncate set)        :contentReference[oaicite:53]{index=53}

# 4) select token
token = argmax(logits) OR sample(softmax(logits))
```

**Key interaction notes**

* Penalties impact **greedy** decoding too (they change argmax). So “I’m greedy so penalties don’t matter” is false.
* `temperature=0` (or effectively ~0) reduces randomness, but penalties can still steer the deterministic choice.
* `top_p/top_k/min_p` can mask the practical effect of penalties if the token is filtered out anyway, but penalties can *also* cause a token to drop out of the candidate set earlier.

---

## Close equivalents you should map explicitly

### 1) `logit_bias` / “token bias” / “sequence bias”

* OpenAI-style `logit_bias` is an **additive per-token** adjustment. ([OpenAI Developers][1])
* HF’s `sequence_bias` applies additive bias to the last token of a sequence when that sequence is about to be completed (more powerful than single-token bias). ([GitHub][2])

### 2) Bans & constraints

* **Bad words / banned tokens**: HF `bad_words_ids` becomes a `-inf` bias for forbidden sequences. ([GitHub][2])
* **No-repeat ngram**: forbids repeating token n-grams via setting banned next tokens to `-inf`. ([GitHub][2])
* llama.cpp exposes `logit_bias` and other repetition controls in its server API. ([GitHub][6])

### 3) “repeat_last_n” (window)

* llama.cpp’s `repeat_last_n` means “counts/penalties consider only last N tokens,” which is not equivalent to OpenAI-style “entire generated text so far.” ([GitHub][6])

---

## Best practices & tuning guidance (actionable)

### Parameter ranges (typical starting zones)

* **repetition_penalty**

  * 1.00 = off; **1.03–1.12** common for general chat; **1.15+** can cause avoidance of necessary tokens and odd synonyms. HF paper guidance mentions ~1.2 as a balance point. ([GitHub][2])
* **presence_penalty**

  * **0.0–0.6** useful; above ~0.8 you may see topic thrash / forced novelty (esp. smaller models).
* **frequency_penalty**

  * **0.0–0.6**; higher can suppress legitimate repeated function words in long outputs unless windowed.

### Symptom→knob mapping

* **Loops / repeated phrases / “stuck” continuations**

  * First try: `repetition_penalty` 1.05–1.10 (or small `frequency_penalty` 0.1–0.3).
* **Verbose restatements / repeating the question**

  * Raise `frequency_penalty` (0.2–0.5) more than presence.
* **Blandness / low lexical diversity**

  * Raise `presence_penalty` (0.2–0.5), keep frequency moderate.
* **RAG citation repetition / repeated source lines**

  * Keep penalties low; prefer **hard constraints** (stop sequences / structured templates) and retrieval-side deduping.

### Use-case presets

* **Tool-calling / JSON / function arguments**

  * Penalties OFF (`presence=0`, `frequency=0`, `repetition_penalty=1.0`). Penalties can break schema by discouraging required repeated tokens like braces/quotes/keys.
* **Chat with long context**

  * Prefer **windowed** repetition controls when available (e.g., llama.cpp `repeat_last_n`) to avoid “global” suppression of important repeated terms.
* **Summarization**

  * Mild repetition penalty + mild frequency penalty; watch for proper nouns (penalties can cause name mutation).
* **Creative**

  * Presence higher than frequency; repetition penalty modest to avoid bizarre thesaurus behavior.

### Failure modes when too high

* **Excessive avoidance**: model refuses to reuse key nouns, producing awkward paraphrases.
* **Semantic drift**: forced novelty moves the topic.
* **Premature refusal to close**: if EOS is penalized or indirectly suppressed, outputs may run long.

---

## Server vs client responsibilities (and how to avoid double-penalty bugs)

### What belongs where

**Server/engine (authoritative):**

* The actual logit adjustment + sampling implementation.
* Default values (so behavior is consistent across clients).
* Observability: record the **effective** decode params used.

**Client/UI/SDK/proxy (policy & UX):**

* Surface controls to the user (sliders, presets).
* Validate ranges / normalize naming.
* Pass-through parameters without applying any local penalty logic.

### Avoiding double-application in layered systems

Common failure pattern:

`UI adds repetition_penalty` → `SDK also applies repetition processor` → `proxy “helpfully” adds frequency_penalty` → `engine applies penalties again`

Mitigations:

* Define a **single “sampling owner” boundary** (usually the engine).
* In proxies, implement **idempotent pass-through**: if a request already includes penalty fields, do not add/override unless explicitly configured.
* Log **both requested and effective** params at the engine, and optionally attach a “sampling_signature” hash to responses for reproducibility.

### Defaulting strategy

* Engine sets defaults (e.g., OpenAI-style: 0/0; HF-style repetition: 1.0). ([OpenAI Developers][1])
* Clients omit fields unless the user changes a setting.
* Gateways/proxies should avoid injecting defaults unless they must translate between schemas; if they do, they should also **annotate** (internally) that a default was injected.

### What to log for reproducibility

Minimum:

* model id / revision
* tokenizer revision (if applicable)
* prompt + all decoding params (temperature/top_p/top_k/min_p + penalties)
* stop conditions (stop strings / stop token ids)
* whether prompt tokens were included in repetition/history counts (HF can ignore prompt length; llama.cpp can window via `repeat_last_n`). ([GitHub][2])

---

## Vendor / provider documentation survey (what they *say*, not what we infer)

* **OpenAI Chat/Completions** documents `presence_penalty`, `frequency_penalty`, and `logit_bias` as supported request parameters. ([OpenAI Developers][1])
* **vLLM** documents that its sampling parameters follow OpenAI’s completion API and explicitly lists defaults `presence_penalty=0.0`, `frequency_penalty=0.0`, `repetition_penalty=1.0`, plus definitions for each. ([vLLM][5])
* **Cohere** explains that frequency penalty scales with token count, while presence penalty applies once a token has appeared (and frames this over preceding text, including prompt). ([Cohere Documentation][8])
* **Google Gemini API** states presencePenalty applies if the token has already been seen and is “binary on/off,” and points to frequencyPenalty for increasing penalty with reuse. ([Google AI for Developers][7])
* **Mistral** documents `presence_penalty` with default `0` and has separate sampling guidance that includes presence and frequency penalties. ([Mistral AI][9])
* **Hugging Face Transformers** shows an explicit, sign-aware `RepetitionPenaltyLogitsProcessor` implementation and explains how `penalty` values above/below 1 affect repetition. ([GitHub][2])
* **TensorRT-LLM** documents `repetition_penalty` in its runtime/config surfaces (commonly defaulting to `1.0`). ([NVIDIA GitHub][11])

[1]: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create "https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create"
[2]: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/generation/logits_process.py "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/generation/logits_process.py"
[3]: https://developers.openai.com/api/reference/resources/completions/methods/create/ "https://developers.openai.com/api/reference/resources/completions/methods/create/"
[4]: https://platform.openai.com/docs/api-reference/responses "https://platform.openai.com/docs/api-reference/responses"
[5]: https://docs.vllm.ai/en/v0.6.2/dev/sampling_params.html "https://docs.vllm.ai/en/v0.6.2/dev/sampling_params.html"
[6]: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md"
[7]: https://ai.google.dev/api/generate-content "https://ai.google.dev/api/generate-content"
[8]: https://docs.cohere.com/docs/advanced-generation-hyperparameters "https://docs.cohere.com/docs/advanced-generation-hyperparameters"
[9]: https://docs.mistral.ai/api "https://docs.mistral.ai/api"
[10]: https://docs.mistral.ai/capabilities/completion/sampling "https://docs.mistral.ai/capabilities/completion/sampling"
[11]: https://nvidia.github.io/TensorRT-LLM/python-api/tensorrt_llm.runtime.html "https://nvidia.github.io/TensorRT-LLM/python-api/tensorrt_llm.runtime.html"
