## Implementation Plan — Option A Reasoning Streaming UX + Keep Option B `previous_response_id` Intact

### Goal

Upgrade the existing Responses ↔ Chat Completions adapter so that **whenever the upstream produces reasoning** (stream `delta.reasoning` and/or non-stream `message.reasoning`), the proxy:

1. **Always** emits **Responses SSE reasoning delta events**:

   * `response.reasoning_text.delta`
   * `response.reasoning_text.done`

2. **Always** emits a **Responses reasoning output item** whose **raw reasoning is stored under `content`** as `reasoning_text` parts (not under `summary`).

3. **Only when** the request includes `include: ["reasoning.encrypted_content"]`, attach an `encrypted_content` blob to that same reasoning item (keep the project’s current “integrity-protected encoding” approach unchanged).

4. **Keeps existing Option B behavior unchanged**:

   * `previous_response_id` registry/history replay remains as-is (`responses/router.py`, `responses/registry.py` behavior preserved).

---

## Target End-State (Concrete)

### When upstream emits reasoning

* **Streaming**: any `delta.reasoning` fragments are surfaced immediately as `response.reasoning_text.delta` events (unconditional).
* **Non-stream**: any `message.reasoning` is surfaced as a reasoning output item plus (if streaming endpoint is used) a single delta/done emission.

### Responses reasoning output item shape (OpenAI-aligned)

Use **`content`** for raw reasoning text parts:

```json
{
  "id": "rs_…",
  "type": "reasoning",
  "status": "completed",
  "content": [
    { "type": "reasoning_text", "text": "…raw reasoning…" }
  ],
  "summary": [],
  "encrypted_content": "v1.…"   // only if include requests it
}
```

Notes:

* `summary` is reserved for actual summaries (e.g., `summary_text` parts) and should **not** be repurposed for raw chain-of-thought.
* `encrypted_content` remains include-gated and format-unchanged.

---

## PR 1 — Schema + Non-Streaming Reasoning Item

### 1.1 Update Responses schema model to include `content`

**File:** `src/mlx_omni_server/responses/schema.py`

Update `ResponseOutputReasoning` to support:

* `content: Optional[list[dict]]`
* `summary: Optional[list[dict]]` (keep for future summary support)
* existing `encrypted_content`

Example:

```py
class ResponseOutputReasoning(BaseModel):
    id: str
    type: Literal["reasoning"] = "reasoning"
    status: ResponseOutputItemStatus = ResponseOutputItemStatus.COMPLETED

    # NEW: raw reasoning goes here
    content: Optional[list[dict]] = None

    # Keep for spec alignment (summary is not raw CoT)
    summary: Optional[list[dict]] = None

    # existing
    encrypted_content: Optional[str] = None
```

Canonical reasoning content part shape:

```py
{"type": "reasoning_text", "text": "<reasoning string>"}
```

### 1.2 Non-stream mapping: always emit reasoning output item with `content`

**File:** `src/mlx_omni_server/responses/adapter.py`
**Function:** `chat_response_to_response(...)`

If `message.reasoning` exists:

* Always set `item["content"] = [{"type":"reasoning_text","text": message.reasoning}]`
* Set `item["summary"] = []` (or omit; choose one consistent convention)
* Only set `item["encrypted_content"]` when include requests it

Pseudo:

```py
if message and message.reasoning:
    item = {"id": ..., "type":"reasoning", "status":"completed"}
    item["content"] = [{"type":"reasoning_text","text": message.reasoning}]
    item["summary"] = []

    if include_reasoning_encrypted:
        item["encrypted_content"] = seal(ReasoningEnvelope(...))

    output_items.append(item)
```

No changes to message/tool-call items.

---

## PR 2 — Streaming: Unconditional `reasoning_text.delta` + Correct Output Item

### 2.1 OutputItemState: store reasoning internally, serialize as `content`

**File:** `src/mlx_omni_server/responses/adapter.py`
**Class:** `OutputItemState`

* Keep internal accumulator `self.text` (internal “full reasoning so far”).
* Update `to_output_dict()` for reasoning items:

```py
if self.kind == "reasoning":
    payload = {"id": self.item_id, "type": "reasoning", "status": self.status}

    # Correct: raw reasoning in `content`
    if self.text is not None:
        payload["content"] = [{"type": "reasoning_text", "text": self.text}]
    else:
        payload["content"] = [{"type": "reasoning_text", "text": ""}]

    payload["summary"] = []

    if self.encrypted_content is not None:
        payload["encrypted_content"] = self.encrypted_content
    return payload
```

### 2.2 Ensure reasoning item is created with a placeholder content part

**File:** `src/mlx_omni_server/responses/adapter.py`
**Class:** `ResponseStreamAdapter`

Update `_ensure_reasoning_item(choice_index)` to create a reasoning item with:

* `content: [{"type":"reasoning_text","text":""}]`
* `summary: []`

This guarantees `content_index = 0` is valid for downstream consumers.

### 2.3 SSE event builders for reasoning deltas/done (include `content_index`)

**File:** `src/mlx_omni_server/responses/adapter.py`
**Class:** `ResponseStreamAdapter`

Add helpers:

```py
def _build_reasoning_delta_event(self, state, delta_fragment):
    return ResponseStreamEvent(
        event="response.reasoning_text.delta",
        data={
            "type": "response.reasoning_text.delta",
            "sequence_number": self._next_sequence(),
            "output_index": state.index,
            "item_id": state.item_id,
            "content_index": 0,
            "delta": delta_fragment,
        },
    )

def _build_reasoning_done_event(self, state):
    return ResponseStreamEvent(
        event="response.reasoning_text.done",
        data={
            "type": "response.reasoning_text.done",
            "sequence_number": self._next_sequence(),
            "output_index": state.index,
            "item_id": state.item_id,
            "content_index": 0,
            "text": state.text or "",
        },
    )
```

### 2.4 Streaming mapping: unconditional reasoning deltas (no include gating)

**File:** `src/mlx_omni_server/responses/adapter.py`
**Method:** `ResponseStreamAdapter.on_chunk(...)`

On every non-empty `delta.reasoning`:

1. Ensure reasoning item exists (`_ensure_reasoning_item`)
2. Merge into `self._reasoning_by_choice[choice.index]`
3. Use `_append_text_delta(r_state, merged_full)` to compute “new fragment”
4. Emit `response.reasoning_text.delta` with that fragment

**Critical placement:** perform this **early** in `on_chunk()` before any `continue` paths (tool calls, etc.) so reasoning is never suppressed.

### 2.5 Handle “reasoning only appears at end” gracefully

Some upstreams may not stream reasoning deltas. Support:

* If no `delta.reasoning` ever arrived, but final reasoning is known at `on_done()` time, emit:

  * `response.output_item.added` (if not already created)
  * one `response.reasoning_text.delta` containing the full reasoning
  * `response.reasoning_text.done`

This preserves the downstream contract even when upstream isn’t incremental.

### 2.6 Done behavior ordering

**File:** `src/mlx_omni_server/responses/adapter.py`
**Method:** `ResponseStreamAdapter.on_done(...)`

For each reasoning item:

1. Emit `response.reasoning_text.done` **once**
2. Then emit the existing `response.output_item.done` (whose `item` now includes full `content`)

### 2.7 Preserve include-gated `encrypted_content` unchanged

**File:** `src/mlx_omni_server/responses/adapter.py`
**Method:** `ResponseStreamAdapter.on_done(...)`

Keep current logic:

* Only attach `encrypted_content = seal(...)` when include contains `"reasoning.encrypted_content"`.
* Keep current envelope format and validation rules unchanged.

### 2.8 Output index correctness rule (no sparse indices)

Do **not** reserve or “park” reasoning items at giant indices. Keep dense sequential indices:

* Events’ `output_index` must match final `response.output` list position after sorting.
* This preserves downstream client expectations and avoids list/stream divergence.

---

## PR 3 — Explicit Thinking Mapping (Request-Side), Minimal and Safe

### 3.1 Map `Responses.reasoning.effort` → local MLX knobs

**File:** `src/mlx_omni_server/responses/adapter.py`
**Function:** `response_request_to_chat_request(...)`

Add `_apply_reasoning_to_thinking_params(payload: dict)` before validating `ChatCompletionRequest`.

Rules:

* If `payload["reasoning"]["effort"]` exists:

  * Normalize unknown values safely (default to `"medium"` behavior).
  * Set `enable_thinking = (effort != "none")` **only if** not already provided.
  * Set `thinking_budget = map_effort_to_budget(effort)` **only if** not already provided.
* Do not override explicit user-provided `enable_thinking` / `thinking_budget`.

Conservative default mapping (tunable later):

* minimal → 256
* low → 512
* medium → 1024
* high → 2048
* xhigh → 4096
* none → disable thinking (if not explicitly enabled)

Purpose: make “reasoning streaming UX” work reliably on the local model path without vendor coupling.

---

## Tests (Must-Have)

### T1 — Streaming emits reasoning deltas without `include`

* Build `ResponseStreamAdapter(include=[])`
* Feed stream chunks with `delta.reasoning="a"`, then `"b"`
* Assert event sequence includes:

  * `response.output_item.added` (reasoning item with placeholder `content[0]`)
  * `response.reasoning_text.delta` for “a”, then “b” (`content_index: 0`)
  * `response.reasoning_text.done` with “ab” (`content_index: 0`)
* Assert final response output contains reasoning item:

  * `content == [{"type":"reasoning_text","text":"ab"}]`
  * `summary == []` (or omitted, per chosen convention)
  * no `encrypted_content`

### T2 — `encrypted_content` remains include-gated

* Same as T1 but include `["reasoning.encrypted_content"]`
* Assert final reasoning item contains `encrypted_content`
* Assert `unseal(encrypted_content)` succeeds

### T3 — “Reasoning only at end” case

* Feed chunks with no `delta.reasoning`
* Provide final reasoning at `on_done()` (as the adapter currently can infer)
* Assert:

  * reasoning item exists
  * emits a single delta + done (or at least done, per your chosen behavior)
  * final output item has full `content`

### T4 — Tool-call interleaving does not drop reasoning

* Chunk order:

  1. reasoning delta
  2. tool_call delta
  3. reasoning delta
* Assert both reasoning deltas are emitted (placement before tool-call continues)

### T5 — Option B regression: `previous_response_id` unchanged

* Create response #1 with message/tool calls + reasoning
* Store via existing registry path
* Create request #2 with `previous_response_id`
* Assert history reconstruction/tool-call continuity unchanged
* Confirm no required changes in `responses/router.py` and `responses/registry.py`

### T6 — Event `output_index` correctness

* Verify the `output_index` used in reasoning events matches the final sorted output index of the reasoning item.

---

## Definition of Done

* [ ] Reasoning output items carry raw reasoning under `content: [{"type":"reasoning_text","text":...}]` (not `summary`)
* [ ] `response.reasoning_text.delta` emitted for every upstream `delta.reasoning`, unconditionally, with `content_index: 0`
* [ ] `response.reasoning_text.done` emitted once per reasoning item, ordered before `response.output_item.done`
* [ ] If reasoning appears only at end, proxy emits a coherent delta/done sequence and final output is correct
* [ ] `encrypted_content` attached only when requested via `include`, format unchanged
* [ ] No behavioral changes to `previous_response_id` flow; Option B stays intact
