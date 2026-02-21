# Responses

This API is a shim over the existing Chat Completions implementation, exposing an OpenAI-compatible `/v1/responses` surface.

Response IDs use the Responses namespace (`resp_...`) and do not expose the underlying Chat Completions `chatcmpl-...` IDs.

> Note: Most routes are available with and without the `/v1` prefix.

## Create (non-stream)

```python
response = client.responses.create(
    model="mlx-community/Qwen3-1.7B-4bit-DWQ-053125",
    input=[{"role": "user", "content": "Hello!"}],
    max_output_tokens=200,
)
print(response.output_text)
```

## Create (stream)

Streaming is SSE and is event-driven (no `data: [DONE]` sentinel).

```python
events = []
with client.responses.stream(
    model="mlx-community/Qwen3-1.7B-4bit-DWQ-053125",
    input=[{"role": "user", "content": "Hello!"}],
) as stream:
    for event in stream:
        events.append(event)
final = stream.get_final_response()
```

## Structured outputs (`text.format`)

Responses-style `text.format` is mapped to the underlying chat `response_format`.

```python
response = client.responses.create(
    model="mlx-community/Qwen3-1.7B-4bit-DWQ-053125",
    input="Return JSON with a greeting",
    text={
        "format": {
            "type": "json_schema",
            "name": "greeting",
            "schema": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            "strict": True,
        }
    },
)
```

## Penalties and `logit_bias`

The Responses API is a shim over Chat Completions, so the same knobs are supported and passed through:

- `presence_penalty` / `frequency_penalty` (default `0` via Chat): additive, count-based penalties.
- `repetition_penalty` (default `1.0`, disabled) and `repetition_context_size` (default `20`): HF-style repetition penalty.
- `logit_bias` (default `null`): OpenAI-format token-id-as-string mapping.

Migration note: older builds incorrectly treated `presence_penalty` as `repetition_penalty` for text-only models. If you used `presence_penalty` to reduce repetition, switch to `repetition_penalty`.

## Reasoning envelopes (`include=["reasoning.encrypted_content"]`)

Some clients (notably tool-loop clients) need a way to carry “reasoning continuity” across turns without requiring the client to round-trip internal reasoning fields inside chat messages.

If you pass:

```python
include=["reasoning.encrypted_content"]
```

When the underlying model produces reasoning, the server emits a `type="reasoning"` output item with the raw reasoning under `content`:

```json
{
  "type": "reasoning",
  "status": "completed",
  "content": [{ "type": "reasoning_text", "text": "…" }],
  "summary": []
}
```

If you opt in via `include=["reasoning.encrypted_content"]`, the same reasoning item also includes an `encrypted_content` token suitable for replay in a later request `input` list.

Operational note: to keep these tokens valid across server restarts, set the environment variable `MLX_OMNI_SERVER_REASONING_HMAC_KEY` (otherwise the server uses an ephemeral per-process key).

### Streaming reasoning events

For `stream=true` requests, reasoning is streamed using Responses-native SSE events:

- `response.reasoning_text.delta` (with `content_index: 0`)
- `response.reasoning_text.done` (with the full text, `content_index: 0`)

### `reasoning.effort` mapping (request-side)

If a Responses request includes `reasoning: {"effort": ...}`, the server maps it to local MLX thinking knobs (`enable_thinking` and `thinking_budget`) unless those are explicitly provided in the request.

## Stateful chaining (`previous_response_id`)

If `previous_response_id` is set, the server prepends the previous response's stored history to the current request (excluding the previous request's `instructions` system message).

## Background mode (`background=true`)

If `background=true` (and `stream=false`), the server returns a queued Response immediately and runs generation in the background. You can then:

- `GET /v1/responses/{response_id}` to poll the latest status.
- `GET /v1/responses/{response_id}?stream=true` to receive SSE lifecycle events as they are recorded.
- `POST /v1/responses/{response_id}/cancel` for best-effort cancellation.

## Additional endpoints

- `GET /v1/responses/{response_id}` (retrieve)
- `DELETE /v1/responses/{response_id}` (delete; returns `{id, object:"response", deleted:true}`)
- `POST /v1/responses/{response_id}/cancel` (cancel; primarily for background responses)
- `GET /v1/responses/{response_id}/input_items` (inspect resolved input items; returns an OpenAI-style list envelope with `object`, `first_id`, `last_id`, `has_more`)

## Unsupported fields

High-level Responses features that are not implemented are rejected with a `400` error (not silently accepted), including:

- `conversation`
- `include` values other than `reasoning.encrypted_content`

## Storage model

Responses are tracked in an in-memory registry (including streaming events) with a TTL. Records are not persisted across server restarts.
