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

## Reasoning envelopes (`include=["reasoning.encrypted_content"]`)

Some clients (notably tool-loop clients) need a way to carry “reasoning continuity” across turns without requiring the client to round-trip internal reasoning fields inside chat messages.

If you pass:

```python
include=["reasoning.encrypted_content"]
```

the server will include a `type="reasoning"` output item with an `encrypted_content` token when the underlying model produced reasoning. You can send that same reasoning item back in a subsequent request `input` list.

Operational note: to keep these tokens valid across server restarts, set the environment variable `MLX_OMNI_SERVER_REASONING_HMAC_KEY` (otherwise the server uses an ephemeral per-process key).

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
