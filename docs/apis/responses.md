# Responses

This API is a shim over the existing Chat Completions implementation, exposing an OpenAI-compatible `/v1/responses` surface.

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

## Stateful chaining (`previous_response_id`)

If `previous_response_id` is set, the server prepends the previous response's stored history to the current request (excluding the previous request's `instructions` system message).

## Background mode (`background=true`)

If `background=true` (and `stream=false`), the server returns a queued Response immediately and runs generation in the background. You can then:

- `GET /v1/responses/{response_id}` to poll the latest status.
- `GET /v1/responses/{response_id}?stream=true` to receive SSE lifecycle events as they are recorded.
- `POST /v1/responses/{response_id}/cancel` for best-effort cancellation.

## Additional endpoints

- `GET /v1/responses/{response_id}` (retrieve)
- `DELETE /v1/responses/{response_id}` (delete)
- `POST /v1/responses/{response_id}/cancel` (cancel; primarily for background responses)
- `GET /v1/responses/{response_id}/input_items` (inspect the resolved input items; supports `order`, `limit`, `after`, `before`)

## Storage model

Responses are tracked in an in-memory registry (including streaming events) with a TTL. Records are not persisted across server restarts.
