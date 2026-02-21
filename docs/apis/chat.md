# Chat Completions (`/v1/chat/completions`)

The server implements an OpenAI-compatible Chat Completions endpoint (also available without the `/v1` prefix).

## Non-streaming

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:10240/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(resp.choices[0].message.content)
```

## Streaming (SSE)

Chat Completions streaming uses `text/event-stream` with `data: ...` lines and a final `data: [DONE]` sentinel.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:10240/v1", api_key="not-needed")

stream = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "Write one haiku about MLX."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

### Reasoning deltas (`reasoning_content`)

Some model families support a “thinking mode” and stream reasoning separately in `choices[].delta.reasoning_content`.

- `delta.reasoning_content` is incremental (append-only): clients should concatenate it across chunks.
- The final chunk only signals termination via `finish_reason` and does not resend already-streamed reasoning.

## Penalties and `logit_bias`

This server supports both OpenAI-style additive penalties and HF-style repetition penalties:

- `presence_penalty` / `frequency_penalty` (default `0`): additive, count-based penalties.
- `repetition_penalty` (default `1.0`, disabled) and `repetition_context_size` (default `20`): multiplicative, sign-aware repetition penalty applied over a sliding window.
- `logit_bias` (default `null`): additive per-token bias using OpenAI’s format (`{"token_id_as_string": bias}`).
  - Non-integer / out-of-range token ids are dropped with a warning.
  - Bias values are clamped to `[-100, 100]`.

Migration note: older builds incorrectly treated `presence_penalty` as `repetition_penalty` for text-only models. If you used `presence_penalty` to reduce repetition, switch to `repetition_penalty`.

## Tool / function calling

Tool calling is supported by several model families (for example Qwen3 / GLM4 / Minimax M2). Provide `tools` in the standard OpenAI format.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:10240/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit",
    messages=[{"role": "user", "content": "What's the weather in Boston today?"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ],
    tool_choice="auto",
)

print(resp.choices[0].message.tool_calls)
```

## Vision / multimodal messages

VLM-capable models accept OpenAI-style “content parts” (`text`, `image_url`, `input_audio`).

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:10240/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="mlx-community/GLM-4.6V-Flash-mxfp4",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
            ],
        }
    ],
)
print(resp.choices[0].message.content)
```

### Base64 image URLs

You can also use `data:` URLs:

```json
{
  "type": "image_url",
  "image_url": {"url": "data:image/jpeg;base64,<...>"}
}
```

## Non-standard request fields

This server accepts some additional (non-OpenAI) fields via Pydantic `extra="allow"`. Common ones:

- `adapter_path`: local path to a LoRA/adapter to apply.
- `draft_model`: draft model id used for speculative decoding (when supported by the backend).
  - The alias `draft-model` is also accepted and normalized to `draft_model`.
