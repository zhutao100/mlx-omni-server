# Chat Completions

This API is used for chat and text generation.

## Chat Completion with Streaming

```python
response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0,
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

<details>
<summary><strong>cURL Example</strong></summary>

```bash
curl http://localhost:10240/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "stream": true,
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ]
  }'
```
</details>

## Function Calling

Supported by Qwen3 and GLM model families:

```python
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit",
    messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
    tools=tools,
    tool_choice="auto",
)
```

<details>
<summary><strong>cURL Example</strong></summary>

```bash
curl http://localhost:10240/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit",
    "messages": [{"role": "user", "content": "What''s the weather like in Boston?"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }
    ],
    "tool_choice": "auto"
  }'
```
</details>

## Vision/Multimodal Requests

Supported by Vision-Language Models (VLMs) like LLaVA, Qwen-VL, and CogVLM:

```python
# Image description from URL
response = client.chat.completions.create(
    model="mlx-community/GLM-4.6V-Flash-mxfp4",  # VLM model
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

<details>
<summary><strong>Base64 Image Example</strong></summary>

```python
import base64
from io import BytesIO
from PIL import Image
import requests

# Load and encode image as base64
image_url = "https://example.com/image.jpg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

buffered = BytesIO()
image.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# Send to model
response = client.chat.completions.create(
    model="mlx-community/GLM-4.6V-Flash-mxfp4",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_str}"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```
</details>

<details>
<summary><strong>Streaming with Images</strong></summary>

```python
response = client.chat.completions.create(
    model="mlx-community/GLM-4.6V-Flash-mxfp4",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg"
                    }
                }
            ]
        }
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```
</details>

<details>
<summary><strong>Multiple Images</strong></summary>

```python
response = client.chat.completions.create(
    model="mlx-community/GLM-4.6V-Flash-mxfp4",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these two images:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image1.jpg"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image2.jpg"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```
</details>

## Multimodal Chat Completions

This server supports multimodal chat completions with models like Llava. You can include images in your requests by providing a URL or a base64-encoded image.

### Example with Image URL

```python
import requests

response = requests.post(
    "http://localhost:10240/v1/chat/completions",
    json={
        "mlx-community/GLM-4.6V-Flash-mxfp4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                ],
            }
        ],
    },
)

print(response.json())
```

### Example with Base64-Encoded Image

```python
import requests
import base64

with open("image.jpg", "rb") as f:
    encoded_image = base64.b64encode(f.read()).decode("utf-8")

response = requests.post(
    "http://localhost:10240/v1/chat/completions",
    json={
        "mlx-community/GLM-4.6V-Flash-mxfp4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ],
            }
        ],
    },
)

print(response.json())
```
