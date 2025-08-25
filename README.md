# MLX Omni Server

[![PyPI](https://img.shields.io/pypi/v/mlx-omni-server.svg)](https://pypi.python.org/pypi/mlx-omni-server)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zhutao100/mlx-omni-server)
[![License](https://img.shields.io/github/license/zhutao100/mlx-omni-server)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/platform-macOS%20Silicon-red.svg)](https://developer.apple.com/macos/)

![MLX Omni Server Banner](docs/banner.png)

**MLX Omni Server** is a high-performance local inference server built on Apple's MLX framework, optimized for Apple Silicon (M-series) chips. It provides OpenAI-compatible API endpoints, enabling seamless integration with existing OpenAI SDK clients while delivering fast, private AI processing directly on your Mac.

## ✨ Features

- 🚀 **Apple Silicon Optimized**: Built on MLX framework, specifically tuned for M1/M2/M3/M4 chips
- 🔌 **OpenAI API Compatible**: Drop-in replacement for OpenAI API endpoints
- 🎯 **Comprehensive AI Capabilities**:
  - 🗣️ **Chat & Text Generation**: Multi-turn conversations, streaming responses, function calling
  - 🔊 **Audio Processing**: Text-to-Speech (TTS) and Speech-to-Text (STT) support
  - 🖼️ **Image Generation**: High-quality image creation with FLUX models
  - 🔍 **Embeddings**: Text vectorization for semantic search and similarity
- ⚡ **High Performance**: Local inference with hardware acceleration
- 🔐 **Privacy-First**: All processing happens locally on your machine
- 🛠️ **Developer Friendly**: Works with official OpenAI SDK and other compatible clients
- 📦 **Easy Installation**: Simple pip install with minimal dependencies

## Supported API Endpoints

The server implements OpenAI-compatible endpoints:

- [Chat completions](https://platform.openai.com/docs/api-reference/chat): `/v1/chat/completions`
  - ✅ Chat
  - ✅ Tools, Function Calling
  - ✅ Structured Output
  - ✅ LogProbs
  - 🚧 Vision
- [Audio](https://platform.openai.com/docs/api-reference/audio)
  - ✅ `/v1/audio/speech` - Text-to-Speech
  - ✅ `/v1/audio/transcriptions` - Speech-to-Text
- [Models](https://platform.openai.com/docs/api-reference/models/list)
  - ✅ `/v1/models` - List models
  - ✅ `/v1/models/{model}` - Retrieve model info
- [Images](https://platform.openai.com/docs/api-reference/images)
  - ✅ `/v1/images/generations` - Image generation
- [Embeddings](https://platform.openai.com/docs/api-reference/embeddings)
  - ✅ `/v1/embeddings` - Create embeddings for text

## 🚀 Quick Start

Get up and running with MLX Omni Server in minutes:

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4 chip)
- Python 3.11 or higher
- Internet connection for initial model downloads

### Installation

```bash
pip install mlx-omni-server
```

### Start the Server

```bash
mlx-omni-server
```

The server starts on `http://localhost:10240` by default.

### Basic Test

```bash
curl http://localhost:10240/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-3-1b-it-4bit-DWQ",
    "messages": [
      {
        "role": "user",
        "content": "What can you do?"
      }
    ]
  }'
```

### Python Client Example

```python
from openai import OpenAI

# Connect to local server
client = OpenAI(
    base_url="http://localhost:10240/v1",
    api_key="not-needed"
)

# Simple chat request
response = client.chat.completions.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    messages=[{"role": "user", "content": "Hello! How are you?"}]
)
print(response.choices[0].message.content)
```

🎉 **Congratulations!** You're now running AI locally on your Mac.

### Server Configuration

```bash
# Default settings (port 10240, all interfaces)
mlx-omni-server

# Custom port
mlx-omni-server --port 8000

# Specific host and port
mlx-omni-server --host 127.0.0.1 --port 8000

# Development with debug logging
mlx-omni-server --log-level debug

# Production with multiple workers
mlx-omni-server --workers 2 --log-level warning

# View all options
mlx-omni-server --help
```

#### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `0.0.0.0` | Host to bind the server to |
| `--port` | `10240` | Port to bind the server to |
| `--workers` | `1` | Number of worker processes |
| `--log-level` | `info` | Logging level (debug, info, warning, error, critical) |

## 📚 Advanced Usage

### API Interaction Methods

MLX Omni Server supports multiple ways to interact with AI capabilities:

#### 1. REST API Direct Access

```bash
# Chat completions
curl http://localhost:10240/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-3-1b-it-4bit-DWQ",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# List available models
curl http://localhost:10240/v1/models
```

#### 2. OpenAI SDK Integration

```python
from openai import OpenAI

# Standard client setup
client = OpenAI(
    base_url="http://localhost:10240/v1",
    api_key="not-needed"
)
```

#### 3. FastAPI TestClient (Development)

Perfect for testing without starting a server:

```python
from openai import OpenAI
from fastapi.testclient import TestClient
from mlx_omni_server.main import app

client = OpenAI(http_client=TestClient(app))
```

### 🎯 API Examples

#### Chat Completion with Streaming

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

#### Function Calling

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
    "messages": [{"role": "user", "content": "What\'s the weather like in Boston?"}],
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

#### 🔊 Audio Processing

**Text-to-Speech (TTS)**

```python
speech_file_path = "mlx_example.wav"
response = client.audio.speech.create(
    model="lucasnewman/f5-tts-mlx",
    voice="alloy",  # Available voices: alloy, echo, fable, onyx, nova, shimmer
    input="MLX project is awesome!",
)
response.stream_to_file(speech_file_path)
```

<details>
<summary>Curl Example</summary>

```shell
curl -X POST "http://localhost:10240/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lucasnewman/f5-tts-mlx",
    "input": "MLX project is awsome",
    "voice": "alloy"
  }' \
  --output ~/Desktop/mlx.wav
```

</details>

#### Speech-to-Text

```python
audio_file = open("speech.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="mlx-community/whisper-large-v3-turbo",
    file=audio_file
)

print(transcript.text)
```

<details>
<summary><strong>cURL Example</strong></summary>

```bash
curl -X POST "http://localhost:10240/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@mlx_example.wav" \
  -F "model=mlx-community/whisper-large-v3-turbo"
```

**Response:**

```json
{
  "text": " MLX Project is awesome!"
}
```

</details>

#### 🖼️ Image Generation

```python
image_response = client.images.generate(
    model="argmaxinc/mlx-FLUX.1-schnell",
    prompt="A serene landscape with mountains and a lake",
    n=1,
    size="512x512"
)
```

<details>
<summary>Curl Example</summary>

```shell
curl http://localhost:10240/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "argmaxinc/mlx-FLUX.1-schnell",
    "prompt": "A cute baby sea otter",
    "n": 1,
    "size": "1024x1024"
  }'
```

</details>

#### Embeddings

```python
# Generate embedding for a single text
response = client.embeddings.create(
    model="mlx-community/all-MiniLM-L6-v2-4bit", input="I like reading"
)

# Examine the response structure
print(f"Response type: {type(response)}")
print(f"Model used: {response.model}")
print(f"Embedding dimension: {len(response.data[0].embedding)}")
```

<details>
<summary><strong>cURL Example</strong></summary>

```bash
curl http://localhost:10240/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/all-MiniLM-L6-v2-4bit",
    "input": ["Hello world!", "Embeddings are useful for semantic search."]
  }'
```

</details>

For more detailed examples, check out the [examples](examples) directory.

## 🤖 Supported Models

MLX Omni Server supports a comprehensive range of models optimized for Apple Silicon. Here are some popular examples by capability:

### 💬 Chat & Text Generation

| Model Family | Examples | Features |
|--------------|----------|----------|
| **Gemma** | `mlx-community/gemma-3-1b-it-4bit-DWQ` | Lightweight, fast inference |
| **Llama** | `mlx-community/Llama-3.2-3B-Instruct-4bit` | Advanced instruction following |
| **Qwen** | `mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit` | Function calling support |
| **GLM** | `mlx-community/glm-4-9b-chat-4bit` | Multi-language capabilities |

### 🔊 Audio Models

| Type | Models | Description |
|------|--------|-------------|
| **Text-to-Speech** | `lucasnewman/f5-tts-mlx` | Natural voice synthesis |
| **Speech-to-Text** | `mlx-community/whisper-large-v3-turbo` | High accuracy transcription |

### 🖼️ Image Generation

| Model | Description |
|-------|-------------|
| **FLUX** | `argmaxinc/mlx-FLUX.1-schnell` | High-quality image generation |

### 🔍 Embeddings

| Model | Use Case |
|-------|----------|
| **Sentence Transformers** | `mlx-community/all-MiniLM-L6-v2-4bit` | Semantic search, similarity |

> 💡 **Tip**: Look for quantized models (`4bit`, `8bit`) for better performance on resource-constrained systems.

## ❓ Frequently Asked Questions

### How are models managed?

MLX Omni Server automatically downloads models from Hugging Face when first used. For better performance:

```python
# Auto-download on first use
response = client.chat.completions.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",  # Downloads if not available
    messages=[{"role": "user", "content": "Hello"}]
)

# Use pre-downloaded local models
response = client.chat.completions.create(
    model="/path/to/your/local/model",  # Local path
    messages=[{"role": "user", "content": "Hello"}]
)

# List available models
curl http://localhost:10240/v1/models
```

### How do I specify which model to use?

Simply use the `model` parameter in your requests:

```python
response = client.chat.completions.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Can I use TestClient for development?

Perfect for testing without starting a server:

```python
from openai import OpenAI
from fastapi.testclient import TestClient
from mlx_omni_server.main import app

client = OpenAI(http_client=TestClient(app))
response = client.chat.completions.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Server startup troubleshooting

| Issue | Solution |
|-------|----------|
| **Apple Silicon required** | Ensure you have M1/M2/M3/M4 Mac |
| **Python version** | Use Python 3.11+ |
| **Installation** | Verify latest version installed |
| **Debug info** | Check logs for detailed errors |

## Troubleshooting

Common issues and their solutions:

### Model Download Issues

If you encounter problems downloading models:

1. Check your internet connection
2. Verify you have sufficient disk space
3. Try downloading the model directly with Hugging Face tools:

```bash
huggingface-cli download mlx-community/gemma-3-1b-it-4bit-DWQ
```

### Memory Errors

If you get out-of-memory errors:

1. Use a smaller quantized model (4bit instead of 8bit)
2. Close other memory-intensive applications
3. Restart the server with a fresh process

### Performance Issues

If responses are slow:

1. Ensure you're using quantized models
2. Check that you have adequate cooling (thermal throttling can reduce performance)
3. Consider using a model better suited to your hardware

## Model Management

MLX Omni Server provides flexible model management capabilities:

### Automatic Model Downloading

When you specify a model ID that hasn't been downloaded yet, the framework will automatically download it from Hugging Face:

```python
response = client.chat.completions.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",  # Will download if not available
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Using Local Models

To use a locally downloaded model, simply set the `model` parameter to the local model path:

```python
response = client.chat.completions.create(
    model="/path/to/your/local/model",  # Local model path
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Listing Available Models

You can check which models are available on your system:

```bash
curl http://localhost:10240/v1/models
```

Or using the Python client:

```python
response = client.models.list()
for model in response.data:
    print(f"Model ID: {model.id}")
```

## ⚡ Performance Optimization

### Model Selection

| Quantization | Benefits | Best For |
|--------------|----------|----------|
| **4-bit** | Fast inference, low memory | Everyday use, M1/M2 Macs |
| **8-bit** | Better quality, still fast | High-quality results |
| **DWQ** | Optimized for MLX | Specialized workloads |

### Hardware Recommendations

| Component | Recommendation |
|-----------|----------------|
| **Memory** | 16GB+ RAM for larger models |
| **Storage** | SSD for faster loading |
| **Cooling** | Adequate for sustained performance |

### Production Configuration

```bash
# Multi-worker setup for better throughput
mlx-omni-server --workers 2 --log-level warning

# Development with hot reload
uvicorn mlx_omni_server.main:app --reload --port 10240
```

## 🔧 Development

### Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/zhutao100/mlx-omni-server.git
   cd mlx-omni-server
   ```

2. **Install dependencies:**

   ```bash
   pip install -e .
   ```

3. **Run in development mode:**

   ```bash
   uvicorn mlx_omni_server.main:app --reload --host 0.0.0.0 --port 10240
   ```

### Testing

```bash
pytest
```

For detailed development information, see our [Development Guide](docs/development_guide.md).

## 🤝 Contributing

We welcome contributions! Please see our [Development Guide](docs/development_guide.md) for detailed information about:

- Development environment setup
- Running in development mode
- Contribution guidelines
- Testing and documentation

For major changes, please open an issue first to discuss proposed changes.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [MLX](https://github.com/ml-explore/mlx) by Apple
- API design inspired by [OpenAI](https://openai.com)
- Server implementation with [FastAPI](https://fastapi.tiangolo.com/)
- Chat generation by [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
- Image generation by [mflux](https://github.com/filipstrand/mflux)
- Audio processing by [f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx), [mlx-whisper](https://github.com/ml-explore/mlx-examples/blob/main/whisper/README.md), and [mlx-audio](https://github.com/Blaizzy/mlx-audio)
- Embeddings by [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)

## ⚠️ Disclaimer

This project is not affiliated with or endorsed by OpenAI or Apple. It's an independent implementation providing OpenAI-compatible APIs using Apple's MLX framework.
