# MLX Omni Server

[![PyPI](https://img.shields.io/pypi/v/mlx-omni-server.svg)](https://pypi.python.org/pypi/mlx-omni-server)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zhutao100/mlx-omni-server)
[![License](https://img.shields.io/github/license/zhutao100/mlx-omni-server)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/platform-macOS%20Silicon-red.svg)](https://developer.apple.com/macos/)

![MLX Omni Server Banner](docs/banner.png)

**MLX Omni Server** is a local inference server for Apple Silicon that exposes **OpenAI-compatible HTTP APIs** on top of the **MLX** ecosystem (LLM/VLM + embeddings, with optional image/audio modalities). It is optimized for low-latency, low-concurrency “local/LAN trusted client” usage.

## Fork vs Original Project

This repository is a fork of the [original MLX Omni Server](https://github.com/madroidmaq/mlx-omni-server) project with significant enhancements and modifications.

### Key Enhancements in This Fork

- **OpenAI Responses API** (`/v1/responses`) with SSE streaming and `include=["reasoning.encrypted_content"]`.
- **Robust tool calling** (Qwen3 / GLM4 / Minimax M2-focused parsing + recovery).
- **Vision/VLM support** via `mlx-vlm`.
- **Unified concurrency contract** (shared MLX gate + threadpool offload) across endpoints.

### Differences from Original

The upstream project provided dual API compatibility (OpenAI + Anthropic). This fork focuses on **OpenAI-compatible** endpoints, with deeper support for Responses, tools, and multimodal workloads.

For details on the original project, please refer to the [upstream repository](https://github.com/madroidmaq/mlx-omni-server).

## Features

- **OpenAI-compatible** routes and schemas (works with the official OpenAI Python SDK via `base_url=`).
- **Chat + streaming** (`/v1/chat/completions`) and **Responses API** (`/v1/responses`).
- **Vision** (VLM inputs via standard OpenAI “content parts”).
- **Structured outputs** (JSON Schema) and **logprobs** where supported.
- **Embeddings** (`/v1/embeddings`).
- **Optional modalities** (install extras; routes remain registered and return `501` with an install hint if missing):
  - **Images** (`/v1/images/generations`) via `mflux`
  - **Speech-to-text** (`/v1/audio/transcriptions`) via `mlx-whisper`
  - **Text-to-speech** (`/v1/audio/speech`) via `f5-tts-mlx` / `mlx-audio`
- **Local-first**: models run on your Mac; nothing is sent to a hosted API.

## Supported API Endpoints

The server implements these OpenAI-compatible endpoints (most routes are available with and without the `/v1` prefix):

- **Chat completions**: `/v1/chat/completions`
- **Responses**: `/v1/responses`
- **Audio**
  - `/v1/audio/speech` - Text-to-Speech
  - `/v1/audio/transcriptions` - Speech-to-Text
- **Models**
  - `/v1/models` - List models
  - `/v1/models/{model}` - Retrieve model info
- **Images**
  - `/v1/images/generations` - Image generation
- **Embeddings**
  - `/v1/embeddings` - Create embeddings for text

For detailed API documentation and examples, see [`docs/README.md`](docs/README.md) and [`docs/apis/`](docs/apis/).

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M-series)
- Python **3.11+**
- `git` (some dependencies are installed from Git URLs)
- Internet access for first-time model/dependency downloads (or pre-populate your local caches)

### Installation

```bash
# From PyPI
python3 -m pip install .

# Optional modalities (keeps routes, returns 501 if not installed)
python3 -m pip install ".[images]"  # image generation
python3 -m pip install ".[stt]"     # speech-to-text
python3 -m pip install ".[tts]"     # text-to-speech
python3 -m pip install ".[all]"     # all optional features
```

From source:

```bash
git clone https://github.com/zhutao100/mlx-omni-server.git
cd mlx-omni-server
python3 -m pip install -e ".[all]"
```

### Start the Server

```bash
mlx-omni-server --host 127.0.0.1 --port 10240
```

The server starts on `http://localhost:10240` by default.

### Sanity Check (cURL)

```bash
curl http://localhost:10240/v1/models
```

### Python Client Example

```python
from openai import OpenAI

# Connect to local server
client = OpenAI(
    base_url="http://localhost:10240/v1",
    api_key="not-needed"
)

# Simple chat request (Chat Completions)
response = client.chat.completions.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    messages=[{"role": "user", "content": "Hello! How are you?"}]
)
print(response.choices[0].message.content)
```

### Server Configuration

```bash
# Custom port
mlx-omni-server --port 8000

# Specific host and port
mlx-omni-server --host 127.0.0.1 --port 8000

# View all options
mlx-omni-server --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `0.0.0.0` | Host to bind the server to |
| `--port` | `10240` | Port to bind the server to |
| `--workers` | `1` | Number of worker processes |
| `--log-level` | `info` | Logging level (debug, info, warning, error, critical) |
| `--log-file` | `false` | Enable on-disk logging |
| `--log-dir` | `~/Library/Logs/mlx-omni-server` | Directory for on-disk logs (used with `--log-file`) |
| `--log-file-format` | `jsonl` | On-disk log format: `text` or `jsonl` (used with `--log-file`) |

## Configuration Notes

- **Hugging Face downloads** are handled by `huggingface-hub` (use standard env vars like `HF_HOME`, `HF_TOKEN`, etc. as needed).
- **Responses reasoning tokens**: set `MLX_OMNI_SERVER_REASONING_HMAC_KEY` to keep `reasoning.encrypted_content` stable across server restarts.
- **Multi-worker** (`--workers > 1`) is **opt-in** and can be unsafe for MLX/unified-memory workloads; prefer `--workers 1` unless you understand the tradeoffs (see `docs/concurrency_contract.md`).

## Limitations / Non-goals

- Not a full OpenAI platform clone (no Assistants, Files, Fine-tuning, etc.).
- Responses are tracked in-memory (TTL) and are not persisted across restarts.
- Designed for **trusted localhost/LAN** clients; not hardened for untrusted/public internet exposure.

## Documentation

- Start here: [`docs/README.md`](docs/README.md)
- API reference: [`docs/apis/`](docs/apis/)
- Supported models: [`docs/supported_models.md`](docs/supported_models.md)
- Development guide: [`docs/development_guide.md`](docs/development_guide.md)
- Operations: [`docs/operations.md`](docs/operations.md)
- Concurrency model: [`docs/concurrency_contract.md`](docs/concurrency_contract.md)
- Architecture + roadmap: [`docs/architecture_evaluation.md`](docs/architecture_evaluation.md)

## Next Steps (Roadmap)

See `docs/architecture_evaluation.md` for details. Current priorities:

- Add bounded backpressure around the shared MLX gate (explicit overload behavior).
- Centralize model lifecycle/budgets (cache admission + eviction policies).
- Improve observability around queueing/wait time/execution time.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is a fork of [MLX Omni Server by madroidmaq](https://github.com/madroidmaq/mlx-omni-server). We acknowledge and appreciate the original work that laid the foundation for this enhanced version.

Core Frameworks:
- Built with [MLX](https://github.com/ml-explore/mlx) by Apple
- API design inspired by [OpenAI](https://openai.com)
- Server implementation with [FastAPI](https://github.com/fastapi/fastapi)

## Disclaimer

This project is not affiliated with or endorsed by OpenAI or Apple. It's an independent implementation providing OpenAI-compatible APIs using Apple's MLX framework.
