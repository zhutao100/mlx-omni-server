# Supported Models

MLX Omni Server is a local inference server built on Apple’s MLX ecosystem and exposes **OpenAI-compatible** endpoints (Chat Completions + the newer **Responses API**) for multiple modalities. This page lists **known-good, commonly used examples** by capability.

Key project context that affects model support:

- **Core install** (`pip install .`) supports: **chat/text + responses + embeddings + vision/VLM**.
- **Optional installs** keep the routes registered but will return **501** if the modality isn’t installed:
  - `pip install ".[images]"` for image generation (via `mflux`)
  - `pip install ".[stt]"` for speech-to-text
  - `pip install ".[tts]"` for text-to-speech
- This fork includes **advanced tool parsing** and recovery specifically for **Qwen3**, **GLM4**, and **Minimax M2** families, so tool calling works reliably when using `/v1/responses` or `/v1/chat/completions`.

> Tip: Prefer quantized MLX models (`4bit`, `8bit`, `mxfp4`) for Apple Silicon. Always benchmark on your workload—tool calling, OCR, and VLM reasoning can be more sensitive to prompt/template and quantization choices.

---

## Chat & Text Generation (Core)

These models run with the core install and are best paired with the **Responses API** for robust tool calling and structured outputs.

| Model Family | Example Model IDs | Description | Strengths in MLX Omni Server | Typical Use Cases |
|---|---|---|---|---|
| **Qwen3 (Coder / Instruct)** | `mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit` | Coding-focused instruct model in MLX format. | Strong coding, refactors, debugging, and agentic workflows; tool calling works well with this fork’s enhanced parsing. | Coding assistant, tool-using coding agents, code review/refactor endpoints. |
| **GLM 4.x (Chat / Agent)** | `mlx-community/GLM-4.5-air-mxfp4` | Efficient GLM model variant in MLX format. | Solid general chat + reasoning; tool calling benefits from GLM-family parsing enhancements in this fork. | General assistant, tool-based automation flows, mixed reasoning + coding. |
| **Lightweight Instruct (examples)** | `mlx-community/gemma-3-1b-it-4bit-DWQ` | Small instruct model for fast local chat. | Very low latency and memory footprint; good for “always-on” local usage. | Fast chat, command-style helpers, lightweight developer tools. |

---

## Vision-Language Models (VLMs) (Core if VLM deps are included in your setup)

This fork adds **comprehensive VLM support** via `mlx-vlm` integration (part of the core install). VLMs can be used through `/v1/responses` (recommended for structured/streaming workflows) and vision-capable request formats.

| Model Family | Example Model IDs | Description | Capabilities | Typical Use Cases |
|---|---|---|---|---|
| **GLM-V** | `mlx-community/GLM-4.6V-mxfp4` | GLM multimodal model in MLX format. | Image understanding, document/screenshot QA, tool calling with vision inputs, interleaved text+image style workflows (model-dependent). | Screenshot reasoning, multimodal agents, document understanding (images/PDF renders), UI analysis + generation. |
| **Qwen-VL** | `mlx-community/Qwen3-VL-32B-Instruct-8bit` | Qwen multimodal model in MLX format. | Strong OCR + image analysis, diagram/table understanding, vision-augmented tool use. | OCR-heavy pipelines, screenshot QA, multimodal RAG front-ends, visual inspection tasks. |

---

## Embeddings (Core)

Embeddings are served via `/v1/embeddings` and are suitable for local RAG, semantic search, clustering, and similarity.

| Model Type | Example Model IDs | Description | Use Cases |
|---|---|---|---|
| **Sentence Transformers** | `mlx-community/all-MiniLM-L6-v2-4bit` | Compact embedding model in MLX format. | Semantic search, similarity, clustering, lightweight local RAG. |

---

## Audio: Speech-to-Text (Optional: `.[stt]`)

Served via `/v1/audio/transcriptions`.

| Model | Description | Strengths | Typical Use Cases |
|---|---|---|---|
| `mlx-community/whisper-large-v3-turbo` | MLX Whisper “turbo” variant optimized for speed. | High-quality transcription with strong speed/latency characteristics on Apple Silicon. | Local transcription, meeting notes, voice input for agents. |

---

## Audio: Text-to-Speech (Optional: `.[tts]`)

Served via `/v1/audio/speech`.

| Model | Description | Strengths | Typical Use Cases |
|---|---|---|---|
| `lucasnewman/f5-tts-mlx` | MLX TTS model based on F5-TTS-style architecture. | Natural speech generation and good responsiveness for local inference. | Local voice responses, accessibility tooling, “talking assistant” demos. |
| `mlx-community/Kokoro-82M-4bit` | `mlx-audio`-based TTS model family. | Lightweight, fast local speech generation. | Low-latency TTS, interactive assistants, demos. |

---

## Image Generation (Optional: `.[images]`, via `mflux`)

Served via `/v1/images/generations`.

This project’s image generation stack is built around **`mflux`**, and the model is **Z-Image-Turbo**.

| Model | Description | Strengths | Typical Use Cases |
|---|---|---|---|
| `filipstrand/Z-Image-Turbo-mflux-4bit` | The default image model for this server’s `mflux` pipeline. | Fast, high-quality text-to-image generation optimized for local workflows. | Local image generation, creative tooling, product mockups, “prompt-to-image” endpoints. |

---

## Practical Guidance

### 1) Prefer `/v1/responses` for tool calling and structured outputs
This fork’s Responses support is designed for:
- better streaming event handling,
- more robust tool-call parsing (especially Qwen3 + GLM4 + Minimax M2),
- structured outputs and logprobs workflows.

### 2) Quantization selection
- `4bit`: best for memory-constrained machines; fastest to load; may lose some fidelity.
- `8bit`: better quality; still efficient.
- `mxfp4`: strong perf/quality trade-off on supported paths; test on your target hardware.

### 3) “Supported” vs “Compatible”
The examples above are popular, known-good starting points. In practice, most **MLX-converted** LLM/VLM/embedding models that follow the expected chat/template conventions will run well in MLX Omni Server.
