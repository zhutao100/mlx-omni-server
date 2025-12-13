# Supported Models

MLX Omni Server supports a comprehensive range of models optimized for Apple Silicon. Here are some popular examples by capability:

## Chat & Text Generation

| Model Family | Examples | Features |
|--------------|----------|----------|
| **Gemma** | `mlx-community/gemma-3-1b-it-4bit-DWQ` | Lightweight, fast inference |
| **Llama** | `mlx-community/Llama-3.2-3B-Instruct-4bit` | Advanced instruction following |
| **Qwen** | `mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit` | Function calling support |
| **GLM** | `mlx-community/glm-4-9b-chat-4bit` | Multi-language capabilities |

## Audio Models

| Type | Models | Description |
|------|--------|-------------|
| **Text-to-Speech** | `lucasnewman/f5-tts-mlx` | Natural voice synthesis |
| **Speech-to-Text** | `mlx-community/whisper-large-v3-turbo` | High accuracy transcription |

## Image Generation

| Model | Description |
|-------|-------------|
| **FLUX** | `argmaxinc/mlx-FLUX.1-schnell` | High-quality image generation |

## Embeddings

| Model | Use Case |
|-------|----------|
| **Sentence Transformers** | `mlx-community/all-MiniLM-L6-v2-4bit` | Semantic search, similarity |

## Vision-Language Models (VLMs)

| Model Family | Examples | Capabilities |
|--------------|----------|--------------|
| **LLaVA** | `llava-hf/llava-v1.6-mistral-7b-hf` | Image understanding, description, Q&A |
| **Qwen-VL** | `Qwen/Qwen-VL-Chat` | Multimodal understanding, OCR, image analysis |
| **CogVLM** | `THUDM/cogvlm-chat-hf` | Advanced visual reasoning, complex image tasks |
| **PaliGemma** | `google/paligemma-3b-pt-224` | Image captioning, visual question answering |

> **Note**: VLM models require more memory and computational resources than text-only models. Ensure your system has adequate resources before using these models.

> **Tip**: Look for quantized models (`4bit`, `8bit`) for better performance on resource-constrained systems.
