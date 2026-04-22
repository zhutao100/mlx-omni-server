# Documentation Index

This directory contains the project’s reference documentation. The **source of truth** for behavior is always the implementation under `src/` plus the test suite under `tests/`; these docs summarize the current design and supported workflows.

## Start here

- User-facing overview + quickstart: [`../README.md`](../README.md)
- Supported models and “known-good” starting points: [`supported_models.md`](supported_models.md)

## API reference (OpenAI-compatible)

The server supports most routes with and without the `/v1` prefix. These pages document the supported request/response shapes and examples:

- Chat Completions: [`apis/chat.md`](apis/chat.md)
- Responses API: [`apis/responses.md`](apis/responses.md)
- Embeddings: [`apis/embeddings.md`](apis/embeddings.md)
- Images: [`apis/images.md`](apis/images.md) (optional extra from repo: `pip install -e ".[images]"`)
- Audio (STT/TTS): [`apis/audio.md`](apis/audio.md) (optional extras from repo: `pip install -e ".[stt]"`, `pip install -e ".[tts]"`)

## Operations

- Runtime configuration, logging, and safety notes: [`operations.md`](operations.md)
- Concurrency/runtime contract (MLX gate + threadpool): [`concurrency_contract.md`](concurrency_contract.md)

## Development

- Local dev workflow, formatting, and testing: [`development_guide.md`](development_guide.md)
- Local-only resources (paths to caches / reference repos): copy [`../config/local-resources.example.yaml`](../config/local-resources.example.yaml) → `../config/local-resources.yaml` (gitignored)

## Development plans

- Active / forward-looking plans: [`dev_plans/`](dev_plans/)
- Completed plans and historical research notes: [`archive/`](archive/)

## Architecture and roadmap

- Technical overview (implementation-oriented): [`code_analysis.md`](code_analysis.md)
- Architecture evaluation + prioritized improvement plan: [`architecture_evaluation.md`](architecture_evaluation.md)
- Stack rationale (timeless, constraint-driven): [`stack_assessment.md`](stack_assessment.md)
