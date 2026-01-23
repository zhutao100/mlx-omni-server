# AGENTS.md

This file is a short operating guide for LLM/agent work in this repository.

## Repo map (start here)

- Entry point: `src/mlx_omni_server/main.py` (FastAPI app + CLI args)
- Router aggregation: `src/mlx_omni_server/routers.py`
- Components:
  - Chat: `src/mlx_omni_server/chat/`
  - Responses adapter: `src/mlx_omni_server/responses/`
  - Embeddings: `src/mlx_omni_server/embeddings/`
  - Images (optional extra): `src/mlx_omni_server/images/`
  - Speech-to-text (optional extra): `src/mlx_omni_server/stt/`
  - Text-to-speech (optional extra): `src/mlx_omni_server/tts/`
- Shared runtime contract: `src/mlx_omni_server/inference/runtime.py`

## Docs to read first (by task type)

- Triage / “what is this project?”: `README.md` → `docs/README.md`
- API behavior questions: `docs/apis/` + matching router/service code under `src/mlx_omni_server/`
- Concurrency / responsiveness issues: `docs/concurrency_contract.md` + `src/mlx_omni_server/inference/runtime.py`
- Architecture / roadmap: `docs/code_analysis.md` and `docs/architecture_evaluation.md`
- Model guidance: `docs/supported_models.md`
- Deep internal notes: `docs/.llm_analysis/` (keep in sync if you change relevant code)

## Program constraints (still true)

- Target: macOS + Apple Silicon; optimized for local MLX inference.
- Trusted clients on localhost/LAN; not hardened for public internet exposure.
- Expected concurrency is low (typically 1–5); optimize for predictable latency.
- Prefer single-process by default (`--workers 1`) unless you explicitly accept the tradeoffs.

## Conventions and “source of truth”

- **API contracts**: the code + tests are authoritative. Treat docs as summaries that must match implementation.
- **Optional extras** (`images`, `stt`, `tts`): routes stay registered; if deps are missing, return `501` with an install hint (avoid import-time crashes).
- **Concurrency contract**: never block the event loop; run MLX-backed work via the shared runtime helpers (see `docs/concurrency_contract.md`).

## Workflows and commands

Use the repo’s virtualenv when available:

- Python commands: `PYENV_VERSION=venv313 pyenv exec python3 ...`
- Tests: `PYENV_VERSION=venv313 pyenv exec python3 -m pytest tests/unit`
  - `tests/integration/` may download models and can be slow/heavy.
- Formatting: `PYENV_VERSION=venv313 pyenv exec pre-commit run --all-files`
- Dev server (reload): `PYENV_VERSION=venv313 pyenv exec uvicorn mlx_omni_server.main:app --reload --port 10240`

## Local resources (developer machine)

- Core dependency repos (if you need to inspect upstream behavior):
  - `~/workspace/custom-builds/mlx-lm`
  - `~/workspace/custom-builds/mlx-vlm`
  - `~/workspace/custom-builds/transformers`
  - `~/workspace/custom-builds/mflux`
- Hugging Face model cache is typically under `~/.cache/huggingface/hub`.
