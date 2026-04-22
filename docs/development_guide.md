# Development Guide

This guide is for contributors working on the server implementation (routers/services/runtime) and keeping docs/tests in sync.

## Setup

### Option A: `uv` (recommended for reproducible dev installs)

```bash
uv sync --all-extras --dev
uv run mlx-omni-server --help
```

### Option B: `pip` (simple editable install)

```bash
python3 -m pip install -e ".[all]"
```

### Option C: `pyenv` (repo-local workflow)

If you have a local `pyenv` environment configured for this repo, record the env name in `config/local-resources.yaml` (`local_envs.pyenv_env_name`) and run commands via `pyenv exec`:

```bash
pyenv exec python3 -m pytest tests/unit
```

Note: this repository uses dependency groups for dev tooling (tests/formatters). If you use `pip`, you may need to install dev tools separately (pytest etc.).

### Local-only developer config (paths and reference repos)

If you need to reference machine-local resources (Hugging Face cache location, local clones of upstream repos, etc.), record them in `config/local-resources.yaml` (gitignored; template: `config/local-resources.example.yaml`):

```bash
cp config/local-resources.example.yaml config/local-resources.yaml
```

Use `config/local-resources.yaml` as the source of truth for *local* paths during implementation/research, but keep docs/tests/patches portable (prefer `$HOME/...` or relative paths). If the file is missing, create it from the template rather than guessing locations.

This file is for humans/agents only; the server does not read it.

## Run the server

### CLI entrypoint

```bash
mlx-omni-server --host 127.0.0.1 --port 10240
```

### Uvicorn (reload)

```bash
uvicorn mlx_omni_server.main:app --reload --host 127.0.0.1 --port 10240
```

## Testing

Run unit tests:

```bash
python3 -m pytest tests/unit -q
```

Run the full test suite (includes integration tests):

```bash
python3 -m pytest -q
```

Notes:

- Some integration tests may download models and can be slow/heavy on first run.
- Many tests assume macOS + Apple Silicon + MLX-capable dependencies.

## Implementation pointers

- API routers live under `src/mlx_omni_server/*/router.py` (or `images.py`, `stt.py`, `tts.py`) and are composed in `src/mlx_omni_server/routers.py`.
- Follow the runtime contract in `docs/concurrency_contract.md` when adding new MLX-backed work (threadpool offload + shared MLX gate).
- When adding optional modalities, keep imports safe: routes must remain registered and return `501` when an extra is not installed.
