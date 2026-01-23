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

Note: this repository uses dependency groups for dev tooling (tests/formatters). If you use `pip`, you may need to install dev tools separately (pytest, pre-commit, etc.).

## Run the server

### CLI entrypoint

```bash
mlx-omni-server --host 127.0.0.1 --port 10240
```

### Uvicorn (reload)

```bash
uvicorn mlx_omni_server.main:app --reload --host 127.0.0.1 --port 10240
```

## Formatting

Pre-commit is the expected formatting entrypoint:

```bash
pre-commit run --all-files
```

Configured hooks include Black (via Darker) and isort (Black profile). Line length is `100`.

## Testing

Run unit tests:

```bash
python3 -m pytest tests/unit
```

Run the full test suite (includes integration tests):

```bash
python3 -m pytest
```

Notes:

- Some integration tests may download models and can be slow/heavy on first run.
- Many tests assume macOS + Apple Silicon + MLX-capable dependencies.

## Implementation pointers

- API routers live under `src/mlx_omni_server/*/router.py` (or `images.py`, `stt.py`, `tts.py`) and are composed in `src/mlx_omni_server/routers.py`.
- Follow the runtime contract in `docs/concurrency_contract.md` when adding new MLX-backed work (threadpool offload + shared MLX gate).
- When adding optional modalities, keep imports safe: routes must remain registered and return `501` when an extra is not installed.
