# Operations

This document covers runtime configuration, logging, and operational constraints for running MLX Omni Server on a Mac.

## Run modes

### CLI entrypoint (recommended)

```bash
mlx-omni-server --host 127.0.0.1 --port 10240
```

### Uvicorn (development)

```bash
uvicorn mlx_omni_server.main:app --reload --host 127.0.0.1 --port 10240
```

## Configuration

The server is configured via CLI flags (see `mlx-omni-server --help`):

- `--host` (default `0.0.0.0`): bind address. Use `127.0.0.1` for localhost-only.
- `--port` (default `10240`): listen port.
- `--workers` (default `1`): process count. See “Multi-worker safety” below.
- `--log-level` (default `info`): `debug|info|warning|error|critical`.
- `--log-file` (default off): enable on-disk logging.
- `--log-dir`: directory for on-disk logs (default `~/Library/Logs/mlx-omni-server` on macOS).
- `--log-file-format` (default `jsonl`): `jsonl` or `text`.

## Logging

- Console logs use Rich formatting (optimized for interactive debugging).
- When `--log-file` is enabled, logs are written as rotating files under `--log-dir`.
  - Filenames are run-scoped and process-scoped: `mlx-omni-server-<run_id>-pid<PID>.(jsonl|log)`
  - Rotation defaults: ~20MB max per file, up to 5 backups.
- The request/response logging middleware logs:
  - request method/url + headers, plus a capped preview of textual request bodies,
  - response status/headers, plus a capped preview of textual response bodies.
  - For SSE (`text/event-stream`), binary media (for example `audio/*`), and attachments, it logs status/headers only (it does not buffer the full body).
  - When a body exceeds the cap, the preview includes both the head and tail (so end-of-payload issues are visible).

## Environment variables

- `MLX_OMNI_SERVER_REASONING_HMAC_KEY`
  - Used to sign and validate `reasoning.encrypted_content` tokens for the Responses API.
  - If unset, the server uses a per-process ephemeral key; tokens become invalid after a restart.
- Debugging artifacts (opt-in)
  - `MLX_OMNI_SERVER_LOG_ARTIFACTS` (bool): enable both HTTP-body and prompt artifacts.
  - `MLX_OMNI_SERVER_LOG_HTTP_BODY_ARTIFACTS` (bool): write full HTTP request/response bodies to artifacts.
  - `MLX_OMNI_SERVER_LOG_PROMPT_ARTIFACTS` (bool): write full formatted prompts to artifacts (and avoid huge inline prompt debug logs).
  - `MLX_OMNI_SERVER_LOG_ARTIFACTS_DIR`: override artifact directory (default: `<log_dir>/artifacts/<run_id>/`).
  - `MLX_OMNI_SERVER_LOG_ARTIFACTS_GZIP` (bool): gzip artifacts (`.gz`).
- Hugging Face (`huggingface-hub`)
  - Use standard variables like `HF_HOME`, `HF_TOKEN`, and `HF_HUB_ENABLE_HF_TRANSFER` as needed.

## Optional modalities (install extras)

These features are packaged as optional extras. When an extra is not installed, the routes remain registered but return `501 Not Implemented` with an install hint:

- Images (from repo): `pip install -e ".[images]"` (`/v1/images/generations`)
- Speech-to-text (from repo): `pip install -e ".[stt]"` (`/v1/audio/transcriptions`)
- Text-to-speech (from repo): `pip install -e ".[tts]"` (`/v1/audio/speech`)

### On-disk artifacts

- **Images**: if `response_format="url"` is used, images are written under the system temp dir (for example, `/tmp/mlx_omni_server/images/*.png`) and returned as `file://...` URLs. A background cleanup task removes URL-mode artifacts after a TTL.
- **TTS**: uses request-scoped temporary output directories; responses stream bytes back to the client.

## Multi-worker safety

`--workers > 1` (or `uvicorn --workers N`) spawns multiple processes. Each process has independent caches and independent “global” locks; for MLX/unified-memory workloads this can increase contention and memory spikes.

Prefer `--workers 1` unless you explicitly accept the tradeoffs (see `docs/concurrency_contract.md`).
