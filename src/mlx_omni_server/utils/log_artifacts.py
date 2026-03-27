from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import IO

from .logger import configured_log_dir, configured_run_id, get_logger

_logger = get_logger(__name__)


def _env_bool(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def artifacts_enabled_http() -> bool:
    return _env_bool("MLX_OMNI_SERVER_LOG_ARTIFACTS", default=False) or _env_bool(
        "MLX_OMNI_SERVER_LOG_HTTP_BODY_ARTIFACTS",
        default=False,
    )


def artifacts_enabled_prompt() -> bool:
    return _env_bool("MLX_OMNI_SERVER_LOG_ARTIFACTS", default=False) or _env_bool(
        "MLX_OMNI_SERVER_LOG_PROMPT_ARTIFACTS",
        default=False,
    )


def artifacts_gzip_enabled() -> bool:
    return _env_bool("MLX_OMNI_SERVER_LOG_ARTIFACTS_GZIP", default=False)


def artifacts_root_dir() -> Path:
    configured = os.environ.get("MLX_OMNI_SERVER_LOG_ARTIFACTS_DIR")
    base_dir = Path(configured).expanduser() if configured else (configured_log_dir() / "artifacts")
    return base_dir / configured_run_id()


def redact_home_path(path: Path) -> str:
    home = Path.home()
    try:
        relative = path.expanduser().resolve().relative_to(home.resolve())
    except Exception:
        return str(path)
    return f"$HOME/{relative.as_posix()}"


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def open_artifact_writer(path: Path, *, gzip_enabled: bool) -> IO[bytes]:
    _ensure_parent_dir(path)
    if gzip_enabled:
        if not path.name.endswith(".gz"):
            path = path.with_name(path.name + ".gz")
        return gzip.open(path, "wb")
    return path.open("wb")


def write_artifact_bytes(path: Path, payload: bytes, *, gzip_enabled: bool) -> Path:
    try:
        with open_artifact_writer(path, gzip_enabled=gzip_enabled) as handle:
            handle.write(payload)
    except Exception:
        _logger.debug("Failed to write log artifact: %s", path, exc_info=True)
        raise
    return (
        path.with_name(path.name + ".gz")
        if gzip_enabled and not path.name.endswith(".gz")
        else path
    )


def http_request_artifact_path(*, request_id: str, extension: str) -> Path:
    return artifacts_root_dir() / f"{request_id}-http-request.{extension}"


def http_response_artifact_path(*, request_id: str, extension: str) -> Path:
    return artifacts_root_dir() / f"{request_id}-http-response.{extension}"


def prompt_artifact_path(*, request_id: str) -> Path:
    return artifacts_root_dir() / f"{request_id}-formatted-prompt.txt"
