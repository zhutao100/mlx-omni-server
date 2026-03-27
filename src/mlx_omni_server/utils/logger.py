from __future__ import annotations

import json
import logging
import logging.config
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.logging import RichHandler

LogFileFormat = Literal["text", "jsonl"]

_CONFIG_STATE: dict[str, Any] | None = None
_LOG_CONFIG: dict[str, Any] | None = None


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a named logger for the application (no side effects)."""
    return logging.getLogger(name or "mlx_omni_server")


logger = get_logger()


def _normalize_log_level(log_level: str) -> str:
    normalized = log_level.strip().upper()
    if normalized == "WARN":
        return "WARNING"
    return normalized


def default_log_dir() -> Path:
    """Return a robust default log directory for on-disk logging.

    Uses macOS' `~/Library/Logs` when available; otherwise uses XDG state.
    """
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "mlx-omni-server"

    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home) / "mlx-omni-server"

    return Path.home() / ".local" / "state" / "mlx-omni-server"


def _generate_run_id() -> str:
    # UTC, filename-safe, sortable.
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


_FALLBACK_RUN_ID = _generate_run_id()


def configured_log_dir() -> Path:
    """Return the configured log directory (or the default if unconfigured)."""
    if _CONFIG_STATE and "log_dir" in _CONFIG_STATE:
        try:
            return Path(str(_CONFIG_STATE["log_dir"])).expanduser()
        except Exception:
            return default_log_dir()
    return default_log_dir()


def configured_run_id() -> str:
    """Return the configured run id (or a per-process fallback if unconfigured)."""
    if _CONFIG_STATE and "run_id" in _CONFIG_STATE:
        return str(_CONFIG_STATE["run_id"])
    return _FALLBACK_RUN_ID


class OmniRichHandler(RichHandler):
    """Console handler optimized for real-time human inspection."""

    def __init__(self, **kwargs: Any) -> None:
        console = Console(highlight=False)
        super().__init__(
            console=console,
            show_time=False,
            show_level=True,
            show_path=False,
            enable_link_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_extra_lines=2,
            tracebacks_show_locals=False,
            **kwargs,
        )


_STANDARD_RECORD_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
    }
)


class JsonLineFormatter(logging.Formatter):
    """Emit one compact JSON object per log record (JSONL)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
            "function": record.funcName,
            "process": record.process,
            "thread": record.thread,
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        elif record.exc_text:
            payload["exception"] = record.exc_text

        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        extras = {
            key: value for key, value in record.__dict__.items() if key not in _STANDARD_RECORD_KEYS
        }
        if extras:
            payload["extra"] = _json_safe(extras)

        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class Iso8601TextFormatter(logging.Formatter):
    """Text formatter with ISO-8601 UTC timestamps."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()


class RunScopedRotatingFileHandler(RotatingFileHandler):
    """Rotating file handler with per-process filenames for multi-worker safety."""

    def __init__(
        self,
        *,
        log_dir: str | Path,
        filename_prefix: str,
        run_id: str,
        extension: str,
        max_bytes: int = 20_000_000,
        backup_count: int = 5,
        encoding: str = "utf-8",
        delay: bool = True,
    ) -> None:
        directory = Path(log_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)

        pid = os.getpid()
        filename = f"{filename_prefix}-{run_id}-pid{pid}.{extension}"
        path = directory / filename
        super().__init__(
            filename=str(path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding,
            delay=delay,
        )


def configure_logging(
    *,
    log_level: str = "info",
    log_file: bool = False,
    log_dir: str | Path | None = None,
    log_file_format: LogFileFormat = "jsonl",
) -> dict[str, Any]:
    """Configure process logging (idempotent) and return a uvicorn-compatible config."""

    global _CONFIG_STATE, _LOG_CONFIG

    effective_level = _normalize_log_level(log_level)
    effective_log_dir = Path(log_dir).expanduser() if log_dir is not None else default_log_dir()
    run_id = _CONFIG_STATE["run_id"] if _CONFIG_STATE else _generate_run_id()

    desired_state = {
        "log_level": effective_level,
        "log_file": bool(log_file),
        "log_dir": str(effective_log_dir),
        "log_file_format": log_file_format,
        "run_id": run_id,
    }
    if _CONFIG_STATE == desired_state and _LOG_CONFIG is not None:
        return _LOG_CONFIG

    handlers: list[str] = ["console"]
    handler_defs: dict[str, Any] = {
        "console": {
            "()": "mlx_omni_server.utils.logger.OmniRichHandler",
            "level": effective_level,
            "formatter": "console",
        }
    }

    formatters: dict[str, Any] = {
        "console": {"format": "%(message)s"},
        # Uvicorn expects these keys and will set `use_colors` dynamically.
        "default": {"format": "%(message)s"},
        "access": {"format": "%(message)s"},
        "text_file": {
            "()": "mlx_omni_server.utils.logger.Iso8601TextFormatter",
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s "
            "(%(module)s:%(lineno)d pid=%(process)d tid=%(thread)d)",
        },
        "jsonl_file": {"()": "mlx_omni_server.utils.logger.JsonLineFormatter"},
    }

    if log_file:
        handlers.append("file")
        extension = "jsonl" if log_file_format == "jsonl" else "log"
        file_formatter = "jsonl_file" if log_file_format == "jsonl" else "text_file"
        handler_defs["file"] = {
            "()": "mlx_omni_server.utils.logger.RunScopedRotatingFileHandler",
            "level": effective_level,
            "formatter": file_formatter,
            "log_dir": str(effective_log_dir),
            "filename_prefix": "mlx-omni-server",
            "run_id": run_id,
            "extension": extension,
        }

    log_config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handler_defs,
        "root": {"level": effective_level, "handlers": handlers},
        "loggers": {
            "uvicorn": {"level": effective_level, "handlers": [], "propagate": True},
            "uvicorn.error": {"level": effective_level, "handlers": [], "propagate": True},
            "uvicorn.access": {"level": effective_level, "handlers": [], "propagate": True},
            "uvicorn.asgi": {"level": effective_level, "handlers": [], "propagate": True},
        },
    }

    logging.config.dictConfig(log_config)
    _CONFIG_STATE = desired_state
    _LOG_CONFIG = log_config
    return log_config
