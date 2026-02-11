from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

from fastapi import HTTPException


@dataclass(frozen=True, slots=True)
class OptionalExtra:
    name: str
    feature_label: str
    required_modules: Mapping[str, str]


_OPTIONAL_EXTRAS: dict[str, OptionalExtra] = {
    "images": OptionalExtra(
        name="images",
        feature_label="Image generation",
        required_modules={
            "mflux": "mflux",
        },
    ),
    "stt": OptionalExtra(
        name="stt",
        feature_label="Speech-to-text (STT)",
        required_modules={
            "multipart": "python-multipart",
            "mlx_whisper": "mlx-whisper",
        },
    ),
    "tts": OptionalExtra(
        name="tts",
        feature_label="Text-to-speech (TTS)",
        required_modules={
            "f5_tts_mlx": "f5-tts-mlx",
            "mlx_audio": "mlx-audio",
            "numba": "numba",
            "typing_extensions": "typing-extensions",
        },
    ),
}


def get_optional_extra(name: str) -> OptionalExtra:
    try:
        return _OPTIONAL_EXTRAS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown optional extra '{name}'. Known extras: {', '.join(sorted(_OPTIONAL_EXTRAS))}"
        ) from exc


def install_instructions(extra: str) -> str:
    return f"`pip install '.[{extra}]'`"


@lru_cache(maxsize=None)
def missing_packages(extra: str) -> tuple[str, ...]:
    optional_extra = get_optional_extra(extra)
    missing: list[str] = []
    for module_name, package_label in optional_extra.required_modules.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_label)
    return tuple(missing)


def is_available(extra: str) -> bool:
    return len(missing_packages(extra)) == 0


def not_installed_detail(extra: str, *, missing: tuple[str, ...] | None = None) -> str:
    optional_extra = get_optional_extra(extra)
    missing = missing if missing is not None else missing_packages(extra)
    missing_text = ", ".join(missing) if missing else "unknown"
    return (
        f"{optional_extra.feature_label} support is not installed. "
        f"Install optional dependencies with {install_instructions(extra)}. "
        f"Missing: {missing_text}"
    )


def ensure_extra_available(extra: str) -> None:
    missing = missing_packages(extra)
    if missing:
        raise HTTPException(
            status_code=501,
            detail=not_installed_detail(extra, missing=missing),
        )
