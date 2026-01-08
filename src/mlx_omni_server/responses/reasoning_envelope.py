from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import zlib
from typing import Any

from pydantic import BaseModel, Field

_ENV_HMAC_KEY = "MLX_OMNI_SERVER_REASONING_HMAC_KEY"
_TOKEN_VERSION = "v1"
_ephemeral_hmac_key: bytes | None = None


class ReasoningEnvelope(BaseModel):
    v: int = 1
    model: str
    created_at: int
    tool_call_ids: list[str] = Field(default_factory=list)
    reasoning: str


def _get_hmac_key() -> bytes:
    configured = os.getenv(_ENV_HMAC_KEY)
    if configured:
        return configured.encode("utf-8")

    global _ephemeral_hmac_key
    if _ephemeral_hmac_key is None:
        _ephemeral_hmac_key = secrets.token_bytes(32)
    return _ephemeral_hmac_key


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padded = data + ("=" * (-len(data) % 4))
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def seal(envelope: ReasoningEnvelope | dict[str, Any]) -> str:
    payload = (
        envelope.model_dump()
        if isinstance(envelope, ReasoningEnvelope)
        else ReasoningEnvelope.model_validate(envelope).model_dump()
    )
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    compressed = zlib.compress(raw)

    signature = hmac.new(_get_hmac_key(), compressed, hashlib.sha256).digest()
    return ".".join(
        (
            _TOKEN_VERSION,
            _b64url_encode(compressed),
            _b64url_encode(signature),
        )
    )


def unseal(token: str) -> ReasoningEnvelope:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid reasoning envelope format")

    version, payload_b64, sig_b64 = parts
    if version != _TOKEN_VERSION:
        raise ValueError("Unsupported reasoning envelope version")

    compressed = _b64url_decode(payload_b64)
    signature = _b64url_decode(sig_b64)
    expected = hmac.new(_get_hmac_key(), compressed, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("Invalid reasoning envelope signature")

    try:
        raw = zlib.decompress(compressed)
    except zlib.error as exc:
        raise ValueError("Invalid reasoning envelope payload") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid reasoning envelope JSON") from exc

    return ReasoningEnvelope.model_validate(payload)
