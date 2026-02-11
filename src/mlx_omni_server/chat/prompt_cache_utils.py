from __future__ import annotations

import struct
from hashlib import sha256


def common_prefix_len(a: list[int], b: list[int]) -> int:
    min_len = min(len(a), len(b))
    for i in range(min_len):
        if a[i] != b[i]:
            return i
    return min_len


def hash_tokens(tokens: list[int]) -> str:
    """Pack tokens as 4-byte little-endian ints then hash."""
    if not tokens:
        return "empty"
    packed = b"".join(struct.pack("<I", int(t)) for t in tokens)
    return sha256(packed).hexdigest()


def hash_tokens_with_media(tokens: list[int], media_hashes: list[str] | None = None) -> str:
    """Hash tokens and optionally incorporate media hashes (positional)."""
    if not tokens:
        return "empty"

    base_hash = hash_tokens(tokens)
    if media_hashes:
        media_hash_str = "|".join(media_hashes)
        combined = f"{base_hash}:{media_hash_str}"
        return sha256(combined.encode()).hexdigest()

    return base_hash
