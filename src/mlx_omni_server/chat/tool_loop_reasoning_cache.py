from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class _CacheEntry:
    reasoning: str
    expires_at: float


class ToolLoopReasoningCache:
    """In-memory TTL+LRU cache for tool-loop reasoning replay.

    Keyed by `tool_call_id` so follow-up requests with `role="tool"` can
    restore missing assistant reasoning even if the client dropped it.
    """

    def __init__(self, *, ttl_seconds: float = 3600.0, max_entries: int = 1024) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")

        self._ttl_seconds = float(ttl_seconds)
        self._max_entries = int(max_entries)
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()

    def _now(self) -> float:
        return time.monotonic()

    def _evict(self, now: float) -> None:
        expired_keys: list[str] = []
        for key, entry in self._entries.items():
            if entry.expires_at <= now:
                expired_keys.append(key)
        for key in expired_keys:
            self._entries.pop(key, None)

        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def get(self, tool_call_id: str) -> str | None:
        if not tool_call_id:
            return None

        now = self._now()
        with self._lock:
            entry = self._entries.get(tool_call_id)
            if entry is None:
                self._evict(now)
                return None

            if entry.expires_at <= now:
                self._entries.pop(tool_call_id, None)
                self._evict(now)
                return None

            self._entries.move_to_end(tool_call_id)
            self._evict(now)
            return entry.reasoning

    def set(self, tool_call_id: str, reasoning: str) -> None:
        if not tool_call_id:
            return
        if not reasoning:
            return

        now = self._now()
        expires_at = now + self._ttl_seconds
        with self._lock:
            self._entries[tool_call_id] = _CacheEntry(reasoning=reasoning, expires_at=expires_at)
            self._entries.move_to_end(tool_call_id)
            self._evict(now)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


tool_loop_reasoning_cache = ToolLoopReasoningCache()
