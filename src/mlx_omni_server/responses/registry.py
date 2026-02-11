from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Iterable

from ..chat.schema import ChatMessage
from .schema import ResponseStreamEvent

DEFAULT_RESPONSE_TTL_SECONDS = 60 * 60


@dataclass(slots=True)
class ResponseRecord:
    response_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    request: dict[str, Any] = field(default_factory=dict)
    input_messages: list[ChatMessage] = field(default_factory=list)
    history_messages: list[ChatMessage] = field(default_factory=list)
    response: dict[str, Any] | None = None
    status: str = "in_progress"
    background: bool = False
    store: bool | None = None
    instructions: str | None = None
    cancel_event: asyncio.Event | None = None
    task: asyncio.Task[None] | None = None
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    events: list[ResponseStreamEvent] = field(default_factory=list)
    finished: bool = False
    ttl_seconds: int = DEFAULT_RESPONSE_TTL_SECONDS

    @property
    def expires_at(self) -> float:
        return self.updated_at + self.ttl_seconds

    def touch(self) -> None:
        self.updated_at = time.time()


class ResponseRegistry:
    def __init__(self, *, default_ttl_seconds: int = DEFAULT_RESPONSE_TTL_SECONDS) -> None:
        self._records: dict[str, ResponseRecord] = {}
        self._lock = asyncio.Lock()
        self._default_ttl_seconds = default_ttl_seconds

    async def create(
        self,
        response_id: str,
        *,
        request: dict[str, Any],
        input_messages: list[ChatMessage],
        instructions: str | None,
        background: bool,
        store: bool | None,
        cancel_event: asyncio.Event | None = None,
        task: asyncio.Task[None] | None = None,
    ) -> ResponseRecord:
        async with self._lock:
            self._prune_locked()
            record = ResponseRecord(
                response_id=response_id,
                request=request,
                input_messages=input_messages,
                instructions=instructions,
                background=background,
                store=store,
                cancel_event=cancel_event,
                task=task,
                ttl_seconds=self._default_ttl_seconds,
            )
            self._records[response_id] = record
            return record

    async def rename(self, old_id: str, new_id: str) -> ResponseRecord | None:
        async with self._lock:
            record = self._records.pop(old_id, None)
            if record is None:
                return None
            record.response_id = new_id
            record.touch()
            self._records[new_id] = record
            return record

    async def get(self, response_id: str) -> ResponseRecord | None:
        async with self._lock:
            self._prune_locked()
            record = self._records.get(response_id)
            if record is None:
                return None
            record.touch()
            return record

    async def delete(self, response_id: str) -> bool:
        async with self._lock:
            record = self._records.pop(response_id, None)
            if record is None:
                return False
            record.touch()
            if record.cancel_event is not None:
                record.cancel_event.set()
            if record.task is not None and not record.task.done():
                record.task.cancel()
            return True

    async def append_events(self, response_id: str, events: Iterable[ResponseStreamEvent]) -> None:
        record = await self.get(response_id)
        if record is None:
            return

        async with record.condition:
            for event in events:
                record.events.append(event)
                if event.event == "response.completed":
                    record.finished = True
            record.touch()
            record.condition.notify_all()

    async def set_response(
        self,
        response_id: str,
        *,
        response: dict[str, Any] | None,
        status: str | None = None,
        history_messages: list[ChatMessage] | None = None,
    ) -> None:
        record = await self.get(response_id)
        if record is None:
            return

        if response is not None:
            record.response = response
            record.status = response.get("status") or record.status
        if status is not None:
            record.status = status
        if history_messages is not None:
            record.history_messages = history_messages
        record.touch()

    async def wait_until_finished(
        self, response_id: str, *, timeout_seconds: float | None = None
    ) -> bool:
        record = await self.get(response_id)
        if record is None:
            return False

        if record.finished:
            return True

        deadline = None
        if timeout_seconds is not None:
            deadline = time.monotonic() + timeout_seconds

        while True:
            if record.finished:
                return True

            remaining = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False

            # Periodically wake to detect deletion/pruning without busy-polling.
            wait_timeout = 1.0 if remaining is None else min(remaining, 1.0)

            async with record.condition:
                if record.finished:
                    return True
                try:
                    await asyncio.wait_for(record.condition.wait(), timeout=wait_timeout)
                except asyncio.TimeoutError:
                    pass

            # Refresh record (also touches TTL). If removed, treat as missing.
            record = await self.get(response_id)
            if record is None:
                return False

    async def stream_events(
        self,
        response_id: str,
        *,
        starting_after: int | None = None,
    ) -> AsyncGenerator[ResponseStreamEvent, None]:
        record = await self.get(response_id)
        if record is None:

            async def empty() -> AsyncGenerator[ResponseStreamEvent, None]:
                if False:  # pragma: no cover
                    yield ResponseStreamEvent(event="", data={})

            return empty()

        index = 0
        if starting_after is not None:
            for idx, event in enumerate(record.events):
                seq = event.data.get("sequence_number")
                if isinstance(seq, int) and seq > starting_after:
                    index = idx
                    break
            else:
                index = len(record.events)

        async def generator() -> AsyncGenerator[ResponseStreamEvent, None]:
            nonlocal index
            while True:
                async with record.condition:
                    while index >= len(record.events) and not record.finished:
                        await record.condition.wait()

                    if index < len(record.events):
                        event = record.events[index]
                        index += 1
                        yield event
                        continue

                    if record.finished:
                        break

        return generator()

    def _prune_locked(self) -> None:
        now = time.time()
        for key in list(self._records.keys()):
            record = self._records[key]
            if record.expires_at < now:
                self._records.pop(key, None)


responses_registry = ResponseRegistry()
