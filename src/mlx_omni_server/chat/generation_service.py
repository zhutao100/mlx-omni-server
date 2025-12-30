import asyncio
import contextlib
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Awaitable, Callable, Literal

from mlx_lm.generate import GenerationCancelled

from ..inference.runtime import run_mlx
from .models.models import load_model
from .models.models_service import ModelId
from .schema import ChatCompletionRequest, ChatCompletionResponse
from .text_models import BaseTextModel


@dataclass
class StreamItem:
    kind: Literal["chunk", "error", "done"]
    data: Any = None


@dataclass
class StreamCacheEntry:
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    stop_event: threading.Event = field(default_factory=threading.Event)
    items: list[StreamItem] = field(default_factory=list)
    finished: bool = False
    generation_task: asyncio.Task | None = None
    created_at: float = field(default_factory=time.time)
    active_clients: int = 0


@dataclass
class NonStreamCacheEntry:
    payload: Any
    created_at: float = field(default_factory=time.time)
    is_error: bool = False


@dataclass
class NonStreamResult:
    payload: Any
    is_error: bool
    from_cache: bool


response_cache: dict[str, StreamCacheEntry | NonStreamCacheEntry] = {}
CACHE_TTL = 300
cache_lock = asyncio.Lock()


def make_request_hash(req: ChatCompletionRequest) -> str:
    dumped = req.model_dump(mode="json", exclude_none=True)
    raw = json.dumps(dumped, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _create_text_model(
    model_id: str,
    adapter_path: str | None = None,
    draft_model: str | None = None,
) -> BaseTextModel:
    model_id_obj = ModelId(name=model_id, adapter_path=adapter_path, draft_model=draft_model)
    return load_model(model_id_obj)


class ChatGenerationService:
    def __init__(self) -> None:
        self._response_cache = response_cache
        self._cache_lock = cache_lock

    async def generate_non_stream(
        self,
        request: ChatCompletionRequest,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> NonStreamResult:
        req_hash = make_request_hash(request)

        async with self._cache_lock:
            cached_entry = self._response_cache.get(req_hash)
        if isinstance(cached_entry, NonStreamCacheEntry):
            return NonStreamResult(
                payload=cached_entry.payload,
                is_error=cached_entry.is_error,
                from_cache=True,
            )

        cancel_event = threading.Event()
        watch_task: asyncio.Task[None] | None = None
        if is_disconnected is not None:

            async def watch_disconnect() -> None:
                while not cancel_event.is_set():
                    if await is_disconnected():
                        cancel_event.set()
                        return
                    await asyncio.sleep(0.1)

            watch_task = asyncio.create_task(
                watch_disconnect(), name=f"watch-disconnect-{req_hash}"
            )

        try:
            adapter_path = request.get_extra_params().get("adapter_path")
            draft_model = request.get_extra_params().get("draft_model")

            def load_and_generate() -> ChatCompletionResponse:
                text_model = _create_text_model(
                    request.model,
                    adapter_path,
                    draft_model,
                )
                return text_model.generate(request, should_cancel=cancel_event.is_set)

            completion: ChatCompletionResponse = await run_mlx(load_and_generate)
            entry = NonStreamCacheEntry(payload=completion)
            async with self._cache_lock:
                self._response_cache[req_hash] = entry
            return NonStreamResult(payload=completion, is_error=False, from_cache=False)
        except GenerationCancelled:
            # Best-effort cancellation: never cache cancelled work.
            return NonStreamResult(
                payload={"error": "Request cancelled"},
                is_error=True,
                from_cache=False,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logging.error(
                "Error during non-streaming generation for %s: %s",
                req_hash,
                exc,
                exc_info=True,
            )
            error_payload = {"error": "Generation failed", "message": str(exc)}
            entry = NonStreamCacheEntry(payload=error_payload, is_error=True)
            async with self._cache_lock:
                self._response_cache[req_hash] = entry
            return NonStreamResult(payload=error_payload, is_error=True, from_cache=False)
        finally:
            cancel_event.set()
            if watch_task is not None:
                watch_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watch_task

    async def stream_chat_completion(
        self,
        request: ChatCompletionRequest,
        is_disconnected: Callable[[], Awaitable[bool]],
    ) -> tuple[AsyncGenerator[StreamItem, None], bool]:
        req_hash = make_request_hash(request)

        async with self._cache_lock:
            cached_entry = self._response_cache.get(req_hash)
            if not isinstance(cached_entry, StreamCacheEntry) or cached_entry.stop_event.is_set():
                cached_entry = await self._create_stream_entry(request, req_hash)
                self._response_cache[req_hash] = cached_entry

        is_replay = cached_entry.finished

        async def stream_generator() -> AsyncGenerator[StreamItem, None]:
            next_index = 0
            try:
                async with cached_entry.condition:
                    cached_entry.active_clients += 1

                while True:
                    new_items: list[StreamItem] = []
                    async with cached_entry.condition:
                        if next_index < len(cached_entry.items):
                            new_items = cached_entry.items[next_index:]
                            next_index = len(cached_entry.items)
                        elif cached_entry.finished:
                            break
                        else:
                            if await is_disconnected():
                                break
                            await cached_entry.condition.wait()

                    for item in new_items:
                        yield item

                    if await is_disconnected():
                        logging.info("Client disconnected after yielding items.")
                        break

                    if cached_entry.finished and next_index == len(cached_entry.items):
                        break
            finally:
                async with cached_entry.condition:
                    cached_entry.active_clients -= 1
                    if cached_entry.active_clients == 0 and not cached_entry.finished:
                        cached_entry.stop_event.set()
                        logging.info(
                            "Last client for %s disconnected. Signaling generation to stop.",
                            req_hash,
                        )

        return stream_generator(), is_replay

    async def _create_stream_entry(
        self, request: ChatCompletionRequest, req_hash: str
    ) -> StreamCacheEntry:
        cached_entry = StreamCacheEntry()
        adapter_path = request.get_extra_params().get("adapter_path")
        draft_model = request.get_extra_params().get("draft_model")
        loop = asyncio.get_running_loop()

        def run_blocking_generation() -> None:
            try:
                text_model = _create_text_model(
                    request.model,
                    adapter_path,
                    draft_model,
                )

                for chunk in text_model.stream_generate(
                    request,
                    should_cancel=cached_entry.stop_event.is_set,
                ):
                    stream_item = StreamItem(kind="chunk", data=chunk)

                    async def notify() -> None:
                        async with cached_entry.condition:
                            cached_entry.items.append(stream_item)
                            cached_entry.condition.notify_all()

                    future = asyncio.run_coroutine_threadsafe(notify(), loop)
                    future.result()
            except GenerationCancelled:
                logging.info(
                    "Generation for %s cancelled.",
                    req_hash,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logging.error(
                    "Error during streaming generation for %s: %s",
                    req_hash,
                    exc,
                    exc_info=True,
                )
                error_item = StreamItem(
                    kind="error",
                    data={"error": "Generation failed", "message": str(exc)},
                )

                async def notify_error() -> None:
                    async with cached_entry.condition:
                        cached_entry.items.append(error_item)
                        cached_entry.condition.notify_all()

                future = asyncio.run_coroutine_threadsafe(notify_error(), loop)
                future.result()
            finally:
                async def notify_done() -> None:
                    async with cached_entry.condition:
                        cached_entry.items.append(StreamItem(kind="done"))
                        cached_entry.finished = True
                        cached_entry.condition.notify_all()

                future = asyncio.run_coroutine_threadsafe(notify_done(), loop)
                future.result()

        async def run_generation_task() -> None:
            await run_mlx(run_blocking_generation)

        task = asyncio.create_task(
            run_generation_task(),
            name=f"generate-{req_hash}",
        )
        task.add_done_callback(self._log_task_exception)
        cached_entry.generation_task = task
        return cached_entry

    @staticmethod
    def _log_task_exception(task: asyncio.Task) -> None:
        if not task.cancelled() and (exc := task.exception()):
            logging.error(
                "Background task %s failed",
                task.get_name(),
                exc_info=exc,
            )

    async def background_cache_cleanup(self) -> None:
        while True:
            await asyncio.sleep(60)
            try:
                cutoff = time.time() - CACHE_TTL
                async with self._cache_lock:
                    for key in list(self._response_cache.keys()):
                        entry = self._response_cache[key]
                        if entry.created_at < cutoff:
                            if isinstance(entry, StreamCacheEntry) and entry.active_clients == 0:
                                if entry.generation_task and not entry.generation_task.done():
                                    entry.generation_task.cancel()
                                del self._response_cache[key]
                                logging.debug(
                                    "Cleaned up expired stream cache entry: %s",
                                    key,
                                )
                            elif isinstance(entry, NonStreamCacheEntry):
                                del self._response_cache[key]
                                logging.debug(
                                    "Cleaned up expired non-stream cache entry: %s",
                                    key,
                                )
            except Exception as exc:  # pylint: disable=broad-except
                logging.error(
                    "Error in background cache cleanup: %s",
                    exc,
                    exc_info=True,
                )


chat_generation_service = ChatGenerationService()


async def background_cache_cleanup() -> None:
    await chat_generation_service.background_cache_cleanup()
