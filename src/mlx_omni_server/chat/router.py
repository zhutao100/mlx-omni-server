import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from .models.models import load_model
from .models.models_service import ModelId
from .schema import ChatCompletionRequest, ChatCompletionResponse
from .text_models import BaseTextModel

router = APIRouter(tags=["chat-completions"])


@dataclass
class StreamCacheEntry:
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    chunks: list[str] = field(default_factory=list)
    finished: bool = False
    generation_task: asyncio.Task | None = None
    created_at: float = field(default_factory=time.time)
    active_clients: int = 0


@dataclass
class NonStreamCacheEntry:
    payload: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    is_error: bool = False


response_cache: Dict[str, StreamCacheEntry | NonStreamCacheEntry] = {}
CACHE_TTL = 300
cache_lock = asyncio.Lock()
# Global lock to serialize MLX operations and prevent concurrent GPU access
mlx_lock = asyncio.Lock()


async def background_cache_cleanup():
    while True:
        await asyncio.sleep(60)
        try:
            cutoff = time.time() - CACHE_TTL
            async with cache_lock:
                for k in list(response_cache.keys()):
                    entry = response_cache[k]
                    if entry.created_at < cutoff:
                        if isinstance(entry, StreamCacheEntry) and entry.active_clients == 0:
                            if entry.generation_task and not entry.generation_task.done():
                                entry.generation_task.cancel()
                            del response_cache[k]
                            logging.debug(f"Cleaned up expired stream cache entry: {k}")
                        elif isinstance(entry, NonStreamCacheEntry):
                            del response_cache[k]
                            logging.debug(f"Cleaned up expired non-stream cache entry: {k}")
        except Exception as e:
            logging.error(f"Error in background cache cleanup: {e}", exc_info=True)


def make_request_hash(req: ChatCompletionRequest) -> str:
    dumped = req.model_dump(mode="json", exclude_none=True)
    raw = json.dumps(dumped, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _log_task_exception(task: asyncio.Task) -> None:
    """Done callback for generation tasks to log any unhandled exceptions."""
    if not task.cancelled() and (exc := task.exception()):
        logging.error(f"Background task {task.get_name()} failed", exc_info=exc)


@router.post("/chat/completions", response_model=ChatCompletionResponse)
@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """
    Create a chat completion.

    This endpoint handles both streaming and non-streaming requests.
    It uses an in-memory cache to handle idempotent requests and allow
    multiple clients to attach to the same streaming response.

    Blocking LLM generation is run in a thread pool to avoid blocking the
    server's event loop. All MLX operations are serialized to prevent
    concurrent GPU access issues.
    """
    logging.debug(f"Received chat completion request: {request.log_structured_request(verbose=True)}")
    req_hash = make_request_hash(request)
    logging.debug(f"Hash: {req_hash} for \n{request}\n")

    if not request.stream:
        async with cache_lock:
            cached_entry = response_cache.get(req_hash)
        if isinstance(cached_entry, NonStreamCacheEntry):
            logging.warning(f"Non-stream cache hit for {req_hash}")
            status_code = 500 if cached_entry.is_error else 200
            return JSONResponse(
                content=cached_entry.payload,
                headers={"X-Idempotent-Replay": "true"},
                status_code=status_code
            )

        logging.warning(f"Non-stream cache missed for {req_hash}")
        text_model = _create_text_model(
            request.model,
            request.get_extra_params().get("adapter_path"),
            request.get_extra_params().get("draft_model")
        )
        try:
            async with mlx_lock:
                completion = await run_in_threadpool(text_model.generate, request)
            payload = completion.model_dump(exclude_none=True)
            entry = NonStreamCacheEntry(payload=payload)
            async with cache_lock:
                response_cache[req_hash] = entry
            return JSONResponse(content=payload)
        except Exception as e:
            # POLICY: Cache failures to ensure idempotency. A request that fails once
            # will continue to fail for the TTL, preventing hammering the backend.
            logging.error(f"Error during non-streaming generation for {req_hash}: {e}", exc_info=True)
            error_payload = {"error": "Generation failed", "message": str(e)}
            entry = NonStreamCacheEntry(payload=error_payload, is_error=True)
            async with cache_lock:
                response_cache[req_hash] = entry
            return JSONResponse(content=error_payload, status_code=500)

    # --- Streaming Logic ---
    async with cache_lock:
        cached_entry = response_cache.get(req_hash)
        if not isinstance(cached_entry, StreamCacheEntry) or cached_entry.stop_event.is_set():
            cached_entry = StreamCacheEntry()
            text_model = _create_text_model(
                request.model,
                request.get_extra_params().get("adapter_path"),
                request.get_extra_params().get("draft_model")
            )
            loop = asyncio.get_running_loop()

            def run_blocking_generation():
                try:
                    for chunk in text_model.stream_generate(request):
                        if cached_entry.stop_event.is_set():
                            logging.info(f"Stopping generation for {req_hash} as all clients disconnected.")
                            break
                        # Format chunk as proper SSE data
                        chunk_data = chunk.model_dump(exclude_none=True)
                        # Log if this chunk has usage information
                        sse_data = f"data: {json.dumps(chunk_data)}\n\n"

                        async def notify():
                            async with cached_entry.condition:
                                cached_entry.chunks.append(sse_data)
                                cached_entry.condition.notify_all()
                        asyncio.run_coroutine_threadsafe(notify(), loop)
                except Exception as e:
                    error_data = {"error": "Generation failed", "message": str(e)}
                    sse_error = f"data: {json.dumps(error_data)}\n\n"

                    async def notify_error():
                        async with cached_entry.condition:
                            cached_entry.chunks.append(sse_error)
                            cached_entry.condition.notify_all()
                    asyncio.run_coroutine_threadsafe(notify_error(), loop)
                finally:
                    async def notify_done():
                        async with cached_entry.condition:
                            cached_entry.chunks.append("data: [DONE]\n\n")
                            cached_entry.finished = True
                            cached_entry.condition.notify_all()
                    asyncio.run_coroutine_threadsafe(notify_done(), loop)

            async def run_generation_task():
                async with mlx_lock:
                    await run_in_threadpool(run_blocking_generation)

            task = asyncio.create_task(run_generation_task(), name=f"generate-{req_hash}")
            task.add_done_callback(_log_task_exception)
            cached_entry.generation_task = task
            response_cache[req_hash] = cached_entry

    async def stream_generator() -> AsyncGenerator[str, None]:
        """
        Consumer that yields chunks from the cache. This logic is carefully
        structured to prevent race conditions when the stream finishes.
        """
        next_chunk_index = 0
        try:
            async with cached_entry.condition:
                cached_entry.active_clients += 1

            while True:
                new_chunks_to_yield = []

                # --- START: CRITICAL SECTION FIX ---
                # This logic ensures that even if the consumer wakes up to a 'finished'
                # state, it will always perform one final check for any chunks that
                # might have been added simultaneously with the 'finished' flag.
                async with cached_entry.condition:
                    # First, always check if there are new chunks available since our last run.
                    if next_chunk_index < len(cached_entry.chunks):
                        new_chunks_to_yield = cached_entry.chunks[next_chunk_index:]
                        next_chunk_index = len(cached_entry.chunks)

                    # If, after checking, we found no new chunks, then we decide
                    # whether to wait for more or to exit.
                    elif cached_entry.finished:
                        # The stream is finished AND we have processed all chunks.
                        # This is the only safe condition to break the loop.
                        break
                    else:
                        # The stream is not finished and there are no chunks. We must wait.
                        # We check for disconnection before waiting to exit promptly.
                        if await raw_request.is_disconnected():
                            break
                        await cached_entry.condition.wait()
                # --- END: CRITICAL SECTION FIX ---

                # Yielding happens outside the lock to prevent blocking other consumers.
                for chunk in new_chunks_to_yield:
                    yield chunk

                # After yielding, a final check for disconnection or completion.
                if await raw_request.is_disconnected():
                    logging.info("Client disconnected after yielding chunks.")
                    break

                # If we yielded the last batch and the stream is finished, we can exit now
                # instead of running the loop one more time.
                if cached_entry.finished and next_chunk_index == len(cached_entry.chunks):
                    break

        finally:
            async with cached_entry.condition:
                cached_entry.active_clients -= 1
                if cached_entry.active_clients == 0 and not cached_entry.finished:
                    cached_entry.stop_event.set()
                    logging.info(f"Last client for {req_hash} disconnected. Signaling generation to stop.")

    is_replay = cached_entry.finished
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Idempotent-Replay": "true" if is_replay else "live",
        }
    )


def _create_text_model(
    model_id: str,
    adapter_path: str | None = None,
    draft_model: str | None = None,
) -> BaseTextModel:
    """Create appropriate model based on whether it's a VLM or LM model."""

    model_id_obj = ModelId(name=model_id, adapter_path=adapter_path, draft_model=draft_model)
    return load_model(model_id_obj)
