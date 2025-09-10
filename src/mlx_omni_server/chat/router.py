import asyncio
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, AsyncGenerator
import hashlib
import json
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.concurrency import run_in_threadpool

from .mlx.models import ModelId, load_model
from .schema import ChatCompletionRequest, ChatCompletionResponse
from .text_models import BaseTextModel


router = APIRouter(tags=["chat-completions"])


@dataclass
class StreamCacheEntry:
    """Cache entry for a streaming response."""

    chunks: list[str] = field(default_factory=list)
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    error_event: asyncio.Event = field(default_factory=asyncio.Event)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    created_at: float = field(default_factory=time.time)
    active_clients: int = 0


@dataclass
class NonStreamCacheEntry:
    """Cache entry for a non-streaming response."""

    payload: Dict[str, Any]
    created_at: float = field(default_factory=time.time)


# In-memory cache (swap for Redis/Postgres in production)
response_cache: Dict[str, StreamCacheEntry | NonStreamCacheEntry] = {}
CACHE_TTL = 300  # 5 minutes
# Lock to prevent race conditions when creating cache entries
cache_lock = asyncio.Lock()
# Global lock to serialize MLX operations and prevent concurrent GPU access
mlx_lock = asyncio.Lock()
# Keep track of background tasks
background_tasks: set[asyncio.Task] = set()


async def background_cache_cleanup():
    """Background task to periodically clean up expired cache entries."""
    while True:
        try:
            await asyncio.sleep(60)  # Run every minute
            cutoff = time.time() - CACHE_TTL
            async with cache_lock:
                # Iterate over a copy of the keys to allow modification
                for k in list(response_cache.keys()):
                    if response_cache[k].created_at < cutoff:
                        # If it's a stream, ensure no clients are connected before cleaning up
                        if isinstance(response_cache[k], StreamCacheEntry):
                            if response_cache[k].active_clients == 0:  # type: ignore
                                del response_cache[k]
                                logging.debug(f"Cleaned up expired stream cache entry: {k}")
                        else:
                            del response_cache[k]
                            logging.debug(f"Cleaned up expired non-stream cache entry: {k}")
        except Exception as e:
            logging.error(f"Error in background cache cleanup: {e}")


def make_request_hash(req: ChatCompletionRequest) -> str:
    """Create a stable hash from the request body."""
    dumped = req.model_dump(mode="json", exclude_none=True)
    raw = json.dumps(dumped, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


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
    req_hash = make_request_hash(request)

    # --- Step 1: Handle non-streaming requests ---
    if not request.stream:
        async with cache_lock:
            cached_entry = response_cache.get(req_hash)
        if isinstance(cached_entry, NonStreamCacheEntry):
            return JSONResponse(
                content=cached_entry.payload,
                headers={"X-Idempotent-Replay": "true"},
            )

        text_model = _create_text_model(
            request.model,
            request.get_extra_params().get("adapter_path"),
            request.get_extra_params().get("draft_model"),
        )

        # Run blocking call in a thread pool with MLX serialization
        async with mlx_lock:
            completion = await run_in_threadpool(text_model.generate, request)
        payload = completion.model_dump(exclude_none=True)
        async with cache_lock:
            response_cache[req_hash] = NonStreamCacheEntry(payload=payload)
        return JSONResponse(content=payload)

    # --- Step 2: Handle streaming requests ---

    # Get or create the cache entry for this stream and check if generation should start
    should_start_generation = False
    async with cache_lock:
        cached_entry = response_cache.get(req_hash)
        if not isinstance(cached_entry, StreamCacheEntry):
            cached_entry = StreamCacheEntry()
            response_cache[req_hash] = cached_entry
            should_start_generation = True

    # The first request to arrive kicks off the generation
    if should_start_generation:
        text_model = _create_text_model(
            request.model,
            request.get_extra_params().get("adapter_path"),
            request.get_extra_params().get("draft_model"),
        )

        def run_blocking_generation():
            """The synchronous part that runs in a thread."""
            try:
                for chunk in text_model.stream_generate(request):
                    # Check if we should stop due to client disconnection
                    if cached_entry.stop_event.is_set():
                        logging.info(
                            f"Stopping generation for {req_hash} as all clients disconnected."
                        )
                        break
                    # Use actual newlines instead of escaped ones for proper SSE format
                    data = f"data: {json.dumps(chunk.model_dump(exclude_none=True))}\n\n"
                    cached_entry.chunks.append(data)
            except Exception as e:
                logging.error(f"Error during generation for {req_hash}: {e}")
                # Set error information
                error_data = f'data: {{"error": "Generation failed", "message": "{str(e)}"}}\n\n'
                cached_entry.chunks.append(error_data)
                cached_entry.error_event.set()
            finally:
                done_marker = "data: [DONE]\n\n"
                if not cached_entry.chunks or cached_entry.chunks[-1] != done_marker:
                    cached_entry.chunks.append(done_marker)
                cached_entry.done_event.set()

        # Create and track the background task with MLX serialization
        async def run_generation_task():
            async with mlx_lock:
                # Check if we should stop due to client disconnection
                if cached_entry.stop_event.is_set():
                    logging.info(
                        f"Stopping generation for {req_hash} as all clients disconnected."
                    )
                    return
                await run_in_threadpool(run_blocking_generation)

        task = asyncio.create_task(run_generation_task())
        background_tasks.add(task)
        # Remove task from set when it's done to prevent memory leaks
        task.add_done_callback(background_tasks.discard)

    async def stream_generator() -> AsyncGenerator[str, None]:
        """
        Yields chunks from the cache, tracking client connection status.
        """
        next_chunk_index = 0
        try:
            # Register client
            async with cache_lock:
                cached_entry.active_clients += 1

            while True:
                # Yield all chunks that are currently available
                while next_chunk_index < len(cached_entry.chunks):
                    if await raw_request.is_disconnected():
                        return  # Exit quietly
                    yield cached_entry.chunks[next_chunk_index]
                    next_chunk_index += 1

                # If the stream is done, we've sent all chunks, so we can exit
                if cached_entry.done_event.is_set():
                    break

                # If there was an error, we've already sent the error message, so we can exit
                if cached_entry.error_event.is_set():
                    break

                # Wait for the 'done' event to be set.
                try:
                    await asyncio.wait_for(cached_entry.done_event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

        finally:
            # Unregister client and check if generation should be stopped
            async with cache_lock:
                cached_entry.active_clients -= 1
                if cached_entry.active_clients == 0 and not cached_entry.done_event.is_set():
                    cached_entry.stop_event.set()
                    logging.info(
                        f"Last client for {req_hash} disconnected. Signaling generation to stop."
                    )

    is_live = not cached_entry.done_event.is_set() and not cached_entry.error_event.is_set()
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "max-age=300",
            "Connection": "keep-alive",
            "X-Idempotent-Replay": "live" if is_live else "true",
        },
    )


def _create_text_model(
    model_id: str,
    adapter_path: str | None = None,
    draft_model: str | None = None,
) -> BaseTextModel:
    """Create a text model based on the model parameters.

    Creates a ModelId object and passes it to load_model function.
    The caching is handled inside the load_model function.
    """
    current_key = ModelId(
        name=model_id, adapter_path=adapter_path, draft_model=draft_model
    )

    return load_model(current_key)
