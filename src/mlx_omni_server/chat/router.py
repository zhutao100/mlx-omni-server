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
    """
    Cache entry for streaming chat completion responses.

    Manages the state of an ongoing streaming generation, allowing multiple
    clients to connect to the same stream and providing synchronization
    for thread-safe access to chunks and state changes.
    """
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    chunks: list[str] = field(default_factory=list)
    finished: bool = False
    generation_task: asyncio.Task | None = None
    created_at: float = field(default_factory=time.time)
    active_clients: int = 0


@dataclass
class NonStreamCacheEntry:
    """
    Cache entry for non-streaming chat completion responses.

    Stores the complete response payload for idempotent non-streaming requests,
    including error states to ensure consistent behavior for failed requests.
    """
    payload: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    is_error: bool = False


# Global cache storing response entries keyed by request hash
# Entries can be either streaming or non-streaming responses
response_cache: Dict[str, StreamCacheEntry | NonStreamCacheEntry] = {}

# Cache Time-To-Live in seconds (5 minutes)
# Determines how long cached responses remain valid before cleanup
CACHE_TTL = 300

# Lock for thread-safe access to the response cache
# Prevents race conditions when reading/writing cache entries
cache_lock = asyncio.Lock()

# Global lock to serialize MLX operations and prevent concurrent GPU access
mlx_lock = asyncio.Lock()


async def background_cache_cleanup():
    """
    Background task that periodically cleans up expired cache entries.

    Runs every 60 seconds and removes cache entries that have exceeded
    the Time-To-Live (TTL). For streaming entries, ensures no active
    clients are present before cleanup to prevent interrupting ongoing streams.
    """
    while True:
        await asyncio.sleep(60)
        try:
            cutoff = time.time() - CACHE_TTL
            async with cache_lock:
                for k in list(response_cache.keys()):
                    entry = response_cache[k]
                    if entry.created_at < cutoff:
                        if isinstance(entry, StreamCacheEntry) and entry.active_clients == 0:
                            # Cancel any ongoing generation task before cleanup
                            if entry.generation_task and not entry.generation_task.done():
                                entry.generation_task.cancel()
                            del response_cache[k]
                            logging.debug(f"Cleaned up expired stream cache entry: {k}")
                        elif isinstance(entry, NonStreamCacheEntry):
                            # Non-stream entries can be safely removed
                            del response_cache[k]
                            logging.debug(f"Cleaned up expired non-stream cache entry: {k}")
        except Exception as e:
            logging.error(f"Error in background cache cleanup: {e}", exc_info=True)


def make_request_hash(req: ChatCompletionRequest) -> str:
    """
    Generate a unique hash for a chat completion request.

    Creates a deterministic SHA256 hash from the request parameters,
    excluding None values. This hash is used as a cache key to enable
    idempotent requests and allow multiple clients to share the same
    streaming response.

    Args:
        req: The ChatCompletionRequest to hash

    Returns:
        SHA256 hash string representing the request parameters
    """
    dumped = req.model_dump(mode="json", exclude_none=True)
    raw = json.dumps(dumped, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _log_task_exception(task: asyncio.Task) -> None:
    """
    Error logging callback for background generation tasks.

    Attached to asyncio tasks to catch and log any unhandled exceptions
    that occur during background generation, preventing silent failures.

    Args:
        task: The asyncio.Task that may have encountered an exception
    """
    if not task.cancelled() and (exc := task.exception()):
        logging.error(f"Background task {task.get_name()} failed", exc_info=exc)


@router.post("/chat/completions", response_model=ChatCompletionResponse)
@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """
    Create a chat completion with support for both streaming and non-streaming responses.

    This endpoint provides OpenAI-compatible chat completion functionality with enhanced
    features including:
    - Idempotent request handling via request hashing
    - Response caching for improved performance
    - Multi-client streaming support for the same request
    - Thread-safe MLX operations to prevent GPU conflicts

    The endpoint intelligently routes requests to either cached responses or new generation
    based on the request hash. For streaming requests, multiple clients can connect to
    the same ongoing generation, sharing the response stream.

    Args:
        request: ChatCompletionRequest containing the model, messages, and generation parameters
        raw_request: FastAPI Request object for checking client connection status

    Returns:
        JSONResponse for non-streaming requests or StreamingResponse for streaming requests
        with appropriate headers and caching metadata
    """
    logging.debug(f"Received chat completion request: {request.log_structured_request(verbose=True)}")
    req_hash = make_request_hash(request)

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
    # Check for existing streaming cache or create new entry
    async with cache_lock:
        cached_entry = response_cache.get(req_hash)
        if not isinstance(cached_entry, StreamCacheEntry) or cached_entry.stop_event.is_set():
            # Initialize new streaming cache entry
            cached_entry = StreamCacheEntry()
            text_model = _create_text_model(
                request.model,
                request.get_extra_params().get("adapter_path"),
                request.get_extra_params().get("draft_model")
            )
            loop = asyncio.get_running_loop()

            # Thread-local generation function to avoid blocking event loop
            def run_blocking_generation():
                try:
                    for chunk in text_model.stream_generate(request):
                        # Check if all clients disconnected and stop generation
                        if cached_entry.stop_event.is_set():
                            logging.info(f"Stopping generation for {req_hash} as all clients disconnected.")
                            break

                        # Format chunk as Server-Sent Events (SSE) data
                        chunk_data = chunk.model_dump(exclude_none=True)
                        sse_data = f"data: {json.dumps(chunk_data)}\n\n"

                        # Thread-safe notification of new chunk availability
                        async def notify():
                            async with cached_entry.condition:
                                cached_entry.chunks.append(sse_data)
                                cached_entry.condition.notify_all()
                        future = asyncio.run_coroutine_threadsafe(notify(), loop)
                        future.result()
                except Exception as e:
                    # Handle generation errors by notifying all waiting clients
                    error_data = {"error": "Generation failed", "message": str(e)}
                    sse_error = f"data: {json.dumps(error_data)}\n\n"

                    async def notify_error():
                        # POLICY: Cache failures to ensure idempotency. A request that fails once
                        # will continue to fail for the TTL, preventing hammering the backend.
                        async with cached_entry.condition:
                            cached_entry.chunks.append(sse_error)
                            cached_entry.condition.notify_all()
                    future = asyncio.run_coroutine_threadsafe(notify_error(), loop)
                    future.result()
                finally:
                    # Signal completion to all waiting clients
                    async def notify_done():
                        async with cached_entry.condition:
                            cached_entry.chunks.append("data: [DONE]\n\n")
                            cached_entry.finished = True
                            cached_entry.condition.notify_all()
                    future = asyncio.run_coroutine_threadsafe(notify_done(), loop)
                    future.result()

            # Wrap blocking generation in async task with MLX serialization
            async def run_generation_task():
                async with mlx_lock:
                    await run_in_threadpool(run_blocking_generation)

            # Create background task for generation with error logging
            task = asyncio.create_task(run_generation_task(), name=f"generate-{req_hash}")
            task.add_done_callback(_log_task_exception)
            cached_entry.generation_task = task
            response_cache[req_hash] = cached_entry

    async def stream_generator() -> AsyncGenerator[str, None]:
        """
        Async generator that yields SSE-formatted chunks from the streaming cache.

        This consumer implements careful synchronization to prevent race conditions
        when the stream finishes. It handles multiple scenarios:
        - New chunks arriving while the consumer is processing
        - Stream completion detection
        - Client disconnection handling
        - Thread-safe access to shared cache state

        Yields:
            SSE-formatted string chunks (data: {json}\n\n)
        """
        next_chunk_index = 0
        try:
            async with cached_entry.condition:
                cached_entry.active_clients += 1

            while True:
                new_chunks_to_yield = []

                # Critical section to prevent race conditions between chunk availability and stream completion
                async with cached_entry.condition:
                    # Always check for new chunks first - prevents missing chunks added
                    # simultaneously with the 'finished' flag being set
                    if next_chunk_index < len(cached_entry.chunks):
                        new_chunks_to_yield = cached_entry.chunks[next_chunk_index:]
                        next_chunk_index = len(cached_entry.chunks)
                    elif cached_entry.finished:
                        # Safe exit: stream finished AND all chunks processed
                        break
                    else:
                        # Wait for new chunks or disconnection
                        if await raw_request.is_disconnected():
                            break
                        await cached_entry.condition.wait()

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
            # Clean up client count and signal stop if last client disconnects
            async with cached_entry.condition:
                cached_entry.active_clients -= 1
                if cached_entry.active_clients == 0 and not cached_entry.finished:
                    # Stop generation when last client disconnects before completion
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
    """
    Factory function to create the appropriate text model instance.

    Determines and instantiates the correct model type based on the provided
    parameters, including support for adapter paths and draft models for
    enhanced generation capabilities.

    Args:
        model_id: Name or identifier of the base model to load
        adapter_path: Optional path to model adapter/LoRA weights
        draft_model: Optional draft model for speculative decoding

    Returns:
        BaseTextModel instance ready for text generation
    """
    model_id_obj = ModelId(name=model_id, adapter_path=adapter_path, draft_model=draft_model)
    return load_model(model_id_obj)
