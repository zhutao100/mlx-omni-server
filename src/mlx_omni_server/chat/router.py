import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional, AsyncGenerator
import hashlib
import json
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from .mlx.models import ModelId, load_model
from .schema import ChatCompletionRequest, ChatCompletionResponse
from .text_models import BaseTextModel


router = APIRouter(tags=["chat—completions"])

# In-memory cache (swap for Redis/Postgres in production)
response_cache: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 300  # 5 minutes


def make_request_hash(req: ChatCompletionRequest) -> str:
    """Create a stable hash from the request body."""
    dumped = req.model_dump(mode="json", exclude_none=True)
    raw = json.dumps(dumped, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


@router.post("/chat/completions", response_model=ChatCompletionResponse)
@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion"""

    req_hash = make_request_hash(request)

    # Clean old entries
    cutoff = time.time() - CACHE_TTL
    for k, v in list(response_cache.items()):
        if v["created_at"] < cutoff:
            del response_cache[k]

    # --- Step 1: replay if cached ---
    if req_hash in response_cache:
        cached = response_cache[req_hash]

        # If already finished, just replay cached content
        if cached["done"]:
            if cached["stream"]:
                async def replay_generator() -> AsyncGenerator[str, None]:
                    for chunk in cached["result"]:
                        yield chunk
                return StreamingResponse(
                    replay_generator(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Idempotent-Replay": "true",
                    },
                )
            else:
                return JSONResponse(
                    content=cached["result"],
                    headers={"X-Idempotent-Replay": "true"},
                )

        # If still streaming, attach to same live queue
        if cached["stream"] and not cached["done"]:
            queue: asyncio.Queue = cached["queue"]

            async def follower() -> AsyncGenerator[str, None]:
                while True:
                    item = await queue.get()
                    if item is None:  # Sentinel marks stream end
                        break
                    yield item

            return StreamingResponse(
                follower(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Idempotent-Replay": "live",
                },
            )

    # --- Step 2: create model instance ---
    text_model = _create_text_model(
        request.model,
        request.get_extra_params().get("adapter_path"),
        request.get_extra_params().get("draft_model"),
    )

    # --- Step 3a: non-streaming ---
    if not request.stream:
        completion = text_model.generate(request)
        payload = completion.model_dump(exclude_none=True)

        response_cache[req_hash] = {
            "result": payload,
            "created_at": time.time(),
            "stream": False,
            "done": True,
        }

        return JSONResponse(content=payload)

    # --- Step 3b: streaming ---
    queue: asyncio.Queue = asyncio.Queue()
    chunks: list[str] = []
    response_cache[req_hash] = {
        "result": chunks,
        "created_at": time.time(),
        "stream": True,
        "done": False,
        "queue": queue,
    }

    async def leader() -> AsyncGenerator[str, None]:
        try:
            for chunk in text_model.stream_generate(request):
                data = f"data: {json.dumps(chunk.model_dump(exclude_none=True))}\n\n"
                chunks.append(data)
                await queue.put(data)  # feed followers
                yield data

            done_marker = "data: [DONE]\n\n"
            chunks.append(done_marker)
            await queue.put(done_marker)
            yield done_marker
        finally:
            # signal end of stream
            await queue.put(None)
            response_cache[req_hash]["done"] = True
            # remove queue from cache (followers will use chunks afterwards)
            response_cache[req_hash].pop("queue", None)

    return StreamingResponse(
        leader(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def _create_text_model(
    model_id: str,
    adapter_path: Optional[str] = None,
    draft_model: Optional[str] = None,
) -> BaseTextModel:
    """Create a text model based on the model parameters.

    Creates a ModelId object and passes it to load_model function.
    The caching is handled inside the load_model function.
    """
    current_key = ModelId(
        name=model_id, adapter_path=adapter_path, draft_model=draft_model
    )

    return load_model(current_key)
