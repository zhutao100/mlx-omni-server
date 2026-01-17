import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .generation_service import chat_generation_service
from .schema import ChatCompletionRequest, ChatCompletionResponse

router = APIRouter(tags=["chat-completions"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """Create chat completion responses with caching and streaming support."""
    logging.debug(
        "Received chat completion request: %s",
        request.log_structured_request(verbose=True),
    )

    if request.tools and any(getattr(tool, "type", None) == "web_search" for tool in request.tools):
        message = "The /v1/chat/completions endpoint does not support tools of type 'web_search'."
        logging.warning(message)
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": "tools",
                    "code": "invalid_request",
                }
            },
        )

    if not request.stream:
        result = await chat_generation_service.generate_non_stream(
            request,
            raw_request.is_disconnected,
        )
        payload = result.payload
        if isinstance(payload, ChatCompletionResponse):
            body = payload.model_dump(exclude_none=True)
        else:
            body = payload

        headers = {"X-Idempotent-Replay": "true"} if result.from_cache else {}
        status_code = 500 if result.is_error else 200
        return JSONResponse(content=body, headers=headers, status_code=status_code)

    stream, is_replay = await chat_generation_service.stream_chat_completion(
        request,
        raw_request.is_disconnected,
    )

    async def sse_generator() -> AsyncGenerator[str, None]:
        async for item in stream:
            if item.kind == "chunk":
                chunk = item.data
                yield f"data: {json.dumps(chunk.model_dump(exclude_none=True))}\n\n"
            elif item.kind == "error":
                yield f"data: {json.dumps(item.data)}\n\n"
            elif item.kind == "done":
                yield "data: [DONE]\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Idempotent-Replay": "true" if is_replay else "live",
    }
    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers=headers,
    )
