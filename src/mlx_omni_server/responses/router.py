import json
import logging
from typing import AsyncGenerator, Iterable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..chat.generation_service import chat_generation_service, make_request_hash
from ..chat.schema import ChatCompletionResponse
from .adapter import (
    ResponseStreamAdapter,
    chat_response_to_response,
    response_request_to_chat_request,
)
from .schema import ResponseRequest, ResponseResponse, ResponseStreamEvent

router = APIRouter(tags=["responses"])


def _format_sse_event(event: ResponseStreamEvent) -> str:
    return f"event: {event.event}\ndata: {json.dumps(event.data)}\n\n"


@router.post("/responses", response_model=ResponseResponse)
@router.post("/v1/responses", response_model=ResponseResponse)
async def create_response(request: ResponseRequest, raw_request: Request):
    """Handle /responses requests by delegating to the chat generation service."""
    logging.debug("Received responses request: %s", request.model_dump(exclude_none=True))

    chat_request = response_request_to_chat_request(request)

    if not chat_request.stream:
        result = await chat_generation_service.generate_non_stream(
            chat_request,
            raw_request.is_disconnected,
        )
        payload = result.payload

        headers = {"X-Idempotent-Replay": "true"} if result.from_cache else {}
        status_code = 500 if result.is_error else 200

        if isinstance(payload, ChatCompletionResponse):
            response_dict = chat_response_to_response(payload)
            return JSONResponse(
                content=response_dict,
                headers=headers,
                status_code=status_code,
            )

        return JSONResponse(content=payload, headers=headers, status_code=status_code)

    stream, is_replay = await chat_generation_service.stream_chat_completion(
        chat_request,
        raw_request.is_disconnected,
    )

    adapter = ResponseStreamAdapter(
        response_id=make_request_hash(chat_request),
        model=chat_request.model,
    )

    async def event_stream() -> AsyncGenerator[str, None]:
        async for item in stream:
            events: Iterable[ResponseStreamEvent]
            if item.kind == "chunk":
                chunk = item.data
                if chunk.id:
                    adapter.set_response_id(chunk.id)
                events = adapter.on_chunk(chunk)
            elif item.kind == "error":
                events = [adapter.on_error(item.data)]
            elif item.kind == "done":
                events = adapter.on_done()
            else:
                events = []

            for event in events:
                yield _format_sse_event(event)

            if item.kind == "done":
                yield "data: [DONE]\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Idempotent-Replay": "true" if is_replay else "live",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)
