import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Iterable
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from ..chat.generation_service import chat_generation_service, make_request_hash
from ..chat.schema import ChatCompletionResponse, Role
from .adapter import (
    ResponseStreamAdapter,
    build_history_messages_for_next_request,
    build_response_dict,
    chat_messages_to_response_items,
    chat_response_to_response,
    response_request_to_chat_request,
)
from .registry import responses_registry
from .schema import ResponseRequest, ResponseResponse, ResponseStreamEvent

router = APIRouter(tags=["responses"])


def _format_sse_event(event: ResponseStreamEvent) -> str:
    return f"event: {event.event}\ndata: {json.dumps(event.data)}\n\n"


@router.get("/responses/{response_id}", response_model=ResponseResponse)
@router.get("/v1/responses/{response_id}", response_model=ResponseResponse)
async def retrieve_response(response_id: str, raw_request: Request):
    stream = raw_request.query_params.get("stream") in {"1", "true", "True"}
    starting_after = raw_request.query_params.get("starting_after")
    starting_after_int = (
        int(starting_after) if starting_after and starting_after.isdigit() else None
    )

    record = await responses_registry.get(response_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Response not found")

    if not stream:
        if record.response is None:
            raise HTTPException(status_code=404, detail="Response not available yet")
        return JSONResponse(content=record.response)

    async def event_stream() -> AsyncGenerator[str, None]:
        generator = await responses_registry.stream_events(
            response_id,
            starting_after=starting_after_int,
        )
        async for event in generator:
            yield _format_sse_event(event)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


@router.delete("/responses/{response_id}")
@router.delete("/v1/responses/{response_id}")
async def delete_response(response_id: str):
    deleted = await responses_registry.delete(response_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Response not found")
    return Response(status_code=204)


@router.post("/responses/{response_id}/cancel", response_model=ResponseResponse)
@router.post("/v1/responses/{response_id}/cancel", response_model=ResponseResponse)
async def cancel_response(response_id: str):
    record = await responses_registry.get(response_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Response not found")

    if isinstance(record.response, dict):
        status = record.response.get("status")
        if status in {"completed", "failed", "cancelled", "incomplete"}:
            return JSONResponse(content=record.response)

    if record.cancel_event is None:
        if record.response is not None:
            return JSONResponse(content=record.response)
        raise HTTPException(status_code=409, detail="Response cannot be cancelled")

    record.cancel_event.set()

    request_echo = dict(record.request)
    request_echo.setdefault("background", record.background)

    existing = record.response if isinstance(record.response, dict) else {}
    created_at = existing.get("created_at")
    if not isinstance(created_at, int):
        created_at = int(time.time())

    model = existing.get("model") or request_echo.get("model") or "unknown"
    output = existing.get("output") if isinstance(existing.get("output"), list) else []
    usage = existing.get("usage") if isinstance(existing.get("usage"), dict) else None

    cancelled_response = build_response_dict(
        response_id=response_id,
        created_at=created_at,
        model=model,
        status="cancelled",
        output=output,
        usage=usage,
        request_echo=request_echo,
    )
    await responses_registry.set_response(
        response_id, response=cancelled_response, status="cancelled"
    )
    return JSONResponse(content=cancelled_response)


@router.get("/responses/{response_id}/input_items")
@router.get("/v1/responses/{response_id}/input_items")
async def list_input_items(response_id: str, raw_request: Request):
    record = await responses_registry.get(response_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Response not found")

    items = chat_messages_to_response_items(record.input_messages, response_id=response_id)

    order = raw_request.query_params.get("order") or "desc"
    if order not in {"asc", "desc"}:
        raise HTTPException(status_code=400, detail="Invalid order")

    if order == "desc":
        items = list(reversed(items))

    after = raw_request.query_params.get("after")
    before = raw_request.query_params.get("before")
    limit_raw = raw_request.query_params.get("limit")
    limit = int(limit_raw) if limit_raw and limit_raw.isdigit() else 20
    limit = max(1, min(limit, 100))

    if after:
        for idx, item in enumerate(items):
            if item.get("id") == after:
                items = items[idx + 1 :]
                break

    if before:
        for idx, item in enumerate(items):
            if item.get("id") == before:
                items = items[:idx]
                break

    has_more = len(items) > limit
    page = items[:limit]

    return {"data": page, "has_more": has_more}


@router.post("/responses", response_model=ResponseResponse)
@router.post("/v1/responses", response_model=ResponseResponse)
async def create_response(request: ResponseRequest, raw_request: Request):
    """Handle /responses requests by delegating to the chat generation service."""
    request_dump = request.model_dump(exclude_none=True)
    logging.debug("Received responses request: %s", request_dump)

    chat_request = response_request_to_chat_request(request)

    if request.previous_response_id:
        prev_record = await responses_registry.get(request.previous_response_id)
        if prev_record is None or not prev_record.history_messages:
            raise HTTPException(status_code=404, detail="Previous response not found")

        messages = list(chat_request.messages)
        head: list[Any] = []
        tail = messages
        if (
            request.instructions
            and messages
            and messages[0].role == Role.SYSTEM
            and messages[0].content == request.instructions
        ):
            head = [messages[0]]
            tail = messages[1:]

        merged = head + prev_record.history_messages + tail
        chat_request = chat_request.model_copy(update={"messages": merged})

    if request.background:
        if chat_request.stream:
            raise HTTPException(
                status_code=400, detail="background is not supported with stream=true"
            )

        response_id = f"resp_{uuid4().hex}"
        created_at = int(time.time())

        cancel_event = asyncio.Event()

        request_echo = {**request_dump, "background": True}
        initial_response = build_response_dict(
            response_id=response_id,
            created_at=created_at,
            model=chat_request.model,
            status="queued",
            output=[],
            usage=None,
            request_echo=request_echo,
        )

        record = await responses_registry.create(
            response_id,
            request=request_dump,
            input_messages=[m.model_copy(deep=True) for m in chat_request.messages],
            instructions=request.instructions,
            background=True,
            store=request.store,
            cancel_event=cancel_event,
        )
        await responses_registry.set_response(
            response_id, response=initial_response, status="queued"
        )

        async def run_in_background() -> None:
            await responses_registry.append_events(
                response_id,
                [
                    ResponseStreamEvent(
                        event="response.created",
                        data={
                            "type": "response.created",
                            "sequence_number": 1,
                            "response": initial_response,
                        },
                    ),
                    ResponseStreamEvent(
                        event="response.queued",
                        data={
                            "type": "response.queued",
                            "sequence_number": 2,
                            "response": initial_response,
                        },
                    ),
                ],
            )

            in_progress = {**initial_response, "status": "in_progress"}
            await responses_registry.set_response(
                response_id, response=in_progress, status="in_progress"
            )
            await responses_registry.append_events(
                response_id,
                [
                    ResponseStreamEvent(
                        event="response.in_progress",
                        data={
                            "type": "response.in_progress",
                            "sequence_number": 3,
                            "response": in_progress,
                        },
                    )
                ],
            )

            async def should_cancel() -> bool:
                return cancel_event.is_set()

            result = await chat_generation_service.generate_non_stream(chat_request, should_cancel)
            payload = result.payload

            if isinstance(payload, ChatCompletionResponse) and not result.is_error:
                final_response = chat_response_to_response(
                    payload,
                    request_echo=request_echo,
                    response_id_override=response_id,
                )
                history_messages = build_history_messages_for_next_request(
                    input_messages=chat_request.messages,
                    instructions=request.instructions,
                    output_items=final_response.get("output", []),
                )
                await responses_registry.set_response(
                    response_id,
                    response=final_response,
                    history_messages=history_messages,
                )
                await responses_registry.append_events(
                    response_id,
                    [
                        ResponseStreamEvent(
                            event="response.completed",
                            data={
                                "type": "response.completed",
                                "sequence_number": 4,
                                "response": final_response,
                            },
                        )
                    ],
                )
                return

            if isinstance(payload, dict) and payload.get("error") == "Request cancelled":
                final_response = build_response_dict(
                    response_id=response_id,
                    created_at=created_at,
                    model=chat_request.model,
                    status="cancelled",
                    output=[],
                    usage=None,
                    request_echo=request_echo,
                )
            else:
                final_response = build_response_dict(
                    response_id=response_id,
                    created_at=created_at,
                    model=chat_request.model,
                    status="failed",
                    output=[],
                    usage=None,
                    request_echo=request_echo,
                    error={"code": "server_error", "message": str(payload)},
                )

            await responses_registry.set_response(response_id, response=final_response)
            await responses_registry.append_events(
                response_id,
                [
                    ResponseStreamEvent(
                        event="response.completed",
                        data={
                            "type": "response.completed",
                            "sequence_number": 4,
                            "response": final_response,
                        },
                    )
                ],
            )

        task = asyncio.create_task(run_in_background(), name=f"responses-bg-{response_id}")
        record.task = task

        return JSONResponse(content=initial_response)

    if not chat_request.stream:
        result = await chat_generation_service.generate_non_stream(
            chat_request,
            raw_request.is_disconnected,
        )
        payload = result.payload

        headers = {"X-Idempotent-Replay": "true"} if result.from_cache else {}
        status_code = 500 if result.is_error else 200

        if isinstance(payload, ChatCompletionResponse):
            response_dict = chat_response_to_response(payload, request_echo=request_dump)
            await responses_registry.create(
                response_dict["id"],
                request=request_dump,
                input_messages=[m.model_copy(deep=True) for m in chat_request.messages],
                instructions=request.instructions,
                background=bool(request.background),
                store=request.store,
            )
            history_messages = build_history_messages_for_next_request(
                input_messages=chat_request.messages,
                instructions=request.instructions,
                output_items=response_dict.get("output", []),
            )
            await responses_registry.set_response(
                response_dict["id"],
                response=response_dict,
                history_messages=history_messages,
            )
            created_response = build_response_dict(
                response_id=response_dict["id"],
                created_at=response_dict["created_at"],
                model=chat_request.model,
                status="in_progress",
                output=[],
                usage=None,
                request_echo=request_dump,
            )
            await responses_registry.append_events(
                response_dict["id"],
                [
                    ResponseStreamEvent(
                        event="response.created",
                        data={
                            "type": "response.created",
                            "sequence_number": 1,
                            "response": created_response,
                        },
                    ),
                    ResponseStreamEvent(
                        event="response.in_progress",
                        data={
                            "type": "response.in_progress",
                            "sequence_number": 2,
                            "response": created_response,
                        },
                    ),
                    ResponseStreamEvent(
                        event="response.completed",
                        data={
                            "type": "response.completed",
                            "sequence_number": 3,
                            "response": response_dict,
                        },
                    ),
                ],
            )
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

    provisional_id = make_request_hash(chat_request)
    await responses_registry.create(
        provisional_id,
        request=request_dump,
        input_messages=[m.model_copy(deep=True) for m in chat_request.messages],
        instructions=request.instructions,
        background=bool(request.background),
        store=request.store,
    )

    adapter = ResponseStreamAdapter(
        response_id=provisional_id,
        model=chat_request.model,
        request_echo=request_dump,
    )

    async def event_stream() -> AsyncGenerator[str, None]:
        response_id = provisional_id
        async for item in stream:
            events: Iterable[ResponseStreamEvent]
            if item.kind == "chunk":
                chunk = item.data
                if chunk.id:
                    adapter.set_response_id(chunk.id)
                    if chunk.id != response_id:
                        renamed = await responses_registry.rename(response_id, chunk.id)
                        if renamed is not None:
                            response_id = chunk.id
                events = adapter.on_chunk(chunk)
            elif item.kind == "error":
                events = adapter.on_error(item.data)
            elif item.kind == "done":
                events = adapter.on_done()
            else:
                events = []

            await responses_registry.append_events(response_id, events)
            for event in events:
                response_dict = event.data.get("response")
                if isinstance(response_dict, dict) and event.event in {
                    "response.created",
                    "response.queued",
                    "response.in_progress",
                    "response.completed",
                }:
                    await responses_registry.set_response(response_id, response=response_dict)
            for event in events:
                if event.event == "response.completed":
                    response_dict = event.data.get("response")
                    if isinstance(response_dict, dict):
                        history_messages = build_history_messages_for_next_request(
                            input_messages=chat_request.messages,
                            instructions=request.instructions,
                            output_items=response_dict.get("output", []),
                        )
                        await responses_registry.set_response(
                            response_id,
                            response=response_dict,
                            history_messages=history_messages,
                        )

            for event in events:
                yield _format_sse_event(event)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Idempotent-Replay": "true" if is_replay else "live",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)
