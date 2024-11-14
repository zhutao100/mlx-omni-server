import json
from typing import Generator

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from ...schemas.chat_schema import ChatCompletion, ChatCompletionRequest
from ...services.chat_service import ChatService

router = APIRouter(tags=["chat"])


@router.post("/chat/completions", response_model=ChatCompletion)
@router.post("/v1/chat/completions", response_model=ChatCompletion)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion"""
    chat_service = ChatService()

    if not request.stream:
        completion = await chat_service.generate_completion(request)
        return JSONResponse(content=completion.model_dump(exclude_none=True))

    async def event_generator() -> Generator[str, None, None]:
        async for chunk in chat_service.generate_stream(request):
            if chunk.choices[0].finish_reason == "stop":
                yield "data: [DONE]\n\n"
            else:
                yield f"data: {json.dumps(chunk.model_dump(exclude_none=True))}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
