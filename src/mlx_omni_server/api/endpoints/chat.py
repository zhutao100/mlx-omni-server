import json
from typing import Generator

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from ...schemas.chat_schema import ChatCompletionRequest, ChatCompletionResponse
from ...services.chat.models import load_model
from ...services.chat_service import ChatService

router = APIRouter(tags=["chat—completions"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion"""

    chat_service = _create_chat_service(request.model)

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


def _create_chat_service(model_id: str):
    model = load_model(model_id)
    return ChatService(model)
