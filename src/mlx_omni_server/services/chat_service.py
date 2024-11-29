from typing import AsyncGenerator

from ..schemas.chat_schema import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from .chat.models import BaseMLXModel


class ChatService:
    model: BaseMLXModel = None

    def __init__(self, model: BaseMLXModel):
        self.model = model

    async def generate_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Generate a chat completion."""
        return await self.model.generate(request)

    async def generate_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Generate a streaming chat completion."""
        async for chunk in self.model.stream_generate(request):
            yield chunk
