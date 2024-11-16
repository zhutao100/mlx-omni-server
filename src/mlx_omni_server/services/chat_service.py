import time
import uuid
from typing import AsyncGenerator

from ..schemas.chat_schema import (
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionUsage,
    Role,
)
from .chat.models import BaseMLXModel, load_model


class ChatService:
    model: BaseMLXModel = None

    def __init__(self, model: BaseMLXModel):
        self.model = model

    async def generate_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletion:
        try:
            completion = ""
            prompt = ""
            self.model = load_model(request.model)

            async for text, _ in self.model.generate(request):
                completion += text
                if not prompt:  # Get prompt token count on first iteration
                    prompt = text

            prompt_tokens = await self.model.token_count(prompt)
            completion_tokens = await self.model.token_count(completion)

            return ChatCompletion(
                id=f"chatcmpl-{uuid.uuid4().hex[:10]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatCompletionMessage(
                            role=Role.ASSISTANT,
                            content=completion,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate completion: {str(e)}")

    async def generate_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        try:
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"
            created = int(time.time())

            async for response, finished in self.model.generate(request):
                yield ChatCompletionChunk(
                    id=chat_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta={"role": "assistant", "content": response},
                            finish_reason="stop" if finished else None,
                        )
                    ],
                )
        except Exception as e:
            raise RuntimeError(f"Failed to generate stream: {str(e)}")
