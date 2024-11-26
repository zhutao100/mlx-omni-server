import time
import uuid
from typing import AsyncGenerator

from ..schemas.chat_schema import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    ChoiceLogprobs,
    Role,
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
        try:
            completion = ""
            prompt = ""
            logprobs_result_list = []

            async for result in self.model.generate(request):
                completion += result.text

                if request.logprobs:
                    logprobs_result_list.append(result.logprobs)

                if not prompt:  # Get prompt token count on first iteration
                    prompt = result.text

            prompt_tokens = await self.model.token_count(prompt)
            completion_tokens = await self.model.token_count(completion)

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:10]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(
                            role=Role.ASSISTANT,
                            content=completion,
                        ),
                        finish_reason="stop",
                        logprobs=(
                            ChoiceLogprobs(content=logprobs_result_list)
                            if logprobs_result_list
                            else None
                        ),
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
    ) -> AsyncGenerator[ChatCompletionResponse, None]:
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
                            delta=ChatMessage(role=Role.ASSISTANT, content=response),
                            finish_reason="stop" if finished else None,
                        )
                    ],
                )
        except Exception as e:
            raise RuntimeError(f"Failed to generate stream: {str(e)}")
