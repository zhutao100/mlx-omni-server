import asyncio
import time
import uuid
from typing import AsyncGenerator, Dict, List, Tuple

import mlx.core as mx
from mlx_lm.utils import load, stream_generate

from ..schemas.chat_schema import (
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionUsage,
    Message,
    Role,
)


class MLXChatModel:
    def __init__(self):
        self._models = {}
        self._tokenizers = {}
        self._load_locks = {}

    async def _ensure_model_loaded(self, model_name: str) -> Tuple[any, any]:
        """Ensures model is loaded, with locking to prevent concurrent loads"""
        if model_name not in self._load_locks:
            self._load_locks[model_name] = asyncio.Lock()

        async with self._load_locks[model_name]:
            if model_name not in self._models:
                try:
                    model, tokenizer = load(
                        model_name,
                        tokenizer_config={"trust_remote_code": True},
                    )
                    self._models[model_name] = model
                    self._tokenizers[model_name] = tokenizer
                except Exception as e:
                    raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

        return self._models[model_name], self._tokenizers[model_name]

    def format_prompt(self, messages: List[Dict[str, str]], tokenizer) -> str:
        """Format messages into model prompt"""
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to format prompt: {str(e)}")

    async def count_tokens(self, text: str, tokenizer) -> int:
        """Count tokens in text"""
        try:
            tokens = tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            raise RuntimeError(f"Failed to count tokens: {str(e)}")

    async def generate(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> AsyncGenerator[Tuple[str, bool], None]:
        """Generate completion text"""
        try:
            model, tokenizer = await self._ensure_model_loaded(model_name)
            prompt = self.format_prompt(messages, tokenizer)

            # Set random seed for generation
            mx.random.seed(int(time.time()))

            async for response in self._stream_generate(
                model, tokenizer, prompt, max_tokens, temperature, top_p
            ):
                yield response

        except Exception as e:
            raise RuntimeError(f"Generation failed: {str(e)}")

    async def _stream_generate(
        self, model, tokenizer, prompt, max_tokens, temperature, top_p
    ):
        """Wrapper around mlx_lm stream_generate to make it async-friendly"""
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
        ):
            # Ensure we always have a tuple with two elements
            if isinstance(response, tuple) and len(response) == 2:
                text, finished = response
            else:
                # If response is not a tuple or doesn't have 2 elements,
                # assume it's just the text and not finished
                text = str(response)
                finished = False

            yield text, finished

    async def get_token_count(self, text: str, model_name: str) -> int:
        """Get token count for text"""
        _, tokenizer = await self._ensure_model_loaded(model_name)
        return await self.count_tokens(text, tokenizer)


class ChatService:
    def __init__(self):
        self.model = MLXChatModel()

    def _format_messages(self, messages: List[Message]) -> List[dict]:
        return [{"role": msg.role.value, "content": msg.content} for msg in messages]

    async def generate_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletion:
        try:
            messages = self._format_messages(request.messages)
            prompt = ""
            completion = ""

            async for text, _ in self.model.generate(
                request.model,
                messages,
                request.max_tokens or 256,
                request.temperature,
                request.top_p,
            ):
                completion += text
                if not prompt:  # Get prompt token count on first iteration
                    prompt = text

            prompt_tokens = await self.model.get_token_count(prompt, request.model)
            completion_tokens = await self.model.get_token_count(
                completion, request.model
            )

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
            messages = self._format_messages(request.messages)
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"
            created = int(time.time())

            async for response, finished in self.model.generate(
                request.model,
                messages,
                request.max_tokens or 256,
                request.temperature,
                request.top_p,
            ):
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
