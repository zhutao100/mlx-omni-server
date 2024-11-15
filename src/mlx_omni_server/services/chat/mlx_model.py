import asyncio
import time
from typing import Any, AsyncGenerator, Dict, List, Tuple

import mlx.core as mx
from mlx_lm.utils import load, stream_generate

from ...schemas.chat_schema import ChatCompletionRequest
from .models import BaseModel


class MLXModel(BaseModel):
    """MLX Chat Model wrapper with internal parameter management"""

    def __init__(self):
        self._models = {}
        self._tokenizers = {}
        self._load_locks = {}
        self._default_max_tokens = 256
        self._default_temperature = 1.0
        self._default_top_p = 1.0

    async def _ensure_model_loaded(self, model_name: str) -> Tuple[Any, Any]:
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

    def _format_messages(self, messages: List[Dict[str, str]], tokenizer) -> str:
        """Format messages into model prompt"""
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to format prompt: {str(e)}")

    async def _count_tokens(self, text: str, tokenizer) -> int:
        """Count tokens in text"""
        try:
            tokens = tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            raise RuntimeError(f"Failed to count tokens: {str(e)}")

    def _get_generation_params(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Extract and validate generation parameters from request"""
        params = {
            "max_tokens": request.max_tokens or self._default_max_tokens,
            "temp": request.temperature or self._default_temperature,
            "top_p": request.top_p or self._default_top_p,
        }
        # Add any extra parameters from request
        params.update(request.get_extra_params())
        return params

    async def generate(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[Tuple[str, bool], None]:
        """Generate completion text with parameters from request"""
        try:
            model, tokenizer = await self._ensure_model_loaded(request.model)
            messages = [
                {"role": msg.role.value, "content": msg.content}
                for msg in request.messages
            ]
            prompt = self._format_messages(messages, tokenizer)

            # Set random seed for generation
            if request.seed is not None:
                mx.random.seed(request.seed)
            else:
                mx.random.seed(int(time.time()))

            # Get generation parameters including extra params
            gen_params = self._get_generation_params(request)

            async for response in self._stream_generate(
                model, tokenizer, prompt, **gen_params
            ):
                yield response

        except Exception as e:
            raise RuntimeError(f"Generation failed: {str(e)}")

    async def _stream_generate(self, model, tokenizer, prompt, **params):
        """Wrapper around mlx_lm stream_generate to make it async-friendly"""
        for response in stream_generate(model, tokenizer, prompt, **params):
            if isinstance(response, tuple) and len(response) == 2:
                text, finished = response
            else:
                text = str(response)
                finished = False

            yield text, finished

    async def get_token_count(self, text: str, model_name: str) -> int:
        """Get token count for text"""
        _, tokenizer = await self._ensure_model_loaded(model_name)
        return await self._count_tokens(text, tokenizer)
