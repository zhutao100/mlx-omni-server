import time
from typing import Any, AsyncGenerator, Dict, Tuple

import mlx.core as mx
from mlx_lm.utils import GenerationResponse, stream_generate

from ...schemas.chat_schema import ChatCompletionRequest
from .base_models import BaseMLXModel


class MLXModel(BaseMLXModel):
    """MLX Chat Model wrapper with internal parameter management"""

    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer
        self._load_locks = {}
        self._default_max_tokens = 256
        self._default_temperature = 1.0
        self._default_top_p = 1.0

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
            model = self._model
            tokenizer = self._tokenizer

            prompt = tokenizer.apply_chat_template(
                request.messages, tokenize=False, add_generation_prompt=True
            )

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
            if isinstance(response, GenerationResponse):
                # Extract text from GenerationResponse and determine if generation is finished
                text = response.text
                # 如果token是结束标记或者已达到最大生成长度，则认为生成结束
                finished = (
                    response.token == tokenizer.eos_token_id
                    or response.generation_tokens
                    >= params.get("max_tokens", self._default_max_tokens)
                )
            else:
                # 兼容其他可能的返回格式
                text = str(response)
                finished = False

            yield text, finished

    async def token_count(self, prompt: str) -> int:
        tokens = self._tokenizer.encode(prompt)
        return len(tokens)
