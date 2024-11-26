import time
from typing import Any, AsyncGenerator, Dict, Optional

import mlx.core as mx
from mlx_lm.utils import GenerationResponse, stream_generate

from ...schemas.chat_schema import ChatCompletionRequest
from .base_models import BaseMLXModel, GenerateResult


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

    def _process_logprobs(
        self, response: GenerationResponse, top_k: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        """Process logprobs information from generation response to match OpenAI format"""
        current_token = response.token
        current_logprobs = response.logprobs

        # Get current token info
        token_str = self._tokenizer.decode([current_token])
        token_logprob = current_logprobs[current_token].item()
        token_bytes = token_str.encode("utf-8")

        # Base token info
        token_info = {
            "token": token_str,
            "logprob": token_logprob,
            "bytes": list(token_bytes),
        }

        # Process top logprobs
        top_logprobs = []
        if top_k is not None:
            # Get indices of top_k tokens
            top_indices = mx.argpartition(-current_logprobs, kth=top_k - 1)[:top_k]
            top_probs = current_logprobs[top_indices]

            # Create detailed token information for each top token
            for idx, logprob in zip(top_indices.tolist(), top_probs.tolist()):
                token = self._tokenizer.decode([idx])
                token_bytes = token.encode("utf-8")
                top_logprobs.append(
                    {"token": token, "logprob": logprob, "bytes": list(token_bytes)}
                )

        return {**token_info, "top_logprobs": top_logprobs}

    async def _stream_generate(
        self, model, tokenizer, prompt: str, request: ChatCompletionRequest, **params
    ) -> AsyncGenerator[GenerateResult, None]:
        """Stream generate with logprobs support"""
        for response in stream_generate(model, tokenizer, prompt, **params):
            if isinstance(response, GenerationResponse):
                # Extract text and check if generation is finished
                text = response.text
                finished = (
                    response.token == tokenizer.eos_token_id
                    or response.generation_tokens
                    >= params.get("max_tokens", self._default_max_tokens)
                )

                # Process logprobs if available
                logprobs_info = None
                if request.logprobs:
                    logprobs_info = self._process_logprobs(
                        response, request.top_logprobs
                    )

                yield GenerateResult(
                    text=text,
                    token=response.token,
                    finished=finished,
                    logprobs=logprobs_info,
                )
            else:
                # Handle other response formats
                yield GenerateResult(
                    text=str(response),
                    token=-1,  # Invalid token ID to indicate non-token response
                    finished=False,
                    logprobs=None,
                )

    async def generate(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[GenerateResult, None]:
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
                model, tokenizer, prompt, request, **gen_params
            ):
                yield response

        except Exception as e:
            raise RuntimeError(f"Generation failed: {str(e)}")

    async def token_count(self, prompt: str) -> int:
        """Count the number of tokens in the text"""
        tokens = self._tokenizer.encode(prompt)
        return len(tokens)
