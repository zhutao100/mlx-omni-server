from typing import Any, AsyncGenerator, Dict, List, Optional

from mlx_lm.utils import GenerationResponse, stream_generate
from transformers import AutoTokenizer

from ...schemas.chat_schema import ChatCompletionRequest, ChatMessage
from ...schemas.tools_schema import Tool
from .base_models import BaseMLXModel, GenerateResult
from .tools_handler import load_tools_handler


class MLXModel(BaseMLXModel):
    """MLX Chat Model wrapper with internal parameter management"""

    def __init__(self, model_id: str, model, tokenizer: AutoTokenizer):
        self._model_id = model_id
        self._model = model
        self._tokenizer = tokenizer
        self._load_locks = {}
        self._default_max_tokens = 2048
        self._default_temperature = 1.0
        self._default_top_p = 1.0
        self._tools_handler = load_tools_handler(model_id, tokenizer)

    def _get_generation_params(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Extract and validate generation parameters from request"""
        params = {
            "max_tokens": request.max_tokens or self._default_max_tokens,
            "temp": request.temperature or self._default_temperature,
            "top_p": request.top_p or self._default_top_p,
        }
        return params

    async def _stream_generate(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Tool]] = None,
        **kwargs,
    ) -> AsyncGenerator[GenerateResult, None]:
        """Stream generate text from the model."""
        # Format prompt with tools if provided
        prompt = self._tools_handler.encode_tools(
            conversation=messages,
            tools=tools,
            **kwargs,
        )

        accumulated_text = ""
        for response in stream_generate(self._model, self._tokenizer, prompt, **kwargs):
            if isinstance(response, GenerationResponse):
                text = response.text
                accumulated_text += text

                # Try to detect tool calls
                tool_calls = self._tools_handler.decode_tool_calls(accumulated_text)

                yield GenerateResult(
                    text=accumulated_text,
                    token=response.token,
                    finished=False,
                    tool_calls=tool_calls,
                )

        # Final check for tool calls
        tool_calls = self._tools_handler.decode_tool_calls(accumulated_text)

        yield GenerateResult(
            text=accumulated_text,
            token=-1,  # End token
            finished=True,
            tool_calls=tool_calls,
        )

    async def generate(
        self,
        request: ChatCompletionRequest,
        **kwargs,
    ) -> AsyncGenerator[GenerateResult, None]:
        """Generate text from the model."""
        try:
            # Get generation parameters
            params = self._get_generation_params(request)
            params.update(kwargs)

            async for result in self._stream_generate(
                messages=request.messages,
                tools=request.tools,
                **params,
            ):
                yield result
        except Exception as e:
            raise RuntimeError(f"Generation failed: {str(e)}")

    async def token_count(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._tokenizer.encode(text))
