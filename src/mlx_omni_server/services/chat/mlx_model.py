import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import GenerationResponse, stream_generate

from ...schemas.chat_schema import (
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
from ...schemas.tools_schema import Tool
from ...utils.logger import logger
from .base_models import BaseMLXModel, GenerateResult
from .tools_handler import load_tools_handler


class MLXModel(BaseMLXModel):
    """MLX Chat Model wrapper with internal parameter management"""

    def __init__(self, model_id: str, model, tokenizer: TokenizerWrapper):
        self._model_id = model_id
        self._model = model
        self._tokenizer = tokenizer
        self._default_max_tokens = 2048
        self._default_temperature = 1.0
        self._default_top_p = 1.0
        self._tools_handler = load_tools_handler(model_id, tokenizer)
        logger.info(f"Initialized MLXModel with model_id: {model_id}")

    def _get_generation_params(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Extract and validate generation parameters from request"""
        logger.debug(f"Generation parameters: {request.model_dump()}")
        return {}

    async def _stream_generate(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Tool]] = None,
        **kwargs,
    ) -> AsyncGenerator[GenerateResult, None]:
        """Stream generate text from the model."""
        logger.info(f"Starting stream generation for {len(messages)} messages")
        if tools:
            logger.debug(f"Using {len(tools)} tools for generation")

        # Format prompt with tools if provided
        prompt = self._tools_handler.encode_tools(
            conversation=messages,
            tools=tools,
            **kwargs,
        )
        logger.debug(f"Encoded prompt:\n{prompt}")

        accumulated_text = ""
        try:
            for response in stream_generate(
                model=self._model,
                tokenizer=self._tokenizer,
                prompt=prompt,
                max_tokens=self._default_max_tokens,
                sampler=make_sampler(self._default_temperature, self._default_top_p),
                **kwargs,
            ):
                if isinstance(response, GenerationResponse):
                    text = response.text
                    accumulated_text += text
                    logger.debug(
                        f"Current response token: {response.token}, text: {response.text}"
                    )
                    logger.debug(f"Current accumulated text: {accumulated_text}")

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

            # 生成最终结果
            final_result = GenerateResult(
                text=accumulated_text,
                token=None,
                finished=True,
                tool_calls=tool_calls,
            )
            logger.debug(
                f"Yielding final result with text length: {len(final_result.text)}"
            )
            yield final_result
            logger.debug("Final result yielded successfully")

        except Exception as e:
            logger.error(f"Error during stream generation: {str(e)}", exc_info=True)
            raise

    def _update_generation_params(self, request: ChatCompletionRequest):
        """Update generation parameters"""
        self._default_max_tokens = request.max_tokens or self._default_max_tokens
        self._default_temperature = request.temperature or self._default_temperature
        self._default_top_p = request.top_p or self._default_top_p

    def _build_chat_response(
        self,
        model: str,
        message: ChatMessage,
        usage: ChatCompletionUsage,
        logprobs: Optional[ChoiceLogprobs] = None,
    ) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:10]}",
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=message,
                    finish_reason="tool_calls" if message.tool_calls else "stop",
                    logprobs=logprobs,
                )
            ],
            usage=usage,
        )

    async def generate(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Generate a chat completion response."""
        try:
            completion = ""
            prompt = ""
            logprobs_result_list = []
            tool_calls = None

            self._update_generation_params(request)
            params = self._get_generation_params(request)

            async for result in self._stream_generate(
                messages=request.messages,
                tools=request.tools,
                **params,
            ):
                if result.tool_calls:
                    tool_calls = result.tool_calls
                    break

                completion += result.text

                if request.logprobs:
                    logprobs_result_list.append(result.logprobs)

                if not prompt:  # Get prompt token count on first iteration
                    prompt = result.text

            prompt_tokens = await self.token_count(prompt)
            completion_tokens = await self.token_count(completion)

            return self._build_chat_response(
                model=request.model,
                message=ChatMessage(
                    role=Role.ASSISTANT,
                    content=None if tool_calls else completion,
                    tool_calls=tool_calls,
                ),
                usage=ChatCompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
                logprobs=(
                    ChoiceLogprobs(content=logprobs_result_list)
                    if logprobs_result_list
                    else None
                ),
            )
        except Exception as e:
            logger.error(f"Failed to generate completion: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate completion: {str(e)}")

    async def stream_generate(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Stream generate chat completion chunks."""
        try:
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"
            created = int(time.time())

            self._update_generation_params(request)
            params = self._get_generation_params(request)

            async for result in self._stream_generate(
                messages=request.messages,
                tools=request.tools,
                **params,
            ):
                yield ChatCompletionChunk(
                    id=chat_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatMessage(role=Role.ASSISTANT, content=result.text),
                            finish_reason="stop" if result.finished else None,
                            logprobs=result.logprobs,
                        )
                    ],
                )
        except Exception as e:
            logger.error(f"Failed to stream generate: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to stream generate: {str(e)}")

    async def token_count(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._tokenizer.encode(text))
