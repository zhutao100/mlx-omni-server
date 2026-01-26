import logging
import time
import uuid
from typing import Any, Callable, Dict, Generator

import mlx.core as mx
from mlx_lm.generate import GenerationCancelled, GenerationResponse, stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from rich.markup import escape

from ...utils.logger import logger
from ..models.models_service import MlxModelCache
from ..schema import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    PromptTokensDetails,
    Role,
)
from ..text_models import BaseTextModel, GenerateResult, GenerationParams
from ..tool_loop_reasoning_cache import tool_loop_reasoning_cache
from ..tools.chat_tokenizer import ChatTokenizer
from ..tools.tokens_decoder import ReasoningDecoder
from ..utils import (
    normalize_to_list,
    normalize_token,
    safe_decode_token,
    safe_encode_prompt,
)
from .json_logits_processor import JsonLogitsProcessor
from .prompt_cache import PromptCache, PromptCacheManager


class MlxLmModel(BaseTextModel):
    """MLX Chat Model wrapper with internal parameter management"""

    def __init__(
        self,
        model_cache: MlxModelCache,
    ):
        """Initialize MlxLmModel with model cache object.

        Args:
            model_cache: MlxModelCache object containing models and tokenizers
        """
        self._model_cache = model_cache
        self._default_max_tokens = 1048576

        # Import here to avoid circular imports
        from .model_types import load_tools_handler

        # Initialize chat_tokenizer here instead of using from model_cache
        self._chat_tokenizer: ChatTokenizer = load_tools_handler(
            model_cache.model_type, model_cache.tokenizer
        )

        if model_cache.tokenizer is None:
            raise ValueError("model_cache.tokenizer cannot be None")
        self._reasoning_decoder = ReasoningDecoder(thinking_tag=self._chat_tokenizer.thinking_tag)
        self._model_config = self._model_cache.model_config or {}
        if "max_position_embeddings" in self._model_config and isinstance(
            self._model_config["max_position_embeddings"], int
        ):
            max_context_length = self._model_config["max_position_embeddings"]
        else:
            max_context_length = 131072
            logger.warning(
                f"Invalid or missing max_position_embeddings in model config: {self._model_config}\n"
                f"Use default max_position_embeddings: {max_context_length}"
            )
        self._prompt_cache_manager = PromptCacheManager(
            max_position_embeddings=max_context_length, max_caches=2
        )
        self._prompt_cache_tokens_count = 0

    def _get_generation_params(
        self, request: ChatCompletionRequest
    ) -> GenerationParams:
        params = request.get_extra_params()

        # All params declare in `make_sampler`
        sampler_params = {
            "top_k",
            "min_tokens_to_keep",
            "min_p",
            "xtc_probability",
            "xtc_threshold",
            "xtc_special_tokens",
        }
        # Knowned params using in model config
        model_params = {
            "adapter_path",
            "draft_model",
            # Additional config for `apply_chat_template`
            "chat_template_config",
        }
        # Quick template params, same param will be overrided by `chat_template_config`
        template_params = {
            # Qwen3
            "enable_thinking",
            "thinking_budget",
            # Claude
            "thinking",
            # Gemini
            "thinkingConfig",
            # Grok
            "reasoning_effort",
            # Others
            "reasoning",
        }
        incompatible_params = {
            "include",
        }

        sampler_kwargs = {}
        model_kwargs = {}
        generate_kwargs = {}
        template_kwargs = {}

        for key, value in params.items():
            if key in sampler_params:
                sampler_kwargs[key] = value
            elif key in model_params:
                model_kwargs[key] = value
            elif key in template_params:
                template_kwargs[key] = value
            elif key in incompatible_params:
                logging.warning(f"Generation parameter '{key} : {value}' is not supported, dropping.")
            else:
                generate_kwargs[key] = value

        return {
            "sampler_kwargs": sampler_kwargs,
            "model_kwargs": model_kwargs,
            "generate_kwargs": generate_kwargs,
            "template_kwargs": template_kwargs,
        }

    def _process_logprobs(
        self,
        tokenizer: TokenizerWrapper,
        response: GenerationResponse,
        top_k: int | None,
    ) -> Dict[str, Any] | None:
        """Process logprobs information from generation response to match OpenAI format."""
        current_token = response.token
        current_logprobs = response.logprobs

        # Decode current token safely
        token_str = normalize_token(safe_decode_token(tokenizer, current_token))
        token_logprob = mx.clip(current_logprobs[current_token], a_min=-100, a_max=None).item()
        token_bytes = token_str.encode("utf-8")

        token_info = {
            "token": token_str,
            "logprob": token_logprob,
            "bytes": list(token_bytes),
        }

        top_logprobs: list[Dict[str, Any]] = []
        if top_k is not None:
            top_indices = mx.argpartition(-current_logprobs, kth=top_k - 1)[:top_k]
            top_probs = mx.clip(current_logprobs[top_indices], a_min=-100, a_max=None)

            top_indices_list = normalize_to_list(top_indices, int)
            top_probs_list = normalize_to_list(top_probs, float)

            for idx, logprob in zip(top_indices_list, top_probs_list):
                token = normalize_token(safe_decode_token(tokenizer, idx))
                token_bytes = token.encode("utf-8")
                top_logprobs.append(
                    {"token": token, "logprob": logprob, "bytes": list(token_bytes)}
                )

        return {**token_info, "top_logprobs": top_logprobs}

    def _prepare_generation(
        self,
        request: ChatCompletionRequest,
    ) -> tuple[Any, dict[str, Any]]:
        """Prepare all necessary components for generation.

        This function handles parameter processing, tokenizer setup, prompt encoding,
        sampler creation, and other preparation work for text generation.

        Args:
            request: The chat completion request containing generation parameters

        Returns:
            A tuple containing tokenizer, processed prompt, and generation kwargs
        """
        # Process parameters from request
        params = self._get_generation_params(request)

        model_kwargs = params.get("model_kwargs", {})
        logger.debug(f"Model kwargs: {model_kwargs}")

        template_kwargs = params.get("template_kwargs") | model_kwargs.get(
            "chat_template_config", {}
        )
        logger.debug(f"Chat Template kwargs: {template_kwargs}")

        # Prepare generation kwargs
        generate_kwargs = params.get("generate_kwargs", {})

        # Prepare sampler parameters
        sampler_kwargs = {
            "temp": (0.6 if request.temperature is None else request.temperature),
            "top_p": (1.0 if request.top_p is None else request.top_p),
            "min_p": 0.0,
            "min_tokens_to_keep": 1,
            "top_k": -1,
        } | params.get("sampler_kwargs", {})

        logger.debug(f"Sampler kwargs: {sampler_kwargs}")

        # Create sampler and add to generate_kwargs
        generate_kwargs["sampler"] = make_sampler(**sampler_kwargs)

        # Encode prompt with chat template
        prompt = self._chat_tokenizer.encode(
            messages=request.messages,
            tools=request.tools,
            **template_kwargs,
        )
        logger.debug(f"Encoded prompt:\n{escape(prompt)}")

        enable_thinking = template_kwargs.get("enable_thinking", True)
        self._reasoning_decoder.enable_thinking = enable_thinking
        if enable_thinking and prompt.endswith(f"{self._reasoning_decoder.thinking_start_tag}"):
            self._reasoning_decoder.set_thinking_prefix(True)
        else:
            self._reasoning_decoder.set_thinking_prefix(False)

        tokenizer: TokenizerWrapper = self._chat_tokenizer.tokenizer

        # Process prompt cache using the safe caller
        tokenized_prompt: list[int] = safe_encode_prompt(tokenizer, prompt)
        active_cache, processed_prompt, cached_count = (
            self._prompt_cache_manager.get_or_create_cache(
                self._model_cache,
                tokenized_prompt,
                session_key=request.prompt_cache_key,
            )
        )
        generate_kwargs["prompt_cache"] = active_cache.cache
        # keep a reference to extend later
        self._active_cache = active_cache
        self._prompt_cache_tokens_count = cached_count
        logger.debug(
            f"Using {self._prompt_cache_tokens_count} cached tokens out of {len(tokenized_prompt)} total tokens"
        )

        # Setup logits processors
        if request.response_format and request.response_format.json_schema:
            generate_kwargs["logits_processors"] = [
                JsonLogitsProcessor(
                    tokenizer, request.response_format
                )
            ]
        elif request.presence_penalty:
            generate_kwargs["logits_processors"] = make_logits_processors(
                repetition_penalty=request.presence_penalty
            )

        # Calculate max tokens for completion
        generate_kwargs["max_tokens"] = (
            request.max_completion_tokens
            or request.max_tokens
            or self._default_max_tokens
        )

        return processed_prompt, generate_kwargs

    def _stream_generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Generator[GenerateResult, None, None]:
        assert self._model_cache.model is not None
        try:
            # Get tokenizer
            tokenizer = self._chat_tokenizer.tokenizer

            # Prepare all generation components
            processed_prompt, generate_kwargs = self._prepare_generation(request)

            active_cache: PromptCache | None = getattr(self, "_active_cache", None)
            suffix_committed = 0
            if active_cache is not None:
                prompt_suffix = processed_prompt

                def prompt_progress_callback(processed: int, _total: int) -> None:
                    nonlocal suffix_committed
                    if processed <= suffix_committed:
                        return
                    active_cache.tokens.extend(prompt_suffix[suffix_committed:processed])
                    suffix_committed = processed

                generate_kwargs["prompt_progress_callback"] = prompt_progress_callback
            if should_cancel is not None:
                generate_kwargs["should_cancel"] = should_cancel

            response: GenerationResponse | None = None
            for response in stream_generate(
                model=self._model_cache.model,
                tokenizer=tokenizer,
                prompt=processed_prompt,
                draft_model=self._model_cache.draft_model,
                **generate_kwargs,
            ):
                if active_cache is not None:
                    active_cache.tokens.append(response.token)
                if response.finish_reason is not None:
                    break

                logprobs = None
                if request.logprobs:
                    logprobs = self._process_logprobs(
                        tokenizer, response, request.top_logprobs
                    )

                yield GenerateResult(
                    text=response.text,
                    token=response.token,
                    finish_reason=response.finish_reason,
                    prompt_tokens=response.prompt_tokens,
                    generation_tokens=response.generation_tokens,
                    logprobs=logprobs,
                )

            if response is not None:
                logger.debug(
                    f"    prompt tokens: {response.prompt_tokens}, tps: {response.prompt_tps}"
                )
                logger.debug(
                    f"generation tokens: {response.generation_tokens}, tps: {response.generation_tps}"
                )
                logger.debug(f"    finish reason: {response.finish_reason}")

        except GenerationCancelled:
            raise
        except Exception as e:
            logger.error(f"Error during stream generation: {escape(str(e))}", exc_info=True)
            raise

    def generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ChatCompletionResponse:
        try:
            completion = ""
            logprobs_result_list = []
            generated_tokens = []
            finish_reason = "stop"
            result = None

            for result in self._stream_generate(request=request, should_cancel=should_cancel):
                generated_tokens.append(result.token)
                completion += result.text

                if request.logprobs:
                    logprobs_result_list.append(result.logprobs)

                if result.finish_reason:
                    finish_reason = result.finish_reason

            if result is None:
                raise RuntimeError("No tokens generated")

            logger.debug(f"Model Response:\n{escape(completion)}")
            reasoning: str | None = None  # avoid UnboundLocalError
            enable_thinking = self._reasoning_decoder.enable_thinking
            reasoning_result = self._reasoning_decoder.decode(completion)
            if reasoning_result:
                logger.debug(f"Reasoning result:\n{escape(str(reasoning_result))}")
                completion = reasoning_result.get("content") or ""
                if enable_thinking:
                    reasoning = reasoning_result.get("reasoning")

            if request.tools:
                message = self._chat_tokenizer.decode(completion, request.tools)
                message.reasoning = reasoning
            else:
                message = ChatMessage(
                    role=Role.ASSISTANT,
                    content=completion,
                    reasoning=reasoning,
                )

            cached_tokens = self._prompt_cache_tokens_count
            logger.debug(f"Generate response with {cached_tokens} cached tokens")

            prompt_tokens_details = None
            if cached_tokens > 0:
                from ..schema import PromptTokensDetails

                prompt_tokens_details = PromptTokensDetails(cached_tokens=cached_tokens)

            assert message is not None
            if message.tool_calls and message.reasoning:
                for tool_call in message.tool_calls:
                    tool_loop_reasoning_cache.set(tool_call.id, message.reasoning)

            chat_completion_response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:10]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=message,
                        finish_reason=(
                            "tool_calls" if message.tool_calls else finish_reason
                        ),
                        logprobs=(
                            {"content": logprobs_result_list}
                            if logprobs_result_list
                            else None
                        ),
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=result.prompt_tokens + cached_tokens,
                    completion_tokens=result.generation_tokens,
                    total_tokens=result.prompt_tokens
                    + result.generation_tokens
                    + cached_tokens,
                    prompt_tokens_details=prompt_tokens_details,
                ),
            )
            logger.debug(f"ChatCompletionResponse: [{chat_completion_response}]")
            return chat_completion_response
        except GenerationCancelled:
            raise
        except Exception as e:
            logger.error(f"Failed to generate completion: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate completion: {str(e)}")

    def stream_generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Generator[ChatCompletionChunk, None, None]:
        try:
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"
            tool_call_index_by_id: dict[str, int] = {}
            next_tool_call_index = 0

            def ensure_tool_call_indexes(message: ChatMessage | None) -> None:
                nonlocal next_tool_call_index
                if message is None or not message.tool_calls:
                    return

                for tool_call in message.tool_calls:
                    if tool_call.index is not None:
                        continue
                    if tool_call.id in tool_call_index_by_id:
                        tool_call.index = tool_call_index_by_id[tool_call.id]
                        continue
                    tool_call_index_by_id[tool_call.id] = next_tool_call_index
                    tool_call.index = next_tool_call_index
                    next_tool_call_index += 1

            result: GenerateResult | None = None
            raw_completion = ""
            for result in self._stream_generate(request=request, should_cancel=should_cancel):
                if not result.text:
                    logger.warning(f"Generated result [{escape(str(result))}] with empty text")
                    continue
                raw_completion += result.text

                created = int(time.time())
                enable_thinking = self._reasoning_decoder.enable_thinking

                reasoning_result = self._reasoning_decoder.stream_decode(result.text)
                if not reasoning_result:
                    logger.warning(
                        f"Failed to decode reasoning from stream text: {escape(result.text)}"
                    )
                    continue
                logger.debug(f"Stream reasoning result:\n{escape(str(reasoning_result))}")
                delta_content: str | None = reasoning_result.get("delta_content")
                delta_reasoning: str | None = (
                    reasoning_result.get("delta_reasoning") if enable_thinking else None
                )

                reasoning_message: ChatMessage | None = None
                if delta_reasoning is not None:
                    reasoning_message = ChatMessage(
                        role=Role.ASSISTANT,
                        content=None,
                        reasoning=delta_reasoning,
                    )

                content_message: ChatMessage | None = None
                if delta_content is not None:
                    content_message = self._chat_tokenizer.decode_stream(
                        delta_content, request.tools
                    )

                logprobs = result.logprobs
                for message in (reasoning_message, content_message):
                    if message is None:
                        continue
                    ensure_tool_call_indexes(message)
                    choices = [
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=message,
                            finish_reason=result.finish_reason,
                            logprobs=logprobs,
                        )
                    ]
                    # Only yield if we have a message to send (avoid sending empty chunks when filtering XML)
                    yield ChatCompletionChunk(
                        id=chat_id,
                        created=created,
                        model=request.model,
                        choices=choices,
                    )
                    logprobs = None

            final_message = self._chat_tokenizer.parse_buffer(request.tools) or ChatMessage(
                role=Role.ASSISTANT, content=""
            )
            if final_message.tool_calls and self._reasoning_decoder.enable_thinking:
                reasoning_result = self._reasoning_decoder.decode(raw_completion)
                final_reasoning = reasoning_result.get("reasoning") if reasoning_result else None
                if final_reasoning:
                    for tool_call in final_message.tool_calls:
                        tool_loop_reasoning_cache.set(tool_call.id, final_reasoning)
            ensure_tool_call_indexes(final_message)
            finish_reason = "tool_calls" if final_message.tool_calls else "stop"
            # Send final chunk with finish reason
            choices = [
                ChatCompletionChunkChoice(
                    index=0,
                    delta=final_message,
                    finish_reason=finish_reason,
                    logprobs=None,
                )
            ]
            final_chat_completion_chunk = ChatCompletionChunk(
                id=chat_id,
                created=int(time.time()),
                model=request.model,
                choices=choices,
            )
            logger.debug(f"Final ChatCompletionChunk: [{final_chat_completion_chunk}]")
            yield final_chat_completion_chunk

            if result and request.stream_options and request.stream_options.include_usage:
                cached_tokens = self._prompt_cache_tokens_count
                logger.debug(f"Stream response with {cached_tokens} cached tokens")
                prompt_tokens_details = None
                if cached_tokens > 0:
                    prompt_tokens_details = PromptTokensDetails(cached_tokens=cached_tokens)

                usage_chat_completion_chunk = ChatCompletionChunk(
                    id=chat_id,
                    created=int(time.time()),
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatMessage(role=Role.ASSISTANT),
                            finish_reason=None,
                            logprobs=None,
                        )
                    ],
                    usage=ChatCompletionUsage(
                        prompt_tokens=result.prompt_tokens + cached_tokens,
                        completion_tokens=result.generation_tokens,
                        total_tokens=result.prompt_tokens
                        + result.generation_tokens
                        + cached_tokens,
                        prompt_tokens_details=prompt_tokens_details,
                    ),
                )
                logger.debug(f"Usage ChatCompletionChunk: [{usage_chat_completion_chunk}]")
                yield usage_chat_completion_chunk

        except GenerationCancelled:
            raise
        except Exception as e:
            logger.error(f"Error during stream generation: {escape(str(e))}", exc_info=True)
            raise
