import asyncio
import gc
import time
import uuid
from typing import Any, Dict, Generator, List, Tuple, Union

import mlx.core as mx
from mlx_vlm import GenerationResult, generate, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from rich.markup import escape

from ...utils.logger import logger
from ..models.models_service import MlxModelCache
from ..schema import (ChatCompletionChoice, ChatCompletionChunk,
                      ChatCompletionChunkChoice, ChatCompletionRequest,
                      ChatCompletionResponse, ChatCompletionUsage, ChatMessage,
                      MultimodalContentItem, PromptTokensDetails, Role)
from ..text_models import BaseTextModel, GenerateResult
from ..tools.chat_tokenizer import ChatTokenizer
from ..tools.tokens_decoder import ReasoningDecoder
from ..utils import convert_prompt_to_str, safe_encode_prompt
from .media_processor import MediaProcessor
from .prompt_cache import PromptCache, PromptCacheManager, get_vlm_cache_config


class MlxVlmModel(BaseTextModel):
    """Handler for Vision-Language Models that can process both text and multimodal inputs"""

    def __init__(self, model_cache: MlxModelCache, **kwargs):
        self._model_cache = model_cache
        self.media_processor = MediaProcessor()
        self.disable_auto_resize = kwargs.get("disable_auto_resize", False)
        self.context_length = kwargs.get("context_length", 65536)

        # Import here to avoid circular imports
        from .model_types import load_tools_handler

        # Initialize chat_tokenizer here instead of using from model_cache
        self._chat_tokenizer: ChatTokenizer = load_tools_handler(
            model_cache.model_type, model_cache.tokenizer
        )

        if model_cache.tokenizer is None:
            raise ValueError("model_cache.tokenizer cannot be None")
        self._reasoning_decoder = ReasoningDecoder(thinking_tag=self._chat_tokenizer.thinking_tag)
        self.model_created = int(time.time())

        # Initialize prompt cache manager
        cache_config = get_vlm_cache_config(model_cache.model_id.name)
        self._prompt_cache_manager = PromptCacheManager(
            max_position_embeddings=self.context_length,
            max_caches=cache_config.get("max_caches", 5)
        )
        self._prompt_cache_tokens_count = 0
        self._default_max_tokens = 1048576

    def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate a complete response for multimodal requests"""
        try:
            logger.debug(f"Received generate request: {request}")

            # Prepare all generation components
            model, _, generate_kwargs, formatted_prompt = self._prepare_generation(request)
            tokenizer = self._model_cache.tokenizer

            # Call the VLM model
            result = generate(
                model,
                tokenizer, # type: ignore
                formatted_prompt,
                **generate_kwargs
            )

            # Force garbage collection
            gc.collect()

            # Convert to ChatCompletionResponse format
            return self._format_response(result, request.model, request)

        except ValueError as e:
            logger.error(f"Validation error in VLM generation: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"Runtime error in VLM generation: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in VLM generation: {e}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")

    def stream_generate(self, request: ChatCompletionRequest) -> Generator[ChatCompletionChunk, None, None]:
        """Generate a streaming response for multimodal requests following the mlx_lm_model pattern"""
        try:
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"
            result: GenerateResult | None = None

            for result in self._stream_generate(request=request):
                if not result.text:
                    logger.warning(f"Generated result [{escape(str(result))}] with empty text")
                    continue

                created = int(time.time())
                message = None
                enable_thinking = self._reasoning_decoder.enable_thinking
                delta_content: str | None = result.text
                delta_reasoning: str | None = None

                if enable_thinking:
                    reasoning_result = self._reasoning_decoder.stream_decode(
                        result.text
                    )
                    if not reasoning_result:
                        logger.warning(f"Failed to decode reasoning from stream text: {escape(result.text)}")
                        continue
                    logger.debug(f"Stream reasoning result:\n{escape(str(reasoning_result))}")
                    delta_content = reasoning_result.get("delta_content")
                    delta_reasoning = reasoning_result.get("delta_reasoning")

                if delta_reasoning is not None:
                    # If we have a delta reasoning, we need to send it as a message
                    message = ChatMessage(
                        role=Role.ASSISTANT,
                        content=delta_content,
                        reasoning=delta_reasoning,
                    )
                elif delta_content is not None:
                    message = self._chat_tokenizer.decode_stream(delta_content, request.tools)

                if message:
                    choices = [
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=message,
                            finish_reason=result.finish_reason,
                            logprobs=result.logprobs,
                        )
                    ]
                    # Only yield if we have a message to send (avoid sending empty chunks when filtering XML)
                    yield ChatCompletionChunk(
                        id=chat_id,
                        created=created,
                        model=request.model,
                        choices=choices,
                    )

            final_message = self._chat_tokenizer.parse_buffer(
                request.tools) or ChatMessage(role=Role.ASSISTANT, content="")
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
            yield ChatCompletionChunk(
                id=chat_id,
                created=int(time.time()),
                model=request.model,
                choices=choices,
            )

            if result and request.stream_options and request.stream_options.include_usage:
                cached_tokens = self._prompt_cache_tokens_count
                logger.debug(f"Stream response with {cached_tokens} cached tokens")
                prompt_tokens_details = None
                if cached_tokens > 0:
                    prompt_tokens_details = PromptTokensDetails(cached_tokens=cached_tokens)

                yield ChatCompletionChunk(
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
        except Exception as e:
            logger.error(f"Error during stream generation: {escape(str(e))}", exc_info=True)
            raise

    def _prepare_multimodal_request(self, request: ChatCompletionRequest) -> Tuple[List[Dict[str, Any]], List[str], List[str], Dict[str, Any]]:
        """Prepare multimodal request by processing messages with text, images, and audio"""
        chat_messages = []
        image_urls = []
        audio_urls = []

        try:
            # Process each message in the request
            for message in request.messages:
                if message.role in ["system", "assistant"]:
                    # Handle simple string content for system and assistant messages
                    if isinstance(message.content, str):
                        chat_messages.append({"role": message.role, "content": message.content})
                    # Handle list of content items (though this is unusual for system/assistant)
                    elif isinstance(message.content, list):
                        texts = []
                        for item in message.content:
                            if isinstance(item, MultimodalContentItem) and item.type == "text":
                                text = getattr(item, "text", "").strip()
                                if text:
                                    texts.append(text)
                            elif isinstance(item, dict) and item.get("type") == "text":
                                text = item.get("text", "").strip()
                                if text:
                                    texts.append(text)
                        if texts:
                            chat_messages.append({"role": message.role, "content": " ".join(texts)})
                    continue

                if message.role == "user":
                    # Case 1: Simple string content
                    if isinstance(message.content, str):
                        chat_messages.append({"role": "user", "content": message.content})
                        continue

                    # Case 2: Content is a list of dictionaries or objects
                    if isinstance(message.content, list):
                        # Initialize containers for this message
                        texts = []
                        images = []
                        audios = []

                        # Process each content item in the list
                        for item in message.content:
                            # Handle MultimodalContentItem objects
                            if isinstance(item, MultimodalContentItem):
                                if item.type == "text":
                                    text = getattr(item, "text", "").strip()
                                    if text:
                                        texts.append(text)
                                elif item.type == "image_url":
                                    url = getattr(item, "image_url", None)
                                    if url:
                                        # Handle ImageUrl objects
                                        if hasattr(url, "url"):
                                            url = url.url
                                        images.append(url)
                                elif item.type == "input_audio":
                                    audio_input = getattr(item, "input_audio", None)
                                    if audio_input and hasattr(audio_input, "data"):
                                        audio_data = audio_input.data
                                        audio_format = getattr(audio_input, "format", "mp3")
                                        # Create data URL from audio data
                                        audio_url = f"data:audio/{audio_format};base64,{audio_data}"
                                        audios.append(audio_url)
                            # Handle dictionary objects
                            elif isinstance(item, dict):
                                if item.get("type") == "text":
                                    text = item.get("text", "").strip()
                                    if text:
                                        texts.append(text)
                                elif item.get("type") == "image_url":
                                    url = item.get("image_url")
                                    if url:
                                        # Handle ImageUrl objects or URLs
                                        if isinstance(url, dict) and "url" in url:
                                            url = url["url"]
                                        images.append(url)
                                elif item.get("type") == "input_audio":
                                    audio_input = item.get("input_audio")
                                    if audio_input and "data" in audio_input:
                                        audio_data = audio_input["data"]
                                        audio_format = audio_input.get("format", "mp3")
                                        # Create data URL from audio data
                                        audio_url = f"data:audio/{audio_format};base64,{audio_data}"
                                        audios.append(audio_url)

                        # Add collected media to global lists
                        if images:
                            image_urls.extend(images)
                            # Validate constraints
                            if len(images) > 4:
                                raise ValueError("Too many images in a single message (max: 4)")

                        if audios:
                            audio_urls.extend(audios)
                            # Validate constraints
                            if len(audios) > 2:
                                raise ValueError("Too many audio files in a single message (max: 2)")

                        # Add text content if available, otherwise use empty string
                        if texts:
                            chat_messages.append({"role": "user", "content": " ".join(texts)})
                        else:
                            chat_messages.append({"role": "user", "content": ""})
                    else:
                        raise ValueError("Invalid message content format")

            logger.debug(f"Extracted {len(image_urls)} image URLs and {len(audio_urls)} audio URLs from request")
            for image_url in image_urls:
                if image_url.startswith("data:"):
                    logger.debug(f"Image data URL of length {len(image_url)}")
                elif image_url.startswith("http"):
                    logger.debug(f"Image URL: {image_url}")
                else:
                    logger.debug(f"Image file path: {image_url}")
            for audio_url in audio_urls:
                if audio_url.startswith("data:"):
                    logger.debug(f"Audio data URL of length {len(audio_url)}")
                elif audio_url.startswith("http"):
                    logger.debug(f"Audio URL: {audio_url}")
                else:
                    logger.debug(f"Audio file path: {audio_url}")
            # Process images and audio files
            try:
                image_paths, _ = asyncio.run(self.media_processor.process_image_urls(
                    image_urls,
                    resize=not self.disable_auto_resize
                ))
            except Exception as e:
                logger.error(f"Failed to process images: {e}")
                raise ValueError(f"Failed to process images: {str(e)}")

            try:
                audio_paths, _ = asyncio.run(self.media_processor.process_audio_urls(audio_urls))
            except Exception as e:
                logger.error(f"Failed to process audio files: {e}")
                raise ValueError(f"Failed to process audio files: {str(e)}")

            # Filter out None values
            image_paths = [path for path in image_paths if path is not None]
            audio_paths = [path for path in audio_paths if path is not None]

            # Extract model parameters
            model_params = {
                "temperature": request.temperature if request.temperature is not None else 0.7,
                "top_p": request.top_p if request.top_p is not None else 1.0,
                "frequency_penalty": request.frequency_penalty if request.frequency_penalty is not None else 0.0,
                "presence_penalty": request.presence_penalty if request.presence_penalty is not None else 0.0,
                "max_tokens": request.max_tokens if request.max_tokens is not None else 1024,
            }

            return chat_messages, image_paths, audio_paths, model_params

        except ValueError as e:
            logger.error(f"Validation error in preparing multimodal request: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to prepare multimodal request: {e}")
            raise RuntimeError(f"Failed to prepare multimodal request: {str(e)}")

    def _format_response(self, result: Any, model: str, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Format VLM response to match mlx-omni-server response format"""
        # Extract text from result
        response_text = result.text if hasattr(result, 'text') else str(result)

        # Extract usage statistics if available
        prompt_tokens = getattr(result, 'prompt_tokens', 0)
        completion_tokens = getattr(result, 'generation_tokens', 0)
        total_tokens = getattr(result, 'total_tokens', prompt_tokens + completion_tokens)

        # Handle reasoning/thinking
        reasoning: str | None = None
        enable_thinking = self._reasoning_decoder.enable_thinking
        if enable_thinking:
            reasoning_result = self._reasoning_decoder.decode(response_text)
            if reasoning_result:
                logger.debug(f"Reasoning result:\n{escape(str(reasoning_result))}")
                response_text = reasoning_result.get("content") or ""
                reasoning = reasoning_result.get("reasoning")

        # Handle tools (similar to LM model)
        if request.tools:
            message = self._chat_tokenizer.decode(response_text, request.tools)
        else:
            message = ChatMessage(
                role=Role.ASSISTANT,
                content=response_text,
                reasoning=reasoning,
            )

        # Handle cached tokens
        cached_tokens = self._prompt_cache_tokens_count
        logger.debug(f"Generate response with {cached_tokens} cached tokens")

        prompt_tokens_details = None
        if cached_tokens > 0:
            from ..schema import PromptTokensDetails
            prompt_tokens_details = PromptTokensDetails(cached_tokens=cached_tokens)

        # Create response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:10]}",
            created=int(time.time()),
            model=model,
            choices=[ChatCompletionChoice(
                index=0,
                message=message,
                finish_reason="tool_calls" if hasattr(message, 'tool_calls') and message.tool_calls else "stop"
            )],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens + cached_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens + cached_tokens,
                prompt_tokens_details=prompt_tokens_details,
            )
        )

        return response

    async def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'media_processor'):
                await self.media_processor.cleanup()
            gc.collect()
            logger.info("MlxVlmModel cleanup completed")
        except Exception as e:
            logger.error(f"Error during MlxVlmModel cleanup: {e}")
            # Don't raise the exception as cleanup should not fail the overall process
            pass

    def _stream_generate(
        self,
        request: ChatCompletionRequest,
    ) -> Generator[GenerateResult, None, None]:
        """Internal stream generation method that yields GenerateResult objects."""
        try:
            # Prepare all generation components
            model, prompt_tokens, generate_kwargs, formatted_prompt = self._prepare_generation(request)

            # Initialize variables to track tokens
            token_counter = 0
            # Use safe_encode_prompt to get prompt tokens count
            prompt_tokens_len = len(prompt_tokens)
            tokenizer = self._model_cache.tokenizer

            # Call the VLM model with streaming
            for response in stream_generate(
                model,
                tokenizer, # type: ignore
                formatted_prompt,
                **generate_kwargs
            ):
                token_counter += 1
                # For VLM, the response is a GenerationResult object or a plain string;
                # safely extract text if available, otherwise use the string representation.
                text = getattr(response, "text", str(response))

                # Determine if this is the last token (we can't know for sure in streaming)
                # So we'll set finish_reason to None for all tokens
                finish_reason = None

                # For VLM, we don't have individual token information in streaming, so we approximate
                yield GenerateResult(
                    text=text,
                    token=getattr(response, 'token', 0) or 0,  # token ID if available
                    finish_reason=finish_reason,
                    prompt_tokens=prompt_tokens_len,
                    generation_tokens=token_counter,
                    logprobs=None,  # TODO: Implement logprobs processing if needed
                )

                # Force garbage collection periodically
                if token_counter % 10 == 0:
                    gc.collect()

            # Send final result with stop finish reason
            if token_counter > 0:
                yield GenerateResult(
                    text="",  # Empty text for final result
                    token=0,
                    finish_reason="stop",
                    prompt_tokens=prompt_tokens_len,
                    generation_tokens=token_counter,
                    logprobs=None,
                )

            logger.debug(
                f"    prompt tokens: {prompt_tokens_len}"
            )
            logger.debug(
                f"generation tokens: {token_counter}"
            )

        except Exception as e:
            logger.error(f"Error during stream generation: {escape(str(e))}", exc_info=True)
            raise

    def _prepare_generation(
        self,
        request: ChatCompletionRequest,
    ) -> tuple[Any, Any, dict[str, Any], str]:
        """Prepare all necessary components for generation.

        This function handles parameter processing, tokenizer setup, prompt encoding,
        and other preparation work for multimodal generation.

        Args:
            request: The chat completion request containing generation parameters

        Returns:
            A tuple containing model, prompt_tokens, generation kwargs, and formatted_prompt
        """
        # Get model components
        model = self._model_cache.model
        tokenizer = self._model_cache.tokenizer
        assert model is not None, "Model is not loaded"
        model_path = self._model_cache.model_id.name

        # Process multimodal request
        chat_messages, image_paths, audio_paths, model_params = self._prepare_multimodal_request(request)

        # Prepare the prompt using the chat template
        formatted_prompt = convert_prompt_to_str(apply_chat_template(
            tokenizer,
            model.config,
            chat_messages,
            add_generation_prompt=True,
            num_images=len(image_paths) if image_paths else 0,
            num_audios=len(audio_paths) if audio_paths else 0
        ))

        prompt_tokens = safe_encode_prompt(tokenizer, formatted_prompt, add_special_tokens=True)

        # Generate media hashes for cache key
        media_hashes = []
        if image_paths:
            for path in image_paths:
                media_hashes.append(self.media_processor.generate_media_hash(path))
        if audio_paths:
            for path in audio_paths:
                media_hashes.append(self.media_processor.generate_media_hash(path))

        # TODO: re-enable prompt cache for VLM models
        # # Get or create cache using our cache manager
        # model_key = f"{model_path}_{getattr(model.config, 'model_type', 'unknown')}"
        # prompt_cache, _, cached_count = self._prompt_cache_manager.get_or_create_cache(
        #     model, model_key, input_ids, media_hashes
        # )
        # self._prompt_cache_tokens_count = cached_count
        prompt_cache = PromptCache(max_position_embeddings=self.context_length)
        prompt_cache.reset_prompt_cache(
            model, model_key=f"{model_path}_{getattr(model.config, 'model_type', 'unknown')}",
            prompt_tokens=prompt_tokens, media_hashes=media_hashes)
        self._prompt_cache_tokens_count = 0

        # Prepare generation kwargs
        generate_kwargs = {
            "image": image_paths if image_paths else None,
            "audio": audio_paths if audio_paths else None,
            "prompt_cache": prompt_cache.cache if hasattr(prompt_cache, 'cache') else prompt_cache,
            "max_tokens": request.max_completion_tokens or request.max_tokens or self._default_max_tokens,
            "temperature": request.temperature if request.temperature is not None else 0.7,
            "top_p": request.top_p if request.top_p is not None else 1.0,
            "frequency_penalty": request.frequency_penalty if request.frequency_penalty is not None else 0.0,
            "presence_penalty": request.presence_penalty if request.presence_penalty is not None else 0.0,
        }

        enable_thinking = getattr(request, "enable_thinking", True)
        self._reasoning_decoder.enable_thinking = enable_thinking
        if enable_thinking:
            if formatted_prompt.endswith(f"{self._reasoning_decoder.thinking_start_tag}"):
                self._reasoning_decoder.set_thinking_prefix(True)
            else:
                self._reasoning_decoder.set_thinking_prefix(False)

        logger.debug(f"Formatted prompt: {escape(formatted_prompt)}")
        logger.debug(f"Using {self._prompt_cache_tokens_count} cached tokens out of {len(prompt_tokens)} total tokens")
        logger.debug(f"Image paths: {image_paths}")
        logger.debug(f"Audio paths: {audio_paths}")

        return model, prompt_tokens, generate_kwargs, formatted_prompt
