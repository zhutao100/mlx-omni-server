import asyncio
import gc
import math
import threading
import time
import uuid
from typing import Any, Callable, Generator, Tuple

import mlx.core as mx
from mlx_vlm import stream_generate
from mlx_vlm.prompt_utils import apply_chat_template, get_chat_template
from PIL import Image
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
    MultimodalContentItem,
    PromptTokensDetails,
    Role,
)
from ..text_models import BaseTextModel, GenerateResult
from ..tool_loop_reasoning_cache import tool_loop_reasoning_cache
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
        self._generation_lock = threading.Lock()

    def _encode_prompt_tokens(
        self,
        processor: Any,
        model: Any,
        formatted_prompt: str,
        image_paths: list[str],
        audio_paths: list[str],
    ) -> list[int]:
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

        add_special_tokens = (
            not hasattr(processor, "chat_template")
            if getattr(model.config, "model_type", None) in {"gemma3", "gemma3n"}
            else True
        )
        tokens = safe_encode_prompt(
            tokenizer,
            formatted_prompt,
            add_special_tokens=add_special_tokens,
        )
        tokens = [int(token_id) for token_id in tokens]

        if not image_paths:
            return tokens

        model_type = str(getattr(model.config, "model_type", "") or "").lower()
        if not model_type.startswith("glm4v"):
            return tokens

        image_token_id = getattr(model.config, "image_token_id", None)
        if not isinstance(image_token_id, int):
            return tokens

        placeholder_count = tokens.count(image_token_id)
        if placeholder_count < len(image_paths):
            return tokens

        vision_config = getattr(model.config, "vision_config", None)
        if isinstance(vision_config, dict):
            patch_size = vision_config.get("patch_size")
            spatial_merge_size = vision_config.get("spatial_merge_size") or vision_config.get(
                "merge_size"
            )
        else:
            patch_size = getattr(vision_config, "patch_size", None)
            spatial_merge_size = getattr(vision_config, "spatial_merge_size", None) or getattr(
                vision_config, "merge_size", None
            )

        if not isinstance(patch_size, int) or patch_size <= 0:
            return tokens
        if not isinstance(spatial_merge_size, int) or spatial_merge_size <= 0:
            return tokens

        image_token_counts: list[int] = []
        for path in image_paths:
            try:
                with Image.open(path) as image:
                    width, height = image.size
            except Exception:
                logger.debug(
                    "Failed to read image size for token expansion: %s", path, exc_info=True
                )
                return tokens

            patches_w = math.ceil(width / patch_size)
            patches_h = math.ceil(height / patch_size)
            merged_w = max(1, patches_w // spatial_merge_size)
            merged_h = max(1, patches_h // spatial_merge_size)
            image_token_counts.append(merged_w * merged_h)

        expanded_tokens: list[int] = []
        image_index = 0
        for token_id in tokens:
            if token_id == image_token_id and image_index < len(image_token_counts):
                expanded_tokens.extend([image_token_id] * image_token_counts[image_index])
                image_index += 1
            else:
                expanded_tokens.append(token_id)

        if image_index != len(image_token_counts):
            logger.debug(
                "Image token expansion mismatch: expanded %d/%d placeholders",
                image_index,
                len(image_token_counts),
            )
            return tokens

        return expanded_tokens

    def generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ChatCompletionResponse:
        """Generate a complete response for multimodal requests"""
        with self._generation_lock:
            try:
                logger.debug(f"Received generate request: {request}")

                text = ""
                last_token = 0
                prompt_tokens_processed = 0
                generation_tokens = 0

                for chunk in self._stream_generate(request=request):
                    if chunk.text:
                        text += chunk.text
                    if chunk.finish_reason is None:
                        last_token = chunk.token
                    prompt_tokens_processed = chunk.prompt_tokens
                    generation_tokens = chunk.generation_tokens

                # Force garbage collection
                gc.collect()

                result = GenerateResult(
                    text=text,
                    token=last_token,
                    finish_reason="stop",
                    prompt_tokens=prompt_tokens_processed,
                    generation_tokens=generation_tokens,
                    logprobs=None,
                )

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

    def stream_generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Generate a streaming response for multimodal requests following the mlx_lm_model pattern"""
        with self._generation_lock:
            try:
                include_thinking_in_content = bool(
                    getattr(request, "include_thinking_in_content", False)
                )
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

                for result in self._stream_generate(request=request):
                    if not result.text:
                        continue
                    raw_completion += result.text

                    created = int(time.time())
                    message = None
                    if include_thinking_in_content:
                        message = self._chat_tokenizer.decode_stream(result.text, request.tools)
                    else:
                        enable_thinking = self._reasoning_decoder.enable_thinking
                        delta_content: str | None = result.text
                        delta_reasoning: str | None = None

                        reasoning_result = self._reasoning_decoder.stream_decode(result.text)
                        if not reasoning_result:
                            logger.warning(
                                f"Failed to decode reasoning from stream text: {escape(result.text)}"
                            )
                            continue
                        logger.debug(f"Stream reasoning result:\n{escape(str(reasoning_result))}")
                        delta_content = reasoning_result.get("delta_content")
                        if enable_thinking:
                            delta_reasoning = reasoning_result.get("delta_reasoning")

                        if delta_reasoning is not None:
                            # If we have a delta reasoning, we need to send it as a message
                            message = ChatMessage(
                                role=Role.ASSISTANT,
                                content=delta_content,
                                reasoning=delta_reasoning,
                            )
                        elif delta_content is not None:
                            message = self._chat_tokenizer.decode_stream(
                                delta_content, request.tools
                            )

                    if message:
                        ensure_tool_call_indexes(message)
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

                final_message = self._chat_tokenizer.parse_buffer(request.tools) or ChatMessage(
                    role=Role.ASSISTANT, content=""
                )
                if final_message.tool_calls and self._reasoning_decoder.enable_thinking:
                    reasoning_result = self._reasoning_decoder.decode(raw_completion)
                    final_reasoning = (
                        reasoning_result.get("reasoning") if reasoning_result else None
                    )
                    if final_reasoning:
                        final_message.reasoning = final_reasoning
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

    def _prepare_multimodal_request(self, request: ChatCompletionRequest) -> Tuple[list[dict[str, Any]], list[str], list[str]]:
        """Prepare multimodal request by processing messages with text, images, and audio
        Args:
            request: The chat completion request containing messages with multimodal content
        Returns:
            A tuple containing:
            - List of chat messages formatted for the model
            - List of local image file paths
            - List of local audio file paths
        Raises:
            ValueError: If there are too many images or audio files in a single message
        """
        chat_messages = []
        image_urls = []
        audio_urls = []

        try:
            # Process each message in the request
            for message in request.messages:
                if message.role in ["system", "assistant"]:
                    # Handle simple string content for system and assistant messages
                    if isinstance(message.content, str):
                        msg: dict[str, Any] = {"role": message.role, "content": message.content}
                        if message.role == Role.ASSISTANT and message.reasoning is not None:
                            msg["reasoning_content"] = message.reasoning
                        chat_messages.append(msg)
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
                            msg = {"role": message.role, "content": " ".join(texts)}
                            if message.role == Role.ASSISTANT and message.reasoning is not None:
                                msg["reasoning_content"] = message.reasoning
                            chat_messages.append(msg)
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

            return chat_messages, image_paths, audio_paths

        except ValueError as e:
            logger.error(f"Validation error in preparing multimodal request: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to prepare multimodal request: {e}")
            raise RuntimeError(f"Failed to prepare multimodal request: {str(e)}")

    def _format_response(self, result: Any, model: str, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Format VLM response to match mlx-omni-server response format"""
        # Extract text from result
        response_text = result.text if hasattr(result, "text") else str(result)
        include_thinking_in_content = bool(getattr(request, "include_thinking_in_content", False))

        # Extract usage statistics if available
        prompt_tokens = getattr(result, 'prompt_tokens', 0)
        completion_tokens = getattr(result, 'generation_tokens', 0)
        total_tokens = getattr(result, 'total_tokens', prompt_tokens + completion_tokens)

        # Handle reasoning/thinking
        reasoning: str | None = None
        enable_thinking = self._reasoning_decoder.enable_thinking
        if include_thinking_in_content:
            if enable_thinking:
                reasoning_result = self._reasoning_decoder.decode(response_text)
                if reasoning_result:
                    logger.debug(f"Reasoning result:\n{escape(str(reasoning_result))}")
                    reasoning = reasoning_result.get("reasoning")
        else:
            reasoning_result = self._reasoning_decoder.decode(response_text)
            if reasoning_result:
                logger.debug(f"Reasoning result:\n{escape(str(reasoning_result))}")
                response_text = reasoning_result.get("content") or ""
                if enable_thinking:
                    reasoning = reasoning_result.get("reasoning")

        # Handle tools (similar to LM model)
        if request.tools:
            message = self._chat_tokenizer.decode(response_text, request.tools)
            message.reasoning = reasoning
        else:
            message = ChatMessage(
                role=Role.ASSISTANT,
                content=response_text,
                reasoning=reasoning,
            )
        if message.tool_calls and message.reasoning:
            for tool_call in message.tool_calls:
                tool_loop_reasoning_cache.set(tool_call.id, message.reasoning)

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
            prompt_tokens_len = len(prompt_tokens)
            tokenizer = self._model_cache.tokenizer
            active_cache: PromptCache | None = getattr(self, "_active_cache", None)
            prompt_committed = False
            last_committed_generation_tokens = 0
            prompt_tokens_processed = prompt_tokens_len
            detokenizer = getattr(tokenizer, "detokenizer", None)
            last_detokenized_text = ""

            def detokenized_text() -> str:
                if detokenizer is None:
                    return ""
                base_text = getattr(detokenizer, "text", "")
                if not isinstance(base_text, str):
                    try:
                        base_text = str(base_text)
                    except Exception:
                        base_text = ""

                unflushed = getattr(detokenizer, "_unflushed", None)
                if not isinstance(unflushed, str) or not unflushed:
                    return base_text

                trim_space = bool(getattr(detokenizer, "trim_space", False))
                byte_decoder = getattr(detokenizer, "_byte_decoder", None)
                if isinstance(byte_decoder, dict):
                    try:
                        current_text = bytearray(byte_decoder[c] for c in unflushed).decode(
                            "utf-8", errors="ignore"
                        )
                    except Exception:
                        current_text = ""
                else:
                    current_text = unflushed.replace("\u2581", " ")

                if base_text or not trim_space:
                    return f"{base_text}{current_text}"
                if current_text.startswith(" "):
                    current_text = current_text[1:]
                return f"{base_text}{current_text}"

            # Call the VLM model with streaming
            for response in stream_generate(
                model,
                tokenizer,  # type: ignore
                formatted_prompt,
                **generate_kwargs
            ):
                token_id = getattr(response, "token", 0) or 0
                generation_tokens = int(getattr(response, "generation_tokens", 0) or 0)
                response_prompt_tokens = getattr(response, "prompt_tokens", None)
                if isinstance(response_prompt_tokens, int):
                    prompt_tokens_processed = response_prompt_tokens

                if active_cache is not None:
                    if not prompt_committed and generation_tokens > 0:
                        active_cache.tokens.extend(prompt_tokens)
                        bundle = getattr(active_cache, "bundle", None)
                        if bundle is not None:
                            bundle.tokens_processed = len(active_cache.tokens)
                        prompt_committed = True

                    if generation_tokens > last_committed_generation_tokens:
                        try:
                            active_cache.tokens.append(int(token_id))
                            bundle = getattr(active_cache, "bundle", None)
                            if bundle is not None:
                                bundle.tokens_processed = len(active_cache.tokens)
                        except (TypeError, ValueError):
                            pass
                        last_committed_generation_tokens = generation_tokens

                delta_text = ""
                full_text = detokenized_text()
                if full_text and full_text.startswith(last_detokenized_text):
                    delta_text = full_text[len(last_detokenized_text) :]
                    last_detokenized_text = full_text
                else:
                    fallback_text = getattr(response, "text", "")
                    if isinstance(fallback_text, str):
                        delta_text = fallback_text

                yield GenerateResult(
                    text=delta_text,
                    token=int(token_id) if isinstance(token_id, int) else 0,
                    finish_reason=None,
                    prompt_tokens=prompt_tokens_processed,
                    generation_tokens=max(last_committed_generation_tokens, generation_tokens),
                    logprobs=None,  # TODO: Implement logprobs processing if needed
                )

                # Force garbage collection periodically
                if generation_tokens > 0 and generation_tokens % 10 == 0:
                    gc.collect()

            # Send final result with stop finish reason
            if last_committed_generation_tokens > 0:
                yield GenerateResult(
                    text="",  # Empty text for final result
                    token=0,
                    finish_reason="stop",
                    prompt_tokens=prompt_tokens_processed,
                    generation_tokens=last_committed_generation_tokens,
                    logprobs=None,
                )

            logger.debug(f"    prompt tokens: {prompt_tokens_processed}")
            logger.debug(f"generation tokens: {last_committed_generation_tokens}")

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
        chat_messages, image_paths, audio_paths = self._prepare_multimodal_request(request)

        enable_thinking = getattr(request, "enable_thinking", True)
        self._reasoning_decoder.enable_thinking = enable_thinking

        # Prepare the prompt using the chat template
        template_messages = apply_chat_template(
            tokenizer,
            model.config,
            chat_messages,
            add_generation_prompt=True,
            return_messages=True,
            num_images=len(image_paths) if image_paths else 0,
            num_audios=len(audio_paths) if audio_paths else 0,
            enable_thinking=enable_thinking,
        )
        # Normalize return type: ensure template_messages is a list[dict]
        if isinstance(template_messages, str):
            template_messages = [{"role": Role.ASSISTANT, "content": template_messages}]
        elif template_messages is None:
            template_messages = []
        for src, dst in zip(chat_messages, template_messages):
            reasoning_content = src.get("reasoning_content")
            if reasoning_content is not None and dst.get("role") == Role.ASSISTANT:
                dst["reasoning_content"] = reasoning_content

        model_type = str(getattr(model.config, "model_type", "") or "").lower()
        if model_type in {"paligemma", "molmo", "florence2"}:
            formatted_prompt = convert_prompt_to_str(template_messages[-1])
        else:
            formatted_prompt = convert_prompt_to_str(
                get_chat_template(
                    tokenizer,
                    template_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=enable_thinking,
                )
            )

        full_prompt_tokens = self._encode_prompt_tokens(
            tokenizer,
            model,
            formatted_prompt,
            image_paths,
            audio_paths,
        )

        # Generate media hashes for cache key
        media_hashes = []
        if image_paths:
            for path in image_paths:
                media_hashes.append(self.media_processor.generate_media_hash(path))
        if audio_paths:
            for path in audio_paths:
                media_hashes.append(self.media_processor.generate_media_hash(path))

        model_key = f"{model_path}_{getattr(model.config, 'model_type', 'unknown')}"
        prompt_cache, prompt_tokens, cached_count = self._prompt_cache_manager.get_or_create_cache(
            model,
            model_key,
            full_prompt_tokens,
            media_hashes or None,
        )
        self._active_cache = prompt_cache
        self._prompt_cache_tokens_count = cached_count

        bundle = getattr(prompt_cache, "bundle", None)
        if bundle is None:
            prompt_cache.reset_prompt_cache(
                model,
                model_key=model_key,
                prompt_tokens=full_prompt_tokens,
                media_hashes=media_hashes or None,
            )
            bundle = prompt_cache.bundle

        # Prepare generation kwargs
        generate_kwargs = {
            "prompt_cache_bundle": bundle,
            "max_tokens": request.max_completion_tokens or request.max_tokens or self._default_max_tokens,
            "temperature": request.temperature if request.temperature is not None else 0.6,
            "top_p": request.top_p if request.top_p is not None else 1.0,
            "frequency_penalty": request.frequency_penalty if request.frequency_penalty is not None else 0.0,
            "presence_penalty": request.presence_penalty if request.presence_penalty is not None else 0.0,
        }
        if cached_count > 0:
            input_ids = mx.array([prompt_tokens], dtype=mx.int32)
            generate_kwargs.update(
                {
                    "image": None,
                    "audio": None,
                    "input_ids": input_ids,
                    "pixel_values": None,
                    "mask": mx.ones_like(input_ids),
                }
            )
        else:
            generate_kwargs.update(
                {
                    "image": image_paths or None,
                    "audio": audio_paths or None,
                }
            )

        if enable_thinking:
            if formatted_prompt.endswith(f"{self._reasoning_decoder.thinking_start_tag}"):
                self._reasoning_decoder.set_thinking_prefix(True)
            else:
                self._reasoning_decoder.set_thinking_prefix(False)

        logger.debug(f"Formatted prompt: {escape(formatted_prompt)}")
        logger.debug(
            f"Using {self._prompt_cache_tokens_count} cached tokens out of {len(full_prompt_tokens)} total tokens"
        )
        logger.debug(f"Generation kwargs: {generate_kwargs}")

        return model, prompt_tokens, generate_kwargs, formatted_prompt
