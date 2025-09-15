import asyncio
import json
import time
from unittest.mock import AsyncMock, Mock, patch, MagicMock

import pytest

from mlx_omni_server.chat.models.models import load_model
from mlx_omni_server.chat.models.models_service import ModelId
from mlx_omni_server.chat.router import (
    CACHE_TTL,
    NonStreamCacheEntry,
    StreamCacheEntry,
    _create_text_model,
    make_request_hash,
    response_cache,
)
from mlx_omni_server.chat.schema import (
    ChatCompletionRequest,
    ChatMessage,
    Role,
    MultimodalContentItem,
    ImageUrl,
    AudioInput
)
from mlx_omni_server.chat.text_models import (
    BaseTextModel,
    ChatCompletionChunk,
    ChatCompletionResponse,
)

# Constants
VLM_MODEL_ID = "llava-hf/llava-1.5-7b-hf"


# Mock Classes
class MockVlmModel(BaseTextModel):
    """Mock VLM model for testing"""

    def __init__(self):
        self.call_count = 0
        self.stream_call_count = 0

    def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Mock generate method"""
        self.call_count += 1
        content = "This is a test image description."
        if request.messages and len(request.messages) > 0:
            if isinstance(request.messages[0].content, str):
                if "describe" in request.messages[0].content.lower():
                    content = "This is a beautiful landscape with mountains and a lake."

        return ChatCompletionResponse(
            id="test-vlm-response-id",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": ChatMessage(
                        role=Role.ASSISTANT,
                        content=content,
                    ),
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        )

    def stream_generate(self, request: ChatCompletionRequest):
        """Mock stream generate method"""
        self.stream_call_count += 1
        content = "This is a test image description."
        if request.messages and len(request.messages) > 0:
            if isinstance(request.messages[0].content, str):
                if "describe" in request.messages[0].content.lower():
                    content = "This is a beautiful landscape with mountains and a lake."

        # Yield chunks
        for i, word in enumerate(content.split()):
            yield ChatCompletionChunk(
                id="test-vlm-chunk-id",
                created=int(time.time()),
                model=request.model,
                choices=[
                    {
                        "index": 0,
                        "delta": ChatMessage(
                            role=Role.ASSISTANT,
                            content=word + " ",
                        ),
                        "finish_reason": None,
                    }
                ],
            )
        # Final chunk with finish reason
        yield ChatCompletionChunk(
            id="test-vlm-chunk-id",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "delta": ChatMessage(
                        role=Role.ASSISTANT,
                        content="",
                    ),
                    "finish_reason": "stop",
                }
            ],
        )





class TestVlmChatCompletions:

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test"""
        # Clear cache before each test
        response_cache.clear()
        yield
        # Clear cache after each test
        response_cache.clear()

    def test_vlm_chat_completions_normal(self, openai_client):
        """Test normal VLM chat completions with image"""
        with patch("mlx_omni_server.chat.router._create_text_model") as mock_create_text_model:

            # Mock the VLM model
            mock_vlm_model = MockVlmModel()
            mock_create_text_model.return_value = mock_vlm_model

            try:
                response = openai_client.chat.completions.create(
                    model=VLM_MODEL_ID,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "https://example.com/test-image.jpg"
                                    }
                                }
                            ],
                        }
                    ],
                )

                # Validate response
                assert response.model == VLM_MODEL_ID, "Model name is not correct"
                assert response.usage is not None, "No usage in response"
                assert response.object == "chat.completion", "Object type is not correct"
                assert len(response.choices) == 1, "Should have one choice"
                assert response.choices[0].message.content is not None, "No content in response"
                assert "test image description" in response.choices[0].message.content.lower(), "Content is not correct"

                # Verify the mock was called
                mock_create_text_model.assert_called()

            except Exception as e:
                pytest.fail(f"Chat completion failed with error: {e}")

    def test_vlm_chat_completions_streaming(self, openai_client):
        """Test streaming VLM chat completions with image"""
        with patch("mlx_omni_server.chat.router._create_text_model") as mock_create_text_model:

            # Mock the VLM model
            mock_vlm_model = MockVlmModel()
            mock_create_text_model.return_value = mock_vlm_model

            try:
                response = openai_client.chat.completions.create(
                    model=VLM_MODEL_ID,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "https://example.com/test-image.jpg"
                                    }
                                }
                            ],
                        }
                    ],
                    stream=True,
                )

                # Collect all chunks
                chunks = []
                for chunk in response:
                    chunks.append(chunk)

                # Validate chunks
                assert len(chunks) > 0, "Should have received chunks"
                assert chunks[0].model == VLM_MODEL_ID, "Model name is not correct"
                assert chunks[0].object == "chat.completion.chunk", "Object type is not correct"

                # Verify the mock was called
                mock_create_text_model.assert_called()

            except Exception as e:
                pytest.fail(f"Streaming chat completion failed with error: {e}")

    def test_vlm_model_cache_manager(self):
        """Test VLM model cache manager"""
        # Create cache manager
        from mlx_omni_server.chat.models.models import model_cache_manager

        # Load model
        model_id = ModelId(name=VLM_MODEL_ID)
        model = model_cache_manager.load_model(model_id)

        # Verify model was created
        assert model is not None

        # Load same model again (should reuse)
        model2 = model_cache_manager.load_model(model_id)
        assert model2 is model  # Should be the same instance

    def test_vlm_request_multimodal_detection(self):
        """Test detection of multimodal requests"""
        # Create a multimodal request
        request = ChatCompletionRequest(
            model=VLM_MODEL_ID,
            messages=[
                ChatMessage(
                    role=Role.USER,
                    content=[
                        MultimodalContentItem(type="text", text="Describe this image"),
                        MultimodalContentItem(
                            type="image_url",
                            image_url=ImageUrl(url="https://example.com/test-image.jpg")
                        )
                    ]
                )
            ]
        )

        # Verify it's detected as multimodal
        assert request.is_multimodal_request() == True

        # Create a text-only request
        text_request = ChatCompletionRequest(
            model=VLM_MODEL_ID,
            messages=[
                ChatMessage(
                    role=Role.USER,
                    content="Hello, how are you?"
                )
            ]
        )

        # Verify it's not detected as multimodal
        assert text_request.is_multimodal_request() == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
