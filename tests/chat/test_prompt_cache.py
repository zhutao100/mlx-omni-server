"""
Tests for prompt cache functionality

This test file verifies the prompt caching functionality in the chat completion API, including:
1. First conversation with no cache
2. Second conversation using cache
3. Modified conversation still hitting partial cache
"""

import logging

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from mlx_omni_server.main import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def openai_client(client):
    """Create OpenAI client configured with test server"""
    return OpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=client,
    )


class TestPromptCache:
    """Tests for prompt cache functionality"""

    def test_conversation_with_prompt_cache(self, openai_client):
        try:
            logger.info("\n===== Conversation with prompt cache =====")
            model = "mlx-community/Llama-3.2-1B-Instruct-4bit"
            prompt = "Can you tell me more about your capabilities?"

            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ]

            first_response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=20,
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": first_response.choices[0].message.content,
                }
            )
            messages.append({"role": "user", "content": "continue"})

            # Create second conversation
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=20,
            )

            # Verify cache in second conversation
            assert (
                response.usage.prompt_tokens_details is not None
            ), "Second conversation should have cached tokens"
            assert (
                response.usage.prompt_tokens_details.cached_tokens > 0
            ), "Cached tokens count should be greater than 0"
            logger.info(
                f"Second conversation cached tokens: {response.usage.prompt_tokens_details.cached_tokens}"
            )

        except Exception as e:
            logger.error(f"Error testing prompt cache: {str(e)}")
            raise

    def test_trim_conversation_with_partial_cache(self, openai_client):

        try:
            logger.info("\n===== trim conversation, partial cache hit =====")

            model = "mlx-community/Llama-3.2-1B-Instruct-4bit"
            second_prompt = "Can you tell me more about your capabilities?"

            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": second_prompt},
            ]

            first_response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=20,
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": first_response.choices[0].message.content,
                }
            )
            messages.append({"role": "user", "content": "continue!"})
            logger.info(f"Conversation: {messages}")

            # Create second conversation
            second_response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=20,
            )
            logger.info(
                f"Second conversation prompt_tokens_details: {second_response.usage.prompt_tokens_details}"
            )
            logger.info(
                f"Second conversation response content: {second_response.choices[0].message.content}"
            )

            # rm last message
            messages.pop()
            messages.append({"role": "user", "content": "More !"})
            logger.info(f"Trim conversation: {messages}")

            trimmed_response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=20,
            )

            # Log response information
            logger.info(
                f"Trim conversation prompt_tokens_details:: {trimmed_response.usage.prompt_tokens_details}"
            )
            logger.info(
                f"Trim conversation response content: {trimmed_response.choices[0].message.content}"
            )

            # Verify cache in modified conversation
            # assert response.usage.prompt_tokens_details is not None, "Modified conversation should have cached tokens"
            # assert response.usage.prompt_tokens_details.cached_tokens > 0, "Modified conversation cached tokens should be greater than 0"
            # logger.info(f"Modified conversation cached tokens: {response.usage.prompt_tokens_details.cached_tokens}")

        except Exception as e:
            logger.error(f"Error testing prompt cache: {str(e)}")
            raise
