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


class TestChatCompletions:

    def test_chat_completions_normal(self, openai_client):
        try:
            model = "mlx-community/Llama-3.2-1B-Instruct-4bit"
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hello"}],
            )
            logger.info(f"Chat Completion Response:\n{response}\n")

            # Validate response
            assert response.model == model, "Model name is not correct"
            assert response.usage is not None, "No usage in response"
            assert response.object == "chat.completion", "No usage in response"
            choices = response.choices[0]
            assert choices.logprobs is None, "logprobs is not None"
            assert choices.message is not None, "No message in response"
        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_chat_completions_stream(self, openai_client):
        """Test basic streaming chat completion functionality"""
        try:
            model = "mlx-community/Llama-3.2-1B-Instruct-4bit"
            stream = openai_client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "hi"}], stream=True
            )

            # Validate streaming response
            chunk_count = 0
            content = ""
            for chunk in stream:
                logger.info(f"Received stream chunk: {chunk}")
                chunk_count += 1

                # Validate basic structure of each chunk
                assert chunk.model == model, "Incorrect model name"
                assert (
                    chunk.object == "chat.completion.chunk"
                ), "Incorrect response object type"
                assert len(chunk.choices) == 1, "Incorrect number of choices"

                # Collect content
                delta = chunk.choices[0].delta
                if delta.content is not None:
                    content += delta.content

            # Validate overall response
            assert chunk_count > 0, "No chunks received"
            assert content.strip(), "Generated content is empty"
            logger.info(f"Complete generated content: {content}")

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_chat_completions_stream_options(self, openai_client):
        """Test streaming chat completion with additional options"""
        try:
            model = "mlx-community/Llama-3.2-1B-Instruct-4bit"
            stream = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful writing assistant.",
                    },
                    {"role": "user", "content": "Write a short greeting."},
                ],
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=50,
            )

            # Validate streaming response
            chunk_count = 0
            content = ""
            has_usage_info = False

            for chunk in stream:
                logger.info(f"Received stream chunk: {chunk}")
                chunk_count += 1

                # Validate basic structure of each chunk
                assert chunk.model == model, "Incorrect model name"
                assert (
                    chunk.object == "chat.completion.chunk"
                ), "Incorrect response object type"

                choice = chunk.choices[0]
                # Collect content
                if choice.delta.content is not None:
                    content += choice.delta.content

                # Check for usage information
                if chunk.usage is not None:
                    has_usage_info = True
                    logger.info(f"Usage info: {chunk.usage}")

            # Validate overall response
            assert chunk_count > 0, "No chunks received"
            assert content.strip(), "Generated content is empty"
            assert has_usage_info, "No usage information received"
            logger.info(f"Complete generated content: {content}")

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise
