import asyncio
import json
import logging

import pytest
from httpx import AsyncClient

MODEL = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestChatCompletions:

    def test_chat_completions_normal(self, openai_client):
        try:
            model = MODEL
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

    def test_chat_completions_extra_body(self, openai_client):
        try:
            model = MODEL
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hello"}],
                max_completion_tokens=50,
                extra_body={
                    "top_k": 50,
                    "min_p": 0.0,
                    "min_tokens_to_keep": 1,
                    # "adapter_path": "../../adapters/",
                },
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

    def test_chat_completions_draft_model(self, openai_client):
        try:
            model = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr5e-8"
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hello"}],
                max_completion_tokens=50,
                extra_body={
                    "draft-model": MODEL,
                },
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
            model = MODEL
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
            model = MODEL
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
                max_tokens=200,
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


@pytest.mark.asyncio
async def test_retry_canceled_stream_chat_completion(async_client: AsyncClient):
    """
    Tests that retrying a canceled streaming request starts a new generation.
    """
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Write a detailed essay about the history of artificial intelligence and machine learning."}
        ],
        "stream": True,
        "max_tokens": 500,  # Make it longer so we have time to cancel
    }

    # --- First request, which we will cancel ---
    lines_received = []
    is_cancelled = False
    logger.info("\n--- Starting first (canceled) request ---")
    try:
        async with async_client.stream("POST", "/v1/chat/completions", json=payload, timeout=5) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line and line.startswith("data:"):
                    lines_received.append(line)
                    if len(lines_received) >= 5:
                        logger.info("--- Canceling first request by breaking early ---")
                        is_cancelled = True
                        break
    except Exception as e:
        logger.info(f"--- First request terminated as expected: {e} ---")
        pass

    assert is_cancelled, "Test failed to cancel the first stream mid-generation."

    full_first_response = "\n".join(lines_received)
    assert "data: [DONE]" not in full_first_response, "Canceled stream should not be complete"

    # Give the server a moment to process the disconnection
    await asyncio.sleep(1)

    # --- Second request, which should succeed ---
    all_lines = []
    logger.info("\n--- Starting second (retry) request ---")
    async with async_client.stream("POST", "/v1/chat/completions", json=payload, timeout=15) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line:
                all_lines.append(line)

    full_response = "\n".join(all_lines)
    logger.info(f"--- Full response from second request: ---\n{full_response}")

    # The second request should complete successfully
    assert "data: [DONE]" in full_response, "Full stream should end with [DONE]"

    # Verify the content of the full stream
    assert len(all_lines) > len(lines_received), "Second request should return more lines than the canceled one"

    # Check that the content is what we expect from a fresh generation
    content = ""
    for line in all_lines:
        if line.startswith("data:"):
            data_part = line[len("data: "):].strip()
            if data_part and data_part != "[DONE]":
                try:
                    chunk_json = json.loads(data_part)
                    if "choices" in chunk_json and chunk_json["choices"][0].get("delta", {}).get("content"):
                        content += chunk_json["choices"][0]["delta"]["content"]
                except json.JSONDecodeError:
                    pytest.fail(f"Failed to decode JSON chunk: {data_part}")

    logger.info(f"--- Reconstructed content from second request: ---\n{content}")
    # Check for expected content in the response
    assert "artificial" in content.lower() or "machine" in content.lower()
    assert len(content) > 50, "Should have generated a reasonable amount of content"
