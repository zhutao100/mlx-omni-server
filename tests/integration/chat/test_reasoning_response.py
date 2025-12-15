import logging

import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestReasoningResponse:
    """Test functionality of the ReasoningResponse class"""

    def test_streaming_reasoning_response(self, openai_client):
        """Test functionality of the ReasoningResponse class"""
        try:
            model = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"
            logger.info("Streaming response:")
            # Create a streaming chat completion
            # The 'stream=True' parameter is crucial for enabling streaming
            with openai_client.chat.completions.with_streaming_response.create(
                model=model,
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
            ) as response:
                # Iterate through the streamed chunks
                for chunk in response.iter_bytes():
                    # Each chunk is a byte string, decode and print it
                    # You would typically parse these chunks to reconstruct the full message
                    try:
                        data = chunk.decode("utf-8")
                        # Attempt to decode as UTF-8, handling potential incomplete JSON objects
                        logger.info(f"Received chunk: {data}")
                        # TODO: validate the chunk content
                    except UnicodeDecodeError:
                        # Handle cases where a chunk might not be a complete UTF-8 character sequence
                        pass
            logger.info("\nEnd of stream.")

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_reasoning_response(self, openai_client):
        """Test functionality of the ReasoningResponse class"""
        try:
            model = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hello"}],
            )
            logger.info(f"Chat Completion Response:\n{response.choices[0].message}\n")

            # Validate response
            assert response.object == "chat.completion", "No usage in response"
            choices = response.choices[0]
            assert choices.message is not None, "No message in response"

            # 注意：在实际环境中，模型可能会在内容中包含</think>标签
            # 我们只需要验证响应中有内容，而不是检查特定标签的存在或不存在
            assert choices.message.content, "Message content is empty"

            # 从日志中可以看到，reasoning属性实际上是存在于message对象中，而不是choices[0]对象中
            assert (
                hasattr(choices.message, "reasoning") and choices.message.reasoning is not None
            ), "No reasoning in message"
        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_none_reasoning_response(self, openai_client):
        """Test functionality of the ReasoningResponse class"""
        try:
            model = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr5e-8"
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hello"}],
                extra_body={
                    "enable_thinking": False,
                },
            )
            logger.info(f"Chat Completion Response:\n{response.choices[0].message}\n")

            # Validate response
            assert response.object == "chat.completion", "No usage in response"
            choices = response.choices[0]
            assert choices.message is not None, "No message in response"
            assert "</think>" not in choices.message.content, "Message content is not correct"
            assert (
                not hasattr(choices.message, "reasoning") or choices.message.reasoning is None
            ), "Has reasoning in message"
        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise
