"""
Integration tests for streaming finish_reason functionality

This test file verifies that streaming responses correctly include finish_reason
in the final chunk when using the actual server.
"""

import json
import logging

import pytest
from httpx import AsyncClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use a small, fast model for testing
MODEL = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"


class TestStreamFinishReason:
    """Integration tests for streaming finish_reason functionality"""

    def test_stream_finish_reason_stop(self, openai_client):
        """Test that streaming correctly sets finish_reason to 'stop'"""
        try:
            response = openai_client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "Say hello"}],
                stream=True,
                max_tokens=10
            )

            # Collect all chunks
            chunks = []
            for chunk in response:
                chunks.append(chunk)
                logger.info(f"Chunk: {chunk}")

            # Verify we have chunks
            assert len(chunks) > 0, "Should have received chunks"

            # Find the final chunk with finish_reason
            final_chunk = None
            for chunk in chunks:
                if chunk.choices and chunk.choices[0].finish_reason is not None:
                    final_chunk = chunk
                    break

            # Verify final chunk exists and has correct finish_reason
            assert final_chunk is not None, "Should have a final chunk with finish_reason"
            assert final_chunk.choices[0].finish_reason == "stop", f"Expected finish_reason 'stop', got '{final_chunk.choices[0].finish_reason}'"
            logger.info(f"Final chunk with finish_reason: {final_chunk}")

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_stream_finish_reason_with_usage(self, openai_client):
        """Test that streaming with usage includes finish_reason correctly"""
        try:
            response = openai_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2+2?"}
                ],
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=20
            )

            # Collect all chunks
            chunks = []
            has_finish_reason = False
            has_usage = False

            for chunk in response:
                chunks.append(chunk)
                logger.info(f"Chunk: {chunk}")
                
                # Check for finish_reason
                if chunk.choices and chunk.choices[0].finish_reason is not None:
                    has_finish_reason = True
                    
                # Check for usage
                if chunk.usage is not None:
                    has_usage = True

            # Verify we have chunks
            assert len(chunks) > 0, "Should have received chunks"
            assert has_finish_reason, "Should have a chunk with finish_reason"
            assert has_usage, "Should have a chunk with usage information"

            # Find the final chunk with finish_reason
            final_chunk = None
            for chunk in chunks:
                if chunk.choices and chunk.choices[0].finish_reason is not None:
                    final_chunk = chunk
                    break

            # Verify final chunk exists and has correct finish_reason
            assert final_chunk is not None, "Should have a final chunk with finish_reason"
            assert final_chunk.choices[0].finish_reason in ["stop", "length"], f"Expected finish_reason 'stop' or 'length', got '{final_chunk.choices[0].finish_reason}'"
            logger.info(f"Final chunk with finish_reason: {final_chunk}")

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    @pytest.mark.asyncio
    async def test_async_stream_finish_reason(self, async_client: AsyncClient):
        """Test async streaming correctly sets finish_reason to 'stop'"""
        try:
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Count from 1 to 5"}],
                "stream": True,
                "max_tokens": 30
            }

            async with async_client.stream("POST", "/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                
                chunks = []
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        if data.strip() != "[DONE]":
                            try:
                                chunk_data = json.loads(data)
                                chunks.append(chunk_data)
                                logger.info(f"Chunk: {chunk_data}")
                            except json.JSONDecodeError:
                                pass  # Ignore non-JSON lines

            # Verify we have chunks
            assert len(chunks) > 0, "Should have received chunks"

            # Find the final chunk with finish_reason
            final_chunk = None
            for chunk in chunks:
                if "choices" in chunk and chunk["choices"][0].get("finish_reason") is not None:
                    final_chunk = chunk
                    break

            # Verify final chunk exists and has correct finish_reason
            assert final_chunk is not None, "Should have a final chunk with finish_reason"
            finish_reason = final_chunk["choices"][0]["finish_reason"]
            assert finish_reason in ["stop", "length"], f"Expected finish_reason 'stop' or 'length', got '{finish_reason}'"
            logger.info(f"Final chunk with finish_reason: {final_chunk}")

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise