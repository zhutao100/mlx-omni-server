import asyncio
import json
import logging

import pytest
from httpx import AsyncClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"


class TestResponsesIntegration:

    def test_responses_normal(self, openai_client):
        """Test basic non-streaming responses functionality"""
        try:
            response = openai_client.responses.create(
                model=MODEL,
                input=[{"role": "user", "content": "hello"}],
            )
            logger.info(f"Responses Response:\n{response}\n")

            # Validate response
            assert response.model == MODEL, "Model name is not correct"
            assert response.usage is not None, "No usage in response"
            assert response.object == "response", "Incorrect response object type"
            assert response.status == "completed", "Response status is not completed"
            assert len(response.output) > 0, "No output in response"

            # Check that we have a message output
            message_output = next(
                (item for item in response.output if item.type == "message"), None
            )
            assert message_output is not None, "No message output found"
            assert len(message_output.content) > 0, "No content in message output"
            assert message_output.content[0].text.strip(), "Incorrect content in message output"
        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_responses_extra_body(self, openai_client):
        """Test responses with extra body parameters"""
        try:
            response = openai_client.responses.create(
                model=MODEL,
                input=[{"role": "user", "content": "hello"}],
                max_output_tokens=50,
                extra_body={
                    "top_k": 50,
                    "min_p": 0.0,
                    "min_tokens_to_keep": 1,
                },
            )
            logger.info(f"Responses Response with extra body:\n{response}\n")

            # Validate response
            assert response.model == MODEL, "Model name is not correct"
            assert response.usage is not None, "No usage in response"
            assert response.object == "response", "Incorrect response object type"
            assert response.status == "completed", "Response status is not completed"

            # Check that extra body parameters were processed
            message_output = next(
                (item for item in response.output if item.type == "message"), None
            )
            assert message_output is not None, "No message output found"
            assert message_output.content[
                0
            ].text.strip(), "Extra body parameter not processed correctly"
        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_responses_draft_model(self, openai_client):
        try:
            model = "mlx-community/Qwen3-8B-abliterated-v2-mxfp4"
            response = openai_client.responses.create(
                model=model,
                input=[{"role": "user", "content": "hello"}],
                max_output_tokens=50,
                extra_body={
                    "draft-model": MODEL,
                },
            )
            logger.info(f"Responses Response with draft model:\n{response}\n")

            # Validate response
            assert response.model == model, "Model name is not correct"
            assert response.usage is not None, "No usage in response"
            assert response.object == "response", "Incorrect response object type"
            assert response.status == "completed", "Response status is not completed"
            assert len(response.output) > 0, "No output in response"
            message_output = next(
                (item for item in response.output if item.type == "message"), None
            )
            assert message_output is not None, "No message output found"
            assert len(message_output.content) > 0, "No content in message output"
            assert message_output.content[0].text.strip(), "Generated content is empty"

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_responses_stream(self, openai_client):
        """Test basic streaming responses functionality"""
        try:
            events = []
            with openai_client.responses.stream(
                model=MODEL, input=[{"role": "user", "content": "hi"}]
            ) as stream:
                for event in stream:
                    events.append(event)
                # Get final response
                final = stream.get_final_response()

            logger.info(f"Received {len(events)} stream events")

            # Validate events
            event_types = [event.type for event in events]
            assert "response.created" in event_types, "No response.created event received"
            assert "response.completed" in event_types, "No response.completed event received"
            assert len(events) > 0, "No events received"

            # Check for text delta events
            text_deltas = [
                event.delta for event in events if event.type == "response.output_text.delta"
            ]
            content = "".join(text_deltas)

            assert content.strip(), "Generated content is empty"
            logger.info(f"Complete generated content: {content}")

            # Validate final response
            assert final.status == "completed", "Final response status is not completed"
            assert len(final.output) > 0, "No output in final response"

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_responses_stream_with_options(self, openai_client):
        """Test streaming responses with additional options"""
        try:
            events = []
            with openai_client.responses.stream(
                model=MODEL,
                input=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Write a short greeting."},
                ],
                max_output_tokens=200,
            ) as stream:
                for event in stream:
                    events.append(event)
                # Get final response
                final = stream.get_final_response()

            logger.info(f"Received {len(events)} stream events")

            # Validate events
            assert len(events) > 0, "No events received"

            # Collect different types of events
            text_deltas = [
                event.delta for event in events if event.type == "response.output_text.delta"
            ]

            assert len(text_deltas) > 0, "No text delta events received"

            # Check content
            text_content = "".join(text_deltas)
            assert text_content.strip(), "Missing expected text content"

            logger.info(f"Complete generated text content: {text_content}")

            # Validate final response
            assert final.status == "completed", "Final response status is not completed"
            assert len(final.output) > 0, "No output in final response"

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    @pytest.mark.asyncio
    async def test_retry_canceled_stream_responses(self, async_client: AsyncClient):
        """
        Tests that retrying a canceled streaming request for responses starts a new generation.
        """
        payload = {
            "model": MODEL,
            "input": [
                {
                    "role": "user",
                    "content": "Write a detailed essay about the history of artificial intelligence and machine learning.",
                }
            ],
            "stream": True,
            "max_output_tokens": 500,  # Make it longer so we have time to cancel
        }

        # --- First request, which we will cancel ---
        lines_received = []
        is_cancelled = False
        logger.info("\n--- Starting first (canceled) request ---")
        try:
            async with async_client.stream(
                "POST", "/v1/responses", json=payload, timeout=5
            ) as response:
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
        assert "[DONE]" not in full_first_response
        assert (
            'data: {"type":"response.completed"}' not in full_first_response
        ), "Canceled stream should not be complete"

        # Give the server a moment to process the disconnection
        await asyncio.sleep(1)

        # --- Second request, which should succeed ---
        all_lines = []
        logger.info("--- Starting second (retry) request ---")
        async with async_client.stream(
            "POST", "/v1/responses", json=payload, timeout=15
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    all_lines.append(line)

        full_response = "\n".join(all_lines)
        assert "[DONE]" not in full_response
        logger.info(
            f"--- Full response from second request: ---\
{full_response}"
        )

        # The second request should complete successfully
        assert "response.completed" in full_response, "Full stream should end with completed event"

        # Verify the content of the full stream
        assert len(all_lines) > len(
            lines_received
        ), "Second request should return more lines than the canceled one"

        # Check that the content is what we expect from a fresh generation
        content = ""
        for line in all_lines:
            if line.startswith("data:"):
                data_part = line[len("data: ") :].strip()
                if data_part and data_part != "[DONE]":
                    try:
                        chunk_json = json.loads(data_part)
                        if chunk_json.get("type") == "response.output_text.delta":
                            content += chunk_json.get("delta", "")
                    except json.JSONDecodeError:
                        pytest.fail(f"Failed to decode JSON chunk: {data_part}")

        logger.info(
            f"--- Reconstructed content from second request: ---\
{content}"
        )
        # Check for expected content in the response
        assert "artificial" in content.lower() or "machine" in content.lower()
        assert len(content) > 50, "Should have generated a reasonable amount of content"
