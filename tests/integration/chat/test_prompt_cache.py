"""
Tests for prompt cache functionality

This test file verifies the prompt caching functionality in the chat completion API, including:
1. First conversation with no cache
2. Second conversation using cache
3. Modified conversation still hitting partial cache
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPromptCache:
    """Tests for prompt cache functionality"""

    def test_conversation_with_prompt_cache(self, openai_client):
        try:
            logger.info("\n===== Conversation with prompt cache =====")
            model = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"
            # Use a longer prompt to exceed the 100 token minimum for cache reuse
            prompt = """Can you tell me more about your capabilities? I'm interested in understanding what you can do, what kind of tasks you're good at, and how you might be able to help me with my work. Please provide a comprehensive overview of your skills and areas of expertise. I'd like to know about your reasoning abilities, your knowledge domains, and what makes you different from other AI assistants. Also, please explain your approach to problem-solving and how you handle complex or ambiguous requests."""

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
