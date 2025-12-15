# Analysis: tests/integration/chat/test_prompt_cache.py

## Component Verified
Prompt Caching mechanism in Chat Completions.

## Test Cases
1. **test_conversation_with_prompt_cache**:
   - Simulates a multi-turn conversation.
   - First request: Establishes cache.
   - Second request: Verifies `response.usage.prompt_tokens_details.cached_tokens > 0`.

## Observations
- **Constraint**: Relies on a prompt long enough (>100 tokens) to trigger the caching logic.
- **Verification**: Explicitly checks the usage stats for cache hits, which is the correct way to verify this feature.
