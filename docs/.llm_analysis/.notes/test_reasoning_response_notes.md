# Analysis: tests/integration/chat/test_reasoning_response.py

## Component Verified
Reasoning/Thinking models (e.g., DeepSeek R1, Qwen-Reasoning).

## Test Cases
1. **test_streaming_reasoning_response**:
   - Basic streaming test for reasoning models.
2. **test_reasoning_response**:
   - Verifies that the `message` object contains a `reasoning` attribute (custom extension).
   - Notes the potential presence of `</think>` tags.
3. **test_none_reasoning_response**:
   - Tests `extra_body={"enable_thinking": False}`.
   - Verifies `reasoning` attribute is absent and content does not contain thinking tags.

## Observations
- **Custom Extension**: Tests a non-standard OpenAI API property (`reasoning`).
- **Control**: Verifies the ability to disable thinking output.
