# Analysis: tests/unit/chat/test_reasoning_decoder.py

## Component Verified
Reasoning Content Decoder (Thinking Tags).

## Test Cases
1. **test_parse_response_with_thinking**:
   - Extracts content within `<think>` tags into a separate `reasoning` field.
2. **test_parse_stream_response_thinking_mode**:
   - Validates the streaming state machine that switches between "thinking" and "content" modes based on tags.
3. **test_parse_response_missing_start_tag**:
   - **Robustness**: Handles cases where the model starts outputting `</think>` (or content) without an initial `<think>` tag, recovering gracefully.

## Observations
- **Feature**: Supports the "Chain of Thought" visibility feature found in newer reasoning models.
