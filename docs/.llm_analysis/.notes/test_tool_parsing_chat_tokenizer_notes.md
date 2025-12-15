# Analysis: tests/unit/chat/test_tool_parsing_chat_tokenizer.py

## Component Verified
Streaming Tool Parser State Machine.

## Test Cases
1. **test_decode_stream_***:
   - Validates the buffering logic.
   - Ensures that partial matches (e.g., `<t`) are buffered.
   - Ensures that false positives (e.g., `< 30`) are released immediately once proven not to be a tool tag.
2. **test_tool_call_split_across_chunks**:
   - Verifies correct reassembly of tool calls that are fragmented across network/generation chunks.

## Observations
- **Criticality**: This component sits in the hot path of every token generation for streaming responses, ensuring the user doesn't see raw tool tags while waiting for the tool call to complete.
