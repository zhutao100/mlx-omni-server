# Analysis: tests/unit/responses/test_responses_logic.py

## Component Verified
Responses API Logic (Deep Mocking).

## Test Cases
1. **test_responses_non_stream_integration**:
   - Verifies the full flow from API call to response object using a mocked model.
2. **test_responses_streaming_integration**:
   - **Protocol Verification**: Validates the complex SSE event sequence (`output_item.added`, `function_call_arguments.delta`, etc.) required by the Responses API spec.
3. **test_responses_streaming_sequential_tool_calls**:
   - Ensures multiple tool calls are correctly serialized in the stream.

## Observations
- **Complexity**: This test suite is essential because the Responses API streaming protocol is significantly more complex than standard Chat Completions.
- **Mocking**: Uses sophisticated generator mocks to simulate exact token/chunk sequences.
