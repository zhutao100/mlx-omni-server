# Analysis: tests/integration/responses/test_responses_integration.py

## Component Verified
Responses API (New/Experimental/Custom endpoint).

## Test Cases
1. **test_responses_normal**:
   - Basic request/response.
   - Checks `response.object == "response"` and `status == "completed"`.
2. **test_responses_stream**:
   - Verifies streaming events: `response.created`, `response.output_text.delta`, `response.completed`.
   - **Protocol**: This follows a different protocol than `chat.completions` (SSE events have different types).
3. **test_retry_canceled_stream_responses**:
   - Robustness test for canceling and retrying streams on this endpoint.

## Observations
- **Protocol**: Validates a specific event-driven response format that differs from standard chat completions.
