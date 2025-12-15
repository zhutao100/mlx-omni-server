# Analysis: tests/integration/chat/test_stream_finish_reason.py

## Component Verified
Streaming Protocol Compliance (specifically `finish_reason`).

## Test Cases
1. **test_stream_finish_reason_stop**:
   - Verifies the last chunk contains `finish_reason="stop"`.
2. **test_stream_finish_reason_with_usage**:
   - Verifies that when `stream_options={"include_usage": True}` is used, `finish_reason` is still present and correct, alongside usage stats.
3. **test_async_stream_finish_reason**:
   - Async client version of the above.

## Observations
- **Compliance**: This is a critical compatibility test for OpenAI client libraries, which often rely on `finish_reason` to terminate loops.
