# Analysis: tests/unit/responses/test_responses_router.py

## Component Verified
Responses API Adapter Layer (`/v1/responses`).

## Test Cases
1. **test_response_request_to_chat_request_***:
   - These tests verify the *Translation Layer*.
   - The Responses API accepts a different input format (e.g., `input` list, `instructions`) than the internal Chat API.
   - These tests ensure that history, tools, and messages are correctly mapped to the `ChatCompletionRequest` format used by the underlying engine.

## Observations
- **Architecture**: Reveals that `/v1/responses` is likely a facade/adapter over the core `/v1/chat/completions` engine.
