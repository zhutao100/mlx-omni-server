# Analysis: tests/integration/chat/test_chat_completions.py

## Component Verified
Chat Completions API (`/v1/chat/completions`).

## Test Cases
1. **test_chat_completions_normal**:
   - Basic request/response cycle.
   - Validates model name, usage fields, object type, and message content.
2. **test_chat_completions_extra_body**:
   - Verifies `extra_body` parameter handling (e.g., `top_k`, `min_p`).
3. **test_chat_completions_draft_model**:
   - Tests speculative decoding/draft model functionality.
4. **test_chat_completions_stream**:
   - Validates streaming responses (SSE).
   - Checks chunk structure and content accumulation.
5. **test_chat_completions_stream_options**:
   - Tests streaming with `include_usage` option.
   - Verifies usage stats are sent at the end of the stream.
6. **test_retry_canceled_stream_chat_completion** (Async):
   - Simulates client disconnection (cancellation) during streaming.
   - Verifies the server handles it gracefully and a subsequent request succeeds.
   - **Critical** for verifying server stability under load/flakiness.

## Observations
- **Quality**: High. Tests cover standard usage, streaming, advanced features (draft models), and error recovery (cancellation).
- **Hardcoded Model**: Uses `mlx-community/Qwen3-1.7B-4bit-DWQ-053125`.
- **Patterns**: usage of `openai_client` makes tests readable and standard-compliant.
