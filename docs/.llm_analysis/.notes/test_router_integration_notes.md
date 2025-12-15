# Analysis: tests/integration/chat/test_router_integration.py

## Component Verified
Request Caching, Idempotency, and Streaming Concurrency.

## Test Cases
1. **test_non_streaming_cache_integration**:
   - Verifies repeated requests return the same ID and content (Idempotency).
   - Checks `x-idempotent-replay` header.
2. **test_streaming_cache_two_clients_integration**:
   - Simulates two concurrent clients hitting the same stream.
   - Verifies both receive identical chunks.
   - Verifies internal `StreamCacheEntry` state.
3. **test_streaming_cache_late_client_integration**:
   - "Late joiner" scenario: Client 2 joins after Client 1 has started/finished.
   - Verifies Client 2 receives the full replay.
4. **test_streaming_emits_final_chunk_before_done**:
   - **Complex Async Test**: Uses `asyncio` futures and event loop manipulation.
   - Goal: Ensure the protocol sends the final data chunk (with `finish_reason`) *before* the `[DONE]` sentinel.

## Observations
- **Sophistication**: High. This is testing complex async state management and protocol details.
- **Mocks**: Uses `MockTextModel` to simulate generation, focusing the test on the *router/server logic* rather than the model inference.
