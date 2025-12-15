# Analysis: tests/unit/chat/test_router_cache_logic.py

## Component Verified
Cache Eviction and Hashing Logic.

## Test Cases
1. **test_request_hashing**:
   - Verifies stable SHA256 hashing of requests (ignoring key order).
2. **test_cache_cleanup_***:
   - Verifies TTL-based eviction.
   - **Critical**: Ensures `StreamCacheEntry` is NOT evicted if `active_clients > 0`, even if the TTL has passed (prevents disconnecting slow readers).

## Observations
- **Complexity**: High. Managing cache lifecycles for streaming responses is non-trivial.
