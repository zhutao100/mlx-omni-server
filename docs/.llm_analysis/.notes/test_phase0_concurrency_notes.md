# Analysis: tests/integration/concurrency/test_phase0_concurrency.py

## Component Verified
Concurrency Control (MLX Gate) and Resource Isolation.

## Test Cases
1. **test_tts_uses_request_scoped_temp_paths**:
   - Simulates concurrent TTS requests.
   - Verifies that each request writes to a unique path (thread/async safety).
2. **test_mlx_gate_serializes_across_endpoints**:
   - **Critical Architecture Test**: Verifies that the "MLX Gate" correctly serializes requests to different endpoints (TTS vs Embeddings) that compete for the GPU.
   - Uses timing assertions to ensure execution intervals do not overlap.

## Observations
- **Methodology**: Uses `monkeypatch` to inject delays and verify serialization.
- **Importance**: Ensures the server doesn't crash or OOM when multiple different model types are requested simultaneously.
