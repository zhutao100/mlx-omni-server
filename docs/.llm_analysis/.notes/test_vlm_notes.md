# Analysis: tests/integration/chat/test_vlm.py

## Component Verified
Vision Language Model (VLM) Chat Completions.

## Test Cases
1. **test_vlm_chat_completions_normal**:
   - Tests `text` + `image_url` content blocks.
   - **Mocks** the backend model (`MockVlmModel`).
2. **test_vlm_chat_completions_streaming**:
   - Streaming version of the above.
3. **test_vlm_model_cache_manager**:
   - Tests the specific caching logic for VLM models.
4. **test_vlm_request_multimodal_detection**:
   - Unit test for the request object method `is_multimodal_request()`.

## Observations
- **Mocking**: Unlike `test_chat_completions.py`, this uses `unittest.mock` to bypass actual model inference. This is good for speed/stability but doesn't verify the actual MLX VLM integration (loading weights, processing images).
- **Scope**: Verifies the *API layer* handling of multimodal requests, not the *inference layer*.
