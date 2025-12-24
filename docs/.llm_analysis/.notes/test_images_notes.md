# Analysis: tests/integration/images/test_images.py

## Component Verified
Image Generation API (`/v1/images/generations`).

## Test Cases
1. **test_images**:
   - Standard generation returning a URL.
2. **test_images_b64_json**:
   - Generation returning `b64_json`.

## Observations
- **Model**: Uses `filipstrand/Z-Image-Turbo-mflux-4bit`.
- **Constraint**: These tests are expensive (time/compute) if running against real models.

## Flaky Failure Root Cause (Fixed)
- When both tests run in the same process, `ImagesService` reuses a cached `MFluxImageGenerator` (and cached `ZImageTurbo` model instance).
- In low-RAM mode, `mflux.callbacks.instances.memory_saver.MemorySaver` unloads encoders by setting `model.text_encoder = None`.
- The next request uses a different prompt, and `PromptEncoder.encode_prompt(...)` calls the now-`None` `text_encoder`, raising `'NoneType' object is not callable` and returning HTTP 500.
