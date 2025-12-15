# Analysis: tests/integration/images/test_images.py

## Component Verified
Image Generation API (`/v1/images/generations`).

## Test Cases
1. **test_images**:
   - Standard generation returning a URL.
2. **test_images_b64_json**:
   - Generation returning `b64_json`.

## Observations
- **Model**: Uses `dhairyashil/FLUX.1-schnell-mflux-4bit`.
- **Constraint**: These tests are expensive (time/compute) if running against real models.
