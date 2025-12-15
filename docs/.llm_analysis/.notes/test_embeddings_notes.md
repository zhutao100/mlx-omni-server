# Analysis: tests/integration/embeddings/test_embeddings.py

## Component Verified
Embeddings API (`/v1/embeddings`).

## Test Cases
1. **test_embeddings_single_text**: Basic single-string embedding.
2. **test_embeddings_multiple_texts**: Batch embedding (list of strings).
3. **test_embeddings_with_dimensions**: Verifies `dimensions` parameter handling.
4. **test_embeddings_missing_model/input**: Verifies 422 validation errors.

## Observations
- **Model**: Uses `mlx-community/all-MiniLM-L6-v2-4bit` (efficient for testing).
- **Coverage**: Good coverage of standard usage and error states.
