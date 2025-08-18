import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from outlines.models.transformers import TransformerTokenizer
from outlines.processors import JSONLogitsProcessor

from ..schema import ResponseFormat


class OutlinesLogitsProcessor:
    processed_token_count: int = 0

    def __init__(self, tokenizer: TokenizerWrapper, response_format: ResponseFormat):
        json_schema = response_format.json_schema.schema_def
        self.logits_processor = JSONLogitsProcessor(
            json_schema,
            TransformerTokenizer(tokenizer._tokenizer),
            tensor_library_name="mlx",
        )

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        # `tokens` is the full sequence of tokens (prompt + generated).
        # The Outlines processor is stateful and needs the full sequence.

        # Outlines processor expects 1D logits
        logits_1d = logits.reshape(-1)

        # Ensure logits are float32 for the processor
        # The result from outlines is a numpy array, convert it back to mx.array
        processed_logits = self.logits_processor(tokens, logits_1d.astype(mx.float32))

        # Reshape to the original logits shape.
        return mx.array(processed_logits).reshape(logits.shape)
