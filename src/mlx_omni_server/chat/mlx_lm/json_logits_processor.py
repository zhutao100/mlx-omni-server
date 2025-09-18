import mlx.core as mx
import torch
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import \
    build_transformers_prefix_allowed_tokens_fn
from mlx_lm.tokenizer_utils import TokenizerWrapper

from ..schema import ResponseFormat


class JsonLogitsProcessor:
    """
    A logits processor that enforces JSON schema constraints using
    lm-format-enforcer.

    Compatible with mlx arrays.
    """

    processed_token_count: int = 0

    def __init__(self, tokenizer: TokenizerWrapper, response_format: ResponseFormat):
        if response_format.type != "json_schema":
            raise ValueError("JsonLogitsProcessor only supports type='json_schema'")
        if not response_format.json_schema:
            raise ValueError("JsonLogitsProcessor requires a json_schema in response_format")

        json_schema = response_format.json_schema.schema_def
        parser = JsonSchemaParser(json_schema)
        # build_transformers_prefix_allowed_tokens_fn returns a callable(batch_id, input_ids)
        self.prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer._tokenizer, parser)

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        """
        Args:
            tokens (mx.array): Full sequence of token ids (prompt + generated).
            logits (mx.array): Logits for the next token.

        Returns:
            mx.array: Processed logits with invalid tokens masked out.
        """
        # Determine the sequence from tokens
        if tokens.ndim == 1:
            seq = tokens
        elif tokens.ndim == 2 and tokens.shape[0] == 1:
            seq = tokens[0]
        else:
            raise ValueError(f"Unsupported tokens shape: {tokens.shape}")

        # Convert to torch.LongTensor for the prefix function
        input_ids = torch.tensor(seq.tolist(), dtype=torch.long)

        # Get allowed tokens (list of int)
        allowed_tokens = self.prefix_fn(0, input_ids)

        # Flatten logits to 1D
        logits_shape = logits.shape
        logits_1d = logits.reshape(-1)

        # Create masked logits
        masked_logits = mx.full((logits_1d.size,), -mx.inf, dtype=logits_1d.dtype)
        if allowed_tokens:
            allowed_arr = mx.array(allowed_tokens)
            masked_logits[allowed_arr] = logits_1d[allowed_arr]

        # Reshape back to original logits shape
        return masked_logits.reshape(logits_shape)
