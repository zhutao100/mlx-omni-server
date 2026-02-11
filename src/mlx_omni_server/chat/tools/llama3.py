from mlx_lm.tokenizer_utils import TokenizerWrapper

from .json_prefill import JsonToolPrefillChatTokenizer


class Llama3ChatTokenizer(JsonToolPrefillChatTokenizer):
    """Tools handler for Llama models."""

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(
            tokenizer,
            tool_call_start_token="<|python_tag|>",
            tool_call_end_token="",
        )
