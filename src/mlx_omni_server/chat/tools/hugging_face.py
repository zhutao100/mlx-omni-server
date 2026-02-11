from mlx_lm.tokenizer_utils import TokenizerWrapper

from .json_prefill import JsonToolPrefillChatTokenizer


class HuggingFaceChatTokenizer(JsonToolPrefillChatTokenizer):
    """Tools handler for Llama models.
    https://huggingface.co/blog/unified-tool-use
    """

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(
            tokenizer,
            tool_call_start_token="<tool_call>\n",
            tool_call_end_token="</tool_call>",
        )
