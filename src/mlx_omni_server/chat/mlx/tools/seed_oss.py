
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .chat_tokenizer import ToolParsingChatTokenizer
from .qwen3 import Qwen3ToolParser


class SeedOssToolParser(Qwen3ToolParser):
    """Tool parser for Seed OSS XML format that converts to OpenAI JSON format."""

    def __init__(self, strict: bool = False, tool_call_start_token="<seed:tool_call>", tool_call_end_token="</seed:tool_call>"):
        super().__init__(strict=strict, tool_call_start_token=tool_call_start_token, tool_call_end_token=tool_call_end_token)


class SeedOssChatTokenizer(ToolParsingChatTokenizer):
    """Tools handler for Seed OSS models with XML tool parsing support."""

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(tokenizer, thinking_tag="seed:think")
        self.tool_parser = SeedOssToolParser()
