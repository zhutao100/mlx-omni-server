import json
import logging
import textwrap

import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper

from mlx_omni_server.chat.schema import Function, Role, Tool, ToolType
from mlx_omni_server.chat.tools.chat_tokenizer import ToolParsingChatTokenizer
from mlx_omni_server.chat.tools.minimax_m2 import MinimaxM2ToolParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockTokenizer(TokenizerWrapper):
    def __init__(self):
        pass

    def encode(self, text):
        return [ord(c) for c in text] if text else []

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens) if tokens else ""

    def apply_chat_template(
        self, conversation, tools=None, tokenize=False, add_generation_prompt=False, **kwargs
    ):
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])


@pytest.fixture
def minimax_parser():
    return MinimaxM2ToolParser()


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def minimax_chat_tokenizer(mock_tokenizer, minimax_parser):
    tokenizer = ToolParsingChatTokenizer(mock_tokenizer)
    tokenizer.tool_parser = minimax_parser
    return tokenizer


@pytest.fixture
def sample_tools():
    return [
        Tool(
            type=ToolType.FUNCTION,
            function=Function(
                name="get_weather",
                description="Get the current weather in a given location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            ),
        ),
        Tool(
            type=ToolType.FUNCTION,
            function=Function(
                name="get_stock_price",
                description="Get the current stock price for a given symbol",
                parameters={
                    "type": "object",
                    "properties": {"symbol": {"type": "string"}},
                    "required": ["symbol"],
                },
            ),
        ),
    ]


class TestMinimaxM2ToolParser:
    def test_tool_call_parsing(self, minimax_parser, sample_tools):
        text = textwrap.dedent(
            """
            <minimax:tool_call>
            <invoke name="get_weather">
            <parameter name="location">
            Boston, MA
            </parameter>
            <parameter name="unit">celsius</parameter>
            </invoke>
            </minimax:tool_call>
            """
        )
        rest_text, tool_calls = minimax_parser.extract_tool_calls(text, tools=sample_tools)
        assert rest_text.strip() == ""
        assert tool_calls is not None
        assert len(tool_calls) == 1
        tool_call = tool_calls[0]
        assert tool_call.function.name == "get_weather"
        assert json.loads(tool_call.function.arguments) == {
            "location": "Boston, MA",
            "unit": "celsius",
        }

    def test_multiple_tool_calls(self, minimax_parser, sample_tools):
        text = textwrap.dedent(
            """
            Here is the weather:
            <minimax:tool_call>
            <invoke name="get_weather">
            <parameter name="location">New York, NY</parameter>
            </invoke>
            </minimax:tool_call>
            And the stock price:
            <minimax:tool_call>
            <invoke name="get_stock_price">
            <parameter name="symbol">AAPL</parameter>
            </invoke>
            </minimax:tool_call>
            """
        )
        rest_text, tool_calls = minimax_parser.extract_tool_calls(text, tools=sample_tools)
        assert "Here is the weather:" in rest_text
        assert "And the stock price:" in rest_text
        assert tool_calls is not None
        assert len(tool_calls) == 2

        weather_call = next(tc for tc in tool_calls if tc.function.name == "get_weather")
        stock_call = next(tc for tc in tool_calls if tc.function.name == "get_stock_price")
        assert json.loads(weather_call.function.arguments) == {"location": "New York, NY"}
        assert json.loads(stock_call.function.arguments) == {"symbol": "AAPL"}

    def test_no_tool_calls(self, minimax_parser, sample_tools):
        text = "This is a regular message with no tool calls."
        rest_text, tool_calls = minimax_parser.extract_tool_calls(text, tools=sample_tools)
        assert rest_text == text
        assert not tool_calls

    def test_unknown_tool_call(self, minimax_parser, sample_tools):
        text = textwrap.dedent(
            """
            <minimax:tool_call>
            <invoke name="unknown_function">
            <parameter name="some_param">some_value</parameter>
            </invoke>
            </minimax:tool_call>
            """
        )
        rest_text, tool_calls = minimax_parser.extract_tool_calls(text, tools=sample_tools)
        assert not tool_calls
        assert "unknown_function" in rest_text

    def test_tool_call_with_nested_xml_like_content(self, minimax_parser, sample_tools):
        text = textwrap.dedent(
            """
            <minimax:tool_call>
            <invoke name="get_weather">
            <parameter name="location">
            Someplace with <weird> formatting </weird> and another </parameter> tag
            </parameter>
            </invoke>
            </minimax:tool_call>
            """
        )
        rest_text, tool_calls = minimax_parser.extract_tool_calls(text, tools=sample_tools)
        assert rest_text.strip() == ""
        assert tool_calls is not None
        assert len(tool_calls) == 1
        args = json.loads(tool_calls[0].function.arguments)
        assert (
            args["location"]
            == "Someplace with <weird> formatting </weird> and another </parameter> tag"
        )

    def test_incomplete_tool_call_with_invoke_only(self, minimax_parser, sample_tools):
        text = '<minimax:tool_call><invoke name="get_weather">'
        rest_text, tool_calls = minimax_parser.extract_tool_calls(text, tools=sample_tools)
        assert rest_text.strip() == ""
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        assert json.loads(tool_calls[0].function.arguments) == {}


class TestStreaming:
    def test_minimax_streaming_with_indentation(self, minimax_chat_tokenizer, sample_tools):
        minimax_chat_tokenizer.buffer = ""
        minimax_chat_tokenizer.potential_tool_start_pos = -1

        prefix = "Here is the weather:\n"
        first = minimax_chat_tokenizer.decode_stream(prefix, sample_tools)
        assert first is not None
        assert first.role == Role.ASSISTANT
        assert first.content == prefix
        assert first.tool_calls is None

        tool_call_text = (
            "  <minimax:tool_call>\n"
            '  <invoke name="get_weather">\n'
            '  <parameter name="location">\n'
            "  New York, NY\n"
            "  </parameter>\n"
            "  </invoke>\n"
            "  </minimax:tool_call>\n"
        )

        emitted_content = []
        for ch in tool_call_text:
            msg = minimax_chat_tokenizer.decode_stream(ch, sample_tools)
            if msg is not None and msg.content:
                emitted_content.append(str(msg.content))

        final = minimax_chat_tokenizer.parse_buffer(sample_tools)
        assert final is not None
        assert final.role == Role.ASSISTANT
        assert final.tool_calls is not None
        assert len(final.tool_calls) == 1
        assert final.tool_calls[0].function.name == "get_weather"
        assert json.loads(final.tool_calls[0].function.arguments) == {"location": "New York, NY"}

        leaked = "".join(emitted_content)
        assert "<minimax:tool_call>" not in leaked
        assert "<invoke" not in leaked
        assert "<parameter" not in leaked

    def test_minimax_streaming_tool_start_split_across_chunks(
        self, minimax_chat_tokenizer, sample_tools
    ):
        minimax_chat_tokenizer.buffer = ""
        minimax_chat_tokenizer.potential_tool_start_pos = -1

        part1 = "Hello\n<minimax:tool_c"
        msg1 = minimax_chat_tokenizer.decode_stream(part1, sample_tools)
        assert msg1 is not None
        assert msg1.role == Role.ASSISTANT
        assert msg1.content == "Hello\n"
        assert "<minimax:tool_c" not in str(msg1.content)

        part2 = (
            "all>\n"
            '<invoke name="get_weather">\n'
            '<parameter name="location">\n'
            "Boston, MA\n"
            "</parameter>\n"
            "</invoke>\n"
            "</minimax:tool_call>\n"
        )
        msg2 = minimax_chat_tokenizer.decode_stream(part2, sample_tools)
        assert msg2 is None or (msg2.content and "<minimax:tool_call>" not in str(msg2.content))

        final = minimax_chat_tokenizer.parse_buffer(sample_tools)
        assert final is not None
        assert final.tool_calls is not None
        assert len(final.tool_calls) == 1
        assert final.tool_calls[0].function.name == "get_weather"
        assert json.loads(final.tool_calls[0].function.arguments) == {"location": "Boston, MA"}
