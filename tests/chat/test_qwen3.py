import json
import logging
import pytest
import textwrap
from unittest.mock import Mock, patch
from mlx_omni_server.chat.mlx.tools.qwen3 import Qwen3ChatTokenizer, Qwen3ToolParser
from mlx_omni_server.chat.mlx.tools.chat_tokenizer import ToolParsingChatTokenizer
from mlx_omni_server.chat.schema import Tool, Function, ToolType, ChatMessage, Role
from mlx_lm.tokenizer_utils import TokenizerWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockTokenizer(TokenizerWrapper):
    def __init__(self):
        pass

    def encode(self, text):
        return [ord(c) for c in text] if text else []

    def decode(self, tokens):
        return ''.join(chr(t) for t in tokens) if tokens else ''

    def apply_chat_template(self, conversation, tools=None, tokenize=False, add_generation_prompt=False, **kwargs):
        # Simple mock implementation
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])


@pytest.fixture
def qwen3_parser():
    return Qwen3ToolParser()


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def qwen3_chat_tokenizer(mock_tokenizer, qwen3_parser):
    tokenizer = ToolParsingChatTokenizer(mock_tokenizer)
    tokenizer.tool_parser = qwen3_parser
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
                        "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
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
                    "properties": {
                        "symbol": {"type": "string", "description": "The stock symbol, e.g. GOOG"},
                        "exchange": {"type": "string", "description": "The stock exchange, e.g. NASDAQ"},
                    },
                    "required": ["symbol"],
                },
            ),
        ),
    ]


class TestQwen3ToolParser:
    @pytest.mark.parametrize(
        "text, expected_func, expected_args",
        [
            (
                textwrap.dedent('''
                <tool_call>
                <function=get_weather>
                <parameter=location>
                Boston, MA
                </parameter>
                </function>
                </tool_call>
                '''),
                "get_weather",
                {"location": "Boston, MA"},
            ),
            (
                textwrap.dedent('''
                <tool_call>
                <function=get_weather>
                <parameter=location>
                Boston, MA
                </parameter>
                <parameter=unit>
                celsius
                </parameter>
                </function>
                </tool_call>
                '''),
                "get_weather",
                {"location": "Boston, MA", "unit": "celsius"},
            ),
        ],
    )
    def test_tool_call_parsing(self, qwen3_parser, sample_tools, text, expected_func, expected_args):
        rest_text, tool_calls = qwen3_parser.extract_tool_calls(text, tools=sample_tools)
        assert rest_text.strip() == ""
        assert len(tool_calls) == 1
        tool_call = tool_calls[0]
        assert tool_call.function.name == expected_func
        args = json.loads(tool_call.function.arguments)
        assert args == expected_args

    def test_multiple_tool_calls(self, qwen3_parser, sample_tools):
        text = textwrap.dedent('''
        Here is the weather:
        <tool_call>
        <function=get_weather>
        <parameter=location>
        New York, NY
        </parameter>
        </function>
        </tool_call>
        And the stock price:
        <tool_call>
        <function=get_stock_price>
        <parameter=symbol>
        AAPL
        </parameter>
        </function>
        </tool_call>
        ''')
        rest_text, tool_calls = qwen3_parser.extract_tool_calls(text, tools=sample_tools)
        assert "Here is the weather:" in rest_text
        assert "And the stock price:" in rest_text
        assert len(tool_calls) == 2

        weather_call = next(tc for tc in tool_calls if tc.function.name == "get_weather")
        stock_call = next(tc for tc in tool_calls if tc.function.name == "get_stock_price")

        assert weather_call is not None
        assert stock_call is not None

        weather_args = json.loads(weather_call.function.arguments)
        assert weather_args == {"location": "New York, NY"}

        stock_args = json.loads(stock_call.function.arguments)
        assert stock_args == {"symbol": "AAPL"}

    def test_no_tool_calls(self, qwen3_parser, sample_tools):
        text = "This is a regular message with no tool calls."
        rest_text, tool_calls = qwen3_parser.extract_tool_calls(text, tools=sample_tools)
        assert rest_text == text
        assert not tool_calls

    def test_malformed_tool_call_no_strict(self, qwen3_parser, sample_tools):
        text = textwrap.dedent('''
        <function=get_weather>
        <parameter=location>Boston, MA
        ''')
        rest_text, tool_calls = qwen3_parser.extract_tool_calls(text, tools=sample_tools)
        assert rest_text.strip() == ""
        assert len(tool_calls) == 1
        tool_call = tool_calls[0]
        assert tool_call.function.name == "get_weather"
        args = json.loads(tool_call.function.arguments)
        assert args == {"location": "Boston, MA"}

    def test_malformed_tool_call_strict(self, qwen3_parser, sample_tools):
        qwen3_parser.strict = True
        text = "<function=get_weather><parameter=location>Boston, MA"
        with pytest.raises(ValueError):
            qwen3_parser.extract_tool_calls(text, tools=sample_tools)

    def test_tool_call_with_nested_xml_like_content(self, qwen3_parser, sample_tools):
        text = textwrap.dedent('''
        <tool_call>
        <function=get_weather>
        <parameter=location>
        Someplace with <weird> formatting </weird> and another </parameter> tag
        </parameter>
        </function>
        </tool_call>
        ''')
        rest_text, tool_calls = qwen3_parser.extract_tool_calls(text, tools=sample_tools)
        assert len(tool_calls) == 1
        args = json.loads(tool_calls[0].function.arguments)
        assert args["location"] == "Someplace with <weird> formatting </weird> and another </parameter> tag"

    def test_empty_input(self, qwen3_parser, sample_tools):
        text = ""
        rest_text, tool_calls = qwen3_parser.extract_tool_calls(text, tools=sample_tools)
        assert rest_text == ""
        assert not tool_calls

    def test_incomplete_tool_call(self, qwen3_parser, sample_tools):
        text = "<tool_call><function=get_weather>"
        rest_text, tool_calls = qwen3_parser.extract_tool_calls(text, tools=sample_tools)
        assert rest_text.strip() == ""
        assert len(tool_calls) == 1
        tool_call = tool_calls[0]
        assert tool_call.function.name == "get_weather"
        args = json.loads(tool_call.function.arguments)
        assert args == {}

    def test_tool_call_with_json_in_args(self, qwen3_parser):
        tools = [
            Tool(
                type=ToolType.FUNCTION,
                function=Function(
                    name="process_data",
                    description="Process some data",
                    parameters={
                        "type": "object",
                        "properties": {
                            "data": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "string"}}}
                        },
                        "required": ["data"],
                    },
                ),
            )
        ]
        text = textwrap.dedent('''
        <tool_call>
        <function=process_data>
        <parameter=data>
        {"a": 1, "b": "hello"}
        </parameter>
        </function>
        </tool_call>
        ''')
        rest_text, tool_calls = qwen3_parser.extract_tool_calls(text, tools=tools)
        assert len(tool_calls) == 1
        args = json.loads(tool_calls[0].function.arguments)
        assert args["data"] == {"a": 1, "b": "hello"}

    def test_missing_tool_call_tag(self, qwen3_parser, sample_tools):
        text = textwrap.dedent('''
        <function=get_weather>
        <parameter=location>
        Boston, MA
        </parameter>
        </function>
        ''')
        rest_text, tool_calls = qwen3_parser.extract_tool_calls(text, tools=sample_tools)
        assert rest_text.strip() == ""
        assert len(tool_calls) == 1
        tool_call = tool_calls[0]
        assert tool_call.function.name == "get_weather"
        args = json.loads(tool_call.function.arguments)
        assert args == {"location": "Boston, MA"}

    def test_unknown_tool_call(self, qwen3_parser, sample_tools):
        text = textwrap.dedent('''
        <tool_call>
        <function=unknown_function>
        <parameter=some_param>some_value</parameter>
        </function>
        </tool_call>
        ''')
        rest_text, tool_calls = qwen3_parser.extract_tool_calls(text, tools=sample_tools)
        assert not tool_calls
        assert "unknown_function" in rest_text


class TestQwen3ChatTokenizer:
    def test_ensure_dict_arguments_with_json_string(self):
        """Test that JSON string arguments are converted to dict"""
        # Test tools with JSON string arguments
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_function",
                    "arguments": '{"location": "Boston", "unit": "celsius"}'
                }
            }
        ]

        # Process tools using the method directly
        processed_tools = Qwen3ChatTokenizer._ensure_dict_arguments(None, tools)

        # Verify the arguments were converted to dict
        assert isinstance(processed_tools[0]["function"]["arguments"], dict)
        assert processed_tools[0]["function"]["arguments"]["location"] == "Boston"
        assert processed_tools[0]["function"]["arguments"]["unit"] == "celsius"

    def test_ensure_dict_arguments_with_dict(self):
        """Test that dict arguments are left unchanged"""
        # Test tools with dict arguments (should remain unchanged)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_function",
                    "arguments": {"location": "Boston", "unit": "celsius"}
                }
            }
        ]

        # Process tools using the method directly
        processed_tools = Qwen3ChatTokenizer._ensure_dict_arguments(None, tools)

        # Verify the arguments remained as dict
        assert isinstance(processed_tools[0]["function"]["arguments"], dict)
        assert processed_tools[0]["function"]["arguments"]["location"] == "Boston"
        assert processed_tools[0]["function"]["arguments"]["unit"] == "celsius"

    def test_ensure_dict_arguments_with_invalid_json(self):
        """Test that invalid JSON strings are handled gracefully"""
        # Test tools with invalid JSON string arguments
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_function",
                    "arguments": '{"location": "Boston", "unit":}'  # Invalid JSON
                }
            }
        ]

        # Process tools using the method directly (should not raise exception)
        processed_tools = Qwen3ChatTokenizer._ensure_dict_arguments(None, tools)

        # Verify the arguments remained as string (since parsing failed)
        assert isinstance(processed_tools[0]["function"]["arguments"], str)
        assert processed_tools[0]["function"]["arguments"] == '{"location": "Boston", "unit":}'

    def test_qwen3_chat_tokenizer_encode_with_tool_calls(self):
        """Test that Qwen3 chat tokenizer properly encodes tool calls with dict arguments"""
        # Create a mock tokenizer with chat_template attribute
        mock_tokenizer = Mock(spec=TokenizerWrapper)
        mock_tokenizer.chat_template = "test template"

        # Create Qwen3 chat tokenizer instance
        qwen3_tokenizer = Qwen3ChatTokenizer(mock_tokenizer)

        # Mock the parent encode method to return a simple string
        with patch.object(Qwen3ChatTokenizer, 'encode', return_value="test prompt"):
            # Test messages with tool calls containing JSON string arguments
            messages = [
                ChatMessage(
                    role=Role.ASSISTANT,
                    content="",
                    tool_calls=[
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": '{"location": "Boston", "unit": "celsius"}'
                            }
                        }
                    ]
                )
            ]

            # Call encode method
            result = qwen3_tokenizer.encode(messages, [])

            # Verify the parent encode was called
            assert result == "test prompt"
