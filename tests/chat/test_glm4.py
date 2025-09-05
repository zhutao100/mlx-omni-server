import json
import re
import pytest
from mlx_omni_server.chat.mlx.tools.glm4 import Glm4ToolParser
from mlx_omni_server.chat.mlx.tools.chat_tokenizer import ToolParsingChatTokenizer
from mlx_omni_server.chat.schema import Tool, Function, ToolType, Role, FunctionCall, ToolCall
from mlx_lm.tokenizer_utils import TokenizerWrapper


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
def glm4_parser():
    return Glm4ToolParser()


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def glm4_chat_tokenizer(mock_tokenizer, glm4_parser):
    tokenizer = ToolParsingChatTokenizer(mock_tokenizer)
    tokenizer.tool_parser = glm4_parser
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


class TestGlm4ToolParser:
    @pytest.mark.parametrize(
        "text, expected_args",
        [
            (
                '''
                <tool_call>get_weather
                <arg_key>location</arg_key>
                <arg_value>Boston, MA</arg_value>
                </tool_call>
                ''',
                {"location": "Boston, MA"},
            ),
            (
                '''
                <tool_call>get_weather
                <arg_key>location</arg_key>
                <arg_value>Boston, MA</arg_value>
                <arg_key>unit</arg_key>
                <arg_value>celsius</arg_value>
                </tool_call>
                ''',
                {"location": "Boston, MA", "unit": "celsius"},
            ),
        ],
    )
    def test_tool_call_parsing(self, glm4_parser, sample_tools, text, expected_args):
        rest_text, tool_calls = glm4_parser.extract_tool_calls(text, tools=sample_tools)
        assert rest_text.strip() == ""
        assert len(tool_calls) == 1
        tool_call = tool_calls[0]
        assert tool_call.function.name == "get_weather"
        args = json.loads(tool_call.function.arguments)
        assert args == expected_args

    def test_multiple_tool_calls(self, glm4_parser, sample_tools):
        text = '''
        Here is the weather:
        <tool_call>get_weather
        <arg_key>location</arg_key>
        <arg_value>New York, NY</arg_value>
        </tool_call>
        And the stock price:
        <tool_call>get_stock_price
        <arg_key>symbol</arg_key>
        <arg_value>AAPL</arg_value>
        </tool_call>
        '''
        rest_text, tool_calls = glm4_parser.extract_tool_calls(text, tools=sample_tools)
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

    def test_no_tool_calls(self, glm4_parser, sample_tools):
        text = "This is a regular message with no tool calls."
        rest_text, tool_calls = glm4_parser.extract_tool_calls(text, tools=sample_tools)
        assert rest_text == text
        assert not tool_calls

    def test_malformed_tool_call_no_strict(self, glm4_parser, sample_tools):
        text = "<tool_call>get_weather<arg_key>location</arg_key>"
        rest_text, tool_calls = glm4_parser.extract_tool_calls(text, tools=sample_tools)
        assert rest_text.strip() == ""
        assert len(tool_calls) == 1
        tool_call = tool_calls[0]
        assert tool_call.function.name == "get_weather"
        args = json.loads(tool_call.function.arguments)
        assert args == {}
    
        text = '''                                                                                                                                                                      
        <tool_call>
        <tool_call>get_weather
        <arg_key>location</arg_key>
        <arg_value>Boston, MA</arg_value>
        <arg_key>unit</arg_key>
        <arg_value>celsius</arg_value>
        </tool_call>
        '''
        rest_text, tool_calls = glm4_parser.extract_tool_calls(text, tools=sample_tools)
        assert rest_text.strip() == ""
        assert len(tool_calls) == 1
        tool_call = tool_calls[0]
        assert tool_call.function.name == "get_weather"
        args = json.loads(tool_call.function.arguments)
        assert args == {"location": "Boston, MA", "unit": "celsius"} 

    def test_malformed_tool_call_strict(self, glm4_parser, sample_tools):
        glm4_parser.strict = True
        text = "<tool_call>get_weather<arg_key>location</arg_key>"
        with pytest.raises(ValueError):
            glm4_parser.extract_tool_calls(text, tools=sample_tools)

    def test_tool_call_with_nested_xml_like_content(self, glm4_parser, sample_tools):
        text = '''
        <tool_call>get_weather
        <arg_key>location</arg_key>
        <arg_value>Someplace with <weird> formatting</weird></arg_value>
        </tool_call>
        '''
        rest_text, tool_calls = glm4_parser.extract_tool_calls(text, tools=sample_tools)
        assert len(tool_calls) == 1
        args = json.loads(tool_calls[0].function.arguments)
        assert args["location"] == "Someplace with <weird> formatting</weird>"

    def test_empty_input(self, glm4_parser, sample_tools):
        text = ""
        rest_text, tool_calls = glm4_parser.extract_tool_calls(text, tools=sample_tools)
        assert rest_text == ""
        assert not tool_calls

    def test_incomplete_tool_call(self, glm4_parser, sample_tools):
        text = "<tool_call>get_weather"
        rest_text, tool_calls = glm4_parser.extract_tool_calls(text, tools=sample_tools)
        assert rest_text.strip() == ""
        assert len(tool_calls) == 1
        tool_call = tool_calls[0]
        assert tool_call.function.name == "get_weather"
        args = json.loads(tool_call.function.arguments)
        assert args == {}

    def test_tool_call_with_json_in_args(self, glm4_parser):
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
        text = '''
        <tool_call>process_data
        <arg_key>data</arg_key>
        <arg_value>{"a": 1, "b": "hello"}</arg_value>
        </tool_call>
        '''
        rest_text, tool_calls = glm4_parser.extract_tool_calls(text, tools=tools)
        assert len(tool_calls) == 1
        args = json.loads(tool_calls[0].function.arguments)
        assert args["data"] == {"a": 1, "b": "hello"}

    def test_alternate_function_tag_format(self, glm4_parser, sample_tools):
        text = '''
        <tool_call>
        <function=get_weather>
        <arg_key>location</arg_key>
        <arg_value>Miami, FL</arg_value>
        </tool_call>
        '''
        rest_text, tool_calls = glm4_parser.extract_tool_calls(text, tools=sample_tools)
        assert rest_text.strip() == ""
        assert len(tool_calls) == 1
        tool_call = tool_calls[0]
        assert tool_call.function.name == "get_weather"
        args = json.loads(tool_call.function.arguments)
        assert args == {"location": "Miami, FL"}

    @pytest.mark.parametrize(
        "text, expected_args",
        [
            (
                '''
                get_weather
                <arg_key>location</arg_key>
                <arg_value>Boston, MA</arg_value>
                </tool_call>
                ''',
                {"location": "Boston, MA"},
            ),
            (
                '''
                get_weather
                <arg_key>location</arg_key>
                <arg_value>New York, NY</arg_value>
                <arg_key>unit</arg_key>
                <arg_value>fahrenheit</arg_value>
                </tool_call>
                ''',
                {"location": "New York, NY", "unit": "fahrenheit"},
            ),
        ],
    )
    def test_missing_tool_call_tag(self, glm4_parser, sample_tools, text, expected_args):
        tool_call = glm4_parser.parse_tool_call_block(text, tools=sample_tools)
        assert tool_call is not None
        assert tool_call.type == ToolType.FUNCTION
        assert tool_call.function.name == "get_weather"
        args = json.loads(tool_call.function.arguments)
        assert args == expected_args

    def test_missing_tool_call_tag_invalid_function(self, glm4_parser, sample_tools):
        text = '''
        get_cat_fact
        <arg_key>topic</arg_key>
        <arg_value>space</arg_value>
        </tool_call>
        '''
        tool_call = glm4_parser.parse_tool_call_block(text, tools=sample_tools)
        assert tool_call is None

    def test_extract_tool_calls_missing_tag(self, glm4_parser, sample_tools):
        text = '''
        get_weather
        <arg_key>location</arg_key>
        <arg_value>Boston, MA</arg_value>
        </tool_call>
        '''
        rest_text, tool_calls = glm4_parser.extract_tool_calls(text, tools=sample_tools)
        assert rest_text.strip() == ""
        assert len(tool_calls) == 1
        tool_call = tool_calls[0]
        assert tool_call.function.name == "get_weather"
        args = json.loads(tool_call.function.arguments)
        assert args == {"location": "Boston, MA"}

    def test_extract_tool_calls_mixed_valid_and_malformed(self, glm4_parser, sample_tools):
        text = '''
        Here is the weather:
        <tool_call>get_weather
        <arg_key>location</arg_key>
        <arg_value>New York, NY</arg_value>
        </tool_call>
        And the stock price:
        get_stock_price
        <arg_key>symbol</arg_key>
        <arg_value>AAPL</arg_value>
        </tool_call>
        '''
        rest_text, tool_calls = glm4_parser.extract_tool_calls(text, tools=sample_tools)
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

    def test_extract_tool_calls_with_no_valid_tool_name(self, glm4_parser, sample_tools):
        text = '''
        get_cat_fact
        <arg_key>topic</arg_key>
        <arg_value>space</arg_value>
        </tool_call>
        '''
        rest_text, tool_calls = glm4_parser.extract_tool_calls(text, tools=sample_tools)
        assert "get_cat_fact" in rest_text
        assert not tool_calls

    def test_unknown_tool_call(self, glm4_parser, sample_tools):
        text = '''
        <tool_call>unknown_function
        <arg_key>some_param</arg_key>
        <arg_value>some_value</arg_value>
        </tool_call>
        '''
        rest_text, tool_calls = glm4_parser.extract_tool_calls(text, tools=sample_tools)
        assert not tool_calls
        assert "unknown_function" in rest_text

    def test_glm4_tool_pattern_detection(self, glm4_parser, sample_tools):
        """Test that GLM4 parser can detect tool patterns without start tokens."""
        glm4_parser.update_tool_start_pattern(sample_tools)

        assert glm4_parser.tool_start_pattern is not None

        # Test matching tool names followed by "<"
        match = glm4_parser.tool_start_pattern.search("get_weather<")
        assert match is not None
        assert match.group(1) == "get_weather"

        # Test matching tool names followed by "\n"
        match = glm4_parser.tool_start_pattern.search("get_weather\n")
        assert match is not None
        assert match.group(1) == "get_weather"

        # Test not matching tool names not followed by "<" or "\n"
        match = glm4_parser.tool_start_pattern.search("get_weather")
        assert match is None

        match = glm4_parser.tool_start_pattern.search("  get_stock_price")
        assert match is None


class TestToolParsingChatTokenizer:
    def test_glm4_parsing_with_properly_formatted_tool_calls(self, glm4_chat_tokenizer, sample_tools):
        text = '''
        get_weather
        <arg_key>location</arg_key>
        <arg_value>New York, NY</arg_value>
        '''
        result = glm4_chat_tokenizer.decode(text, sample_tools)

        assert result is not None
        assert result.role == Role.ASSISTANT
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert json.loads(result.tool_calls[0].function.arguments) == {"location": "New York, NY"}

    def test_glm4_parsing_capability_for_missing_start_tokens(self, glm4_chat_tokenizer, sample_tools):
        parser = glm4_chat_tokenizer.tool_parser
        text = "get_weather\n<arg_key>location</arg_key>\n<arg_value>Boston, MA</arg_value>\n"
        content, tool_calls = parser.extract_tool_calls(text, sample_tools)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        assert json.loads(tool_calls[0].function.arguments) == {"location": "Boston, MA"}
        assert content.strip() == ""

    def test_glm4_tokenizer_integration_with_pattern_matching(self, glm4_chat_tokenizer, sample_tools):
        parser = glm4_chat_tokenizer.tool_parser
        parser.update_tool_start_pattern(sample_tools)

        assert parser.tool_start_pattern is not None

        # Test matching tool names followed by "<"
        match = parser.tool_start_pattern.search("get_weather<")
        assert match is not None
        assert match.group(1) == "get_weather"

        text = "get_stock_price\n<arg_key>symbol</arg_key>\n<arg_value>GOOG</arg_value>\n"
        result = glm4_chat_tokenizer.decode(text, sample_tools)

        assert result is not None
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_stock_price"
        assert json.loads(result.tool_calls[0].function.arguments) == {"symbol": "GOOG"}

    def test_glm4_parsing_without_tools_still_works(self, glm4_chat_tokenizer):
        text = "get_weather\n<arg_key>location</arg_key>\n<arg_value>Miami, FL</arg_value>\n"
        result = glm4_chat_tokenizer.decode(text, None)

        assert result is not None
        assert result.role == Role.ASSISTANT
        assert result.content is not None
        assert "get_weather" in result.content
        assert "Miami, FL" in result.content
        assert result.tool_calls is None or len(result.tool_calls) == 0

    def test_tool_parsing_chat_tokenizer_with_glm4_pattern(self, mock_tokenizer, glm4_parser, sample_tools):
        chat_tokenizer = ToolParsingChatTokenizer(mock_tokenizer)
        chat_tokenizer.tool_parser = glm4_parser

        # Test with proper tool pattern (tool name followed by <)
        chat_tokenizer.decode_stream("get_weather<", sample_tools)
        assert glm4_parser.tool_start_pattern is not None

        chat_tokenizer.buffer = "get_stock_price\n"
        chat_tokenizer.parse_buffer(sample_tools)
        assert glm4_parser.tool_start_pattern is not None


class TestStreaming:
    def test_glm4_streaming_with_missing_start_token(self, glm4_chat_tokenizer, sample_tools):
        stream_text = "Here is the weather forecast:\n"
        result = glm4_chat_tokenizer.decode_stream(stream_text, sample_tools)
        assert result is not None
        assert result.role == Role.ASSISTANT
        assert result.content == stream_text
        assert result.tool_calls is None

        tool_call_text = "get_weather\n<arg_key>location</arg_key>\n<arg_value>New York, NY</arg_value>\n"

        for char in tool_call_text:
            glm4_chat_tokenizer.decode_stream(char, sample_tools)

        # The streaming implementation buffers potential tool calls and processes them in parse_buffer
        # During streaming, the text is returned as content messages
        final_result = glm4_chat_tokenizer.parse_buffer(sample_tools)
        assert final_result is not None
        assert final_result.role == Role.ASSISTANT
        assert final_result.tool_calls is not None
        assert len(final_result.tool_calls) == 1
        assert final_result.tool_calls[0].function.name == "get_weather"
        assert json.loads(final_result.tool_calls[0].function.arguments) == {"location": "New York, NY"}

    def test_glm4_streaming_mixed_content_and_tool_calls(self, glm4_chat_tokenizer, sample_tools):
        glm4_chat_tokenizer.buffer = ""
        glm4_chat_tokenizer.potential_tool_start_pos = -1

        stream_parts = [
            "Here's the weather:\n",
            "get_weather\n<arg_key>location</arg_key>\n<arg_value>Boston, MA</arg_value>\n",
            "And the stock price:\n",
            "\nget_stock_price\n<arg_key>symbol</arg_key>\n<arg_value>AAPL</arg_value>\n"
        ]

        all_results = []

        for part in stream_parts:
            for char in part:
                result = glm4_chat_tokenizer.decode_stream(char, sample_tools)
                if result is not None:
                    all_results.append(result)

        final_result = glm4_chat_tokenizer.parse_buffer(sample_tools)
        if final_result is not None:
            all_results.append(final_result)

        content_parts = [r for r in all_results if r.content and not r.tool_calls]
        tool_call_results = [r for r in all_results if r.tool_calls]

        assert len(content_parts) > 0

        # The streaming implementation buffers potential tool calls and processes them in parse_buffer
        # During streaming, partial content is returned as content messages
        # The final parse_buffer call should process the buffered tool calls
        total_tool_calls = sum(len(r.tool_calls) for r in tool_call_results if r.tool_calls)
        assert total_tool_calls >= 2

        all_tool_calls = []
        for result in tool_call_results:
            if result.tool_calls:
                all_tool_calls.extend(result.tool_calls)

        tool_names = [tc.function.name for tc in all_tool_calls]
        assert "get_weather" in tool_names
        assert "get_stock_price" in tool_names

    def test_glm4_streaming_vs_non_streaming_consistency(self, glm4_chat_tokenizer, sample_tools):
        text_with_missing_start = "get_weather\n<arg_key>location</arg_key>\n<arg_value>San Francisco, CA</arg_value>\n"

        non_streaming_result = glm4_chat_tokenizer.decode(text_with_missing_start, sample_tools)

        glm4_chat_tokenizer.buffer = ""
        glm4_chat_tokenizer.potential_tool_start_pos = -1

        for char in text_with_missing_start:
            glm4_chat_tokenizer.decode_stream(char, sample_tools)

        # The streaming implementation buffers potential tool calls and processes them in parse_buffer
        streaming_result = glm4_chat_tokenizer.parse_buffer(sample_tools)

        assert non_streaming_result is not None
        assert streaming_result is not None

        assert non_streaming_result.tool_calls is not None
        assert streaming_result.tool_calls is not None
        assert len(non_streaming_result.tool_calls) == len(streaming_result.tool_calls)

        non_stream_tool = non_streaming_result.tool_calls[0]
        stream_tool = streaming_result.tool_calls[0]

        assert non_stream_tool.function.name == stream_tool.function.name
        assert non_stream_tool.function.arguments == stream_tool.function.arguments

    def test_glm4_streaming_with_invalid_tool_name(self, glm4_chat_tokenizer, sample_tools):
        glm4_chat_tokenizer.buffer = ""
        glm4_chat_tokenizer.potential_tool_start_pos = -1

        invalid_tool_text = "get_invalid_tool\n<arg_key>param</arg_key>\n<arg_value>value</arg_value>\n"

        results = []
        for char in invalid_tool_text:
            result = glm4_chat_tokenizer.decode_stream(char, sample_tools)
            if result is not None:
                results.append(result)

        final_result = glm4_chat_tokenizer.parse_buffer(sample_tools)
        if final_result is not None:
            results.append(final_result)

        all_content = "".join([r.content for r in results if r.content])
        assert "get_invalid_tool" in all_content

    def test_glm4_streaming_empty_tools_list(self, glm4_chat_tokenizer):
        glm4_chat_tokenizer.buffer = ""
        glm4_chat_tokenizer.potential_tool_start_pos = -1

        text = "get_weather\n<arg_key>location</arg_key>\n<arg_value>Boston, MA</arg_value>\n"

        results = []
        for char in text:
            result = glm4_chat_tokenizer.decode_stream(char, [])
            if result is not None:
                results.append(result)

        final_result = glm4_chat_tokenizer.parse_buffer([])
        if final_result is not None:
            results.append(final_result)

        all_content = "".join([r.content for r in results if r.content])
        assert "get_weather" in all_content
        assert "Boston, MA" in all_content
