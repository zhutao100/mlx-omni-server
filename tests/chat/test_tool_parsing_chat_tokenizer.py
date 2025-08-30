import unittest
from unittest.mock import Mock

from mlx_lm.tokenizer_utils import TokenizerWrapper

from mlx_omni_server.chat.mlx.tools.chat_tokenizer import ToolParsingChatTokenizer
from mlx_omni_server.chat.mlx.tools.tool_parser import BaseToolParser
from mlx_omni_server.chat.schema import Role, Tool, ToolCall, FunctionCall


class MockToolParser(BaseToolParser):
    """Mock tool parser for testing."""

    def __init__(self, start_token="<tool>", end_token="</tool>"):
        self.tool_call_start_token = start_token
        self.tool_call_end_token = end_token

    def extract_tool_calls(self, model_output: str, tools=None):
        """More robust mock implementation that finds tool calls anywhere in the string."""
        start_idx = model_output.find(self.tool_call_start_token)
        end_idx = model_output.find(self.tool_call_end_token)

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            content_before = model_output[:start_idx]
            tool_content_str = model_output[start_idx + len(self.tool_call_start_token):end_idx]

            tool_call = ToolCall(
                id="call_test123",
                function=FunctionCall(
                    name="test_function",
                    arguments=f'{{"param": "{tool_content_str}"}}'
                )
            )
            # For simplicity, this mock ignores content after the tool call
            return content_before, [tool_call]

        return model_output, None


class TestToolParsingChatTokenizer(unittest.TestCase):

    def setUp(self):
        mock_tokenizer = Mock(spec=TokenizerWrapper)
        self.tokenizer = ToolParsingChatTokenizer(mock_tokenizer)
        # Use a more specific tool token for tests to avoid ambiguity with HTML/XML
        self.tokenizer.tool_parser = MockToolParser(start_token="<tool_code>", end_token="</tool_code>")

    def test_decode_stream_normal_text(self):
        """Test that normal text streams through without buffering."""
        result1 = self.tokenizer.decode_stream("Hello ")
        self.assertIsNotNone(result1)
        self.assertEqual(result1.role, Role.ASSISTANT)
        self.assertEqual(result1.content, "Hello ")

        result2 = self.tokenizer.decode_stream("world!")
        self.assertIsNotNone(result2)
        self.assertEqual(result2.role, Role.ASSISTANT)
        self.assertEqual(result2.content, "world!")

        # Buffer should be empty
        self.assertEqual(self.tokenizer.buffer, "")
        self.assertEqual(self.tokenizer.potential_tool_start_pos, -1)

    def test_decode_stream_false_positive_less_than(self):
        """Test that text with '<' but not matching tool start token streams correctly."""
        result1 = self.tokenizer.decode_stream("The price is ")
        self.assertIsNotNone(result1)
        self.assertEqual(result1.content, "The price is ")

        result2 = self.tokenizer.decode_stream("< $100.")
        self.assertIsNotNone(result2)
        self.assertEqual(result2.content, "< $100.")

        # Buffer should be empty after processing
        self.assertEqual(self.tokenizer.buffer, "")
        self.assertEqual(self.tokenizer.potential_tool_start_pos, -1)

    def test_decode_stream_tool_start_detection(self):
        """Test that actual tool start token is detected and buffered."""
        result1 = self.tokenizer.decode_stream("Hello ")
        self.assertIsNotNone(result1)
        self.assertEqual(result1.content, "Hello ")

        result2 = self.tokenizer.decode_stream("World<tool_code>test")
        self.assertIsNotNone(result2)
        self.assertEqual(result2.content, "World")  # Only content before <tool_code>

        self.assertEqual(self.tokenizer.buffer, "<tool_code>test")
        self.assertEqual(self.tokenizer.potential_tool_start_pos, 0)

    def test_decode_stream_continued_after_tool_start(self):
        """Test that streaming stops after tool start is detected."""
        result1 = self.tokenizer.decode_stream("Hello<tool_code>")
        self.assertIsNotNone(result1)
        self.assertEqual(result1.content, "Hello")

        result2 = self.tokenizer.decode_stream("test_content")
        self.assertIsNone(result2)  # Should be None since we're in tool mode

        self.assertEqual(self.tokenizer.buffer, "<tool_code>test_content")

    def test_parse_buffer_with_valid_tool_call(self):
        """Test that parse_buffer correctly extracts tool calls."""
        self.tokenizer.buffer = "<tool_code>test_content</tool_code>"
        self.tokenizer.potential_tool_start_pos = 0

        result = self.tokenizer.parse_buffer()

        self.assertIsNotNone(result)
        self.assertEqual(result.role, Role.ASSISTANT)
        self.assertIsNone(result.content)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "test_function")

        self.assertEqual(self.tokenizer.buffer, "")
        self.assertEqual(self.tokenizer.potential_tool_start_pos, -1)

    def test_tool_call_in_single_chunk(self):
        """Tests a full tool call received in a single delta."""
        delta = "Here is a tool call: <tool_code>print('hello')</tool_code>"

        msg1 = self.tokenizer.decode_stream(delta)
        self.assertIsNotNone(msg1)
        self.assertEqual(msg1.content, "Here is a tool call: ")

        final_msg = self.tokenizer.parse_buffer()
        self.assertIsNotNone(final_msg)
        self.assertIsNotNone(final_msg.tool_calls)
        self.assertEqual(len(final_msg.tool_calls), 1)
        self.assertEqual(final_msg.tool_calls[0].function.arguments, '{"param": "print(\'hello\')"}')
        self.assertIsNone(final_msg.content)  # No content besides the tool call

    def test_tool_call_split_across_chunks(self):
        """Tests a tool call that is split into multiple streaming deltas."""
        deltas = ["Thinking... ", "<tool_code>", "arg1", "+arg2", "</tool_code>", " Done."]

        msg1 = self.tokenizer.decode_stream(deltas[0])
        self.assertEqual(msg1.content, "Thinking... ")
        self.assertEqual(self.tokenizer.buffer, "")

        self.assertIsNone(self.tokenizer.decode_stream(deltas[1]))
        self.assertEqual(self.tokenizer.buffer, "<tool_code>")
        self.assertEqual(self.tokenizer.potential_tool_start_pos, 0)

        self.assertIsNone(self.tokenizer.decode_stream(deltas[2]))
        self.assertIsNone(self.tokenizer.decode_stream(deltas[3]))
        self.assertIsNone(self.tokenizer.decode_stream(deltas[4]))
        self.assertEqual(self.tokenizer.buffer, "<tool_code>arg1+arg2</tool_code>")

        tool_msg = self.tokenizer.parse_buffer()
        self.assertIsNotNone(tool_msg)
        self.assertIsNotNone(tool_msg.tool_calls)
        self.assertEqual(len(tool_msg.tool_calls), 1)
        self.assertEqual(tool_msg.tool_calls[0].function.arguments, '{"param": "arg1+arg2"}')
        self.assertIsNone(tool_msg.content)
        self.assertEqual(self.tokenizer.buffer, "")

        msg6 = self.tokenizer.decode_stream(deltas[5])
        self.assertEqual(msg6.content, " Done.")
        final_msg = self.tokenizer.parse_buffer()
        self.assertIsNone(final_msg)

    def test_partial_start_token_buffering(self):
        """Tests that a partial start token at the end of a delta is correctly buffered."""
        deltas = ["Here is a partial start <tool", "_code>content</tool_code>"]

        msg1 = self.tokenizer.decode_stream(deltas[0])
        self.assertEqual(msg1.content, "Here is a partial start ")
        self.assertEqual(self.tokenizer.buffer, "<tool")
        self.assertEqual(self.tokenizer.potential_tool_start_pos, -1)

        msg2 = self.tokenizer.decode_stream(deltas[1])
        self.assertIsNone(msg2)

        final_msg = self.tokenizer.parse_buffer()
        self.assertIsNotNone(final_msg.tool_calls)
        self.assertEqual(len(final_msg.tool_calls), 1)
        self.assertEqual(final_msg.tool_calls[0].function.arguments, '{"param": "content"}')

    def test_false_positive_partial_tool_start(self):
        """Tests handling of text that looks like a partial tool start but isn't."""
        deltas = ["Text with < that is not a tool.", " More text."]

        msg1 = self.tokenizer.decode_stream(deltas[0])
        self.assertEqual(msg1.content, "Text with < that is not a tool.")
        self.assertEqual(self.tokenizer.buffer, "")

        msg2 = self.tokenizer.decode_stream(deltas[1])
        self.assertEqual(msg2.content, " More text.")
        self.assertEqual(self.tokenizer.buffer, "")

    def test_stream_ending_with_partial_token(self):
        """Tests when the stream ends right after a partial start token."""
        delta = "This is some text <tool_co"
        msg = self.tokenizer.decode_stream(delta)
        self.assertEqual(msg.content, "This is some text ")

        final_msg = self.tokenizer.parse_buffer()
        self.assertEqual(final_msg.content, "<tool_co")
        self.assertIsNone(final_msg.tool_calls)

    def test_empty_and_whitespace_deltas(self):
        """Tests that empty or whitespace deltas don't disrupt the parsing."""
        deltas = ["Hello", "", " ", "\n", "World"]

        content = ""
        for delta in deltas:
            msg = self.tokenizer.decode_stream(delta)
            if msg and msg.content:
                content += msg.content

        final_msg = self.tokenizer.parse_buffer()
        if final_msg and final_msg.content:
            content += final_msg.content

        self.assertEqual(content, "Hello \nWorld")

    def test_stream_with_diverging_tool_token(self):
        """Tests a stream that starts like a tool call but then diverges."""
        # 1. Stream a partial token that looks like a valid start
        msg1 = self.tokenizer.decode_stream("Thinking... <tool")
        self.assertEqual(msg1.content, "Thinking... ")
        self.assertEqual(self.tokenizer.buffer, "<tool")

        # 2. Stream text that makes the sequence invalid
        msg2 = self.tokenizer.decode_stream("_foo> is not a tool.")
        # The buffer becomes "<tool_foo> is not a tool."
        # The tokenizer should realize this is not a valid tool call and release it.
        self.assertIsNotNone(msg2)
        self.assertEqual(msg2.content, "<tool_foo> is not a tool.")
        self.assertEqual(self.tokenizer.buffer, "")

        # 3. Ensure subsequent text is handled normally
        msg3 = self.tokenizer.decode_stream(" The end.")
        self.assertIsNotNone(msg3)
        self.assertEqual(msg3.content, " The end.")
        self.assertEqual(self.tokenizer.buffer, "")

    def test_parse_buffer_with_false_positive(self):
        """Test that parse_buffer handles false positives correctly."""
        self.tokenizer.buffer = "Hello < world"
        self.tokenizer.potential_tool_start_pos = 6

        result = self.tokenizer.parse_buffer()

        self.assertIsNotNone(result)
        self.assertEqual(result.role, Role.ASSISTANT)
        self.assertEqual(result.content, "Hello < world")
        self.assertIsNone(result.tool_calls)

        self.assertEqual(self.tokenizer.buffer, "")
        self.assertEqual(self.tokenizer.potential_tool_start_pos, -1)

    def test_decode_stream_buffer_release_on_overflow(self):
        """Test that buffer releases content when it gets too large without tool start."""
        long_text = "x" * 100
        result = self.tokenizer.decode_stream(long_text)

        self.assertIsNotNone(result)
        self.assertTrue(len(result.content) > 0)
        self.assertTrue(len(self.tokenizer.buffer) < len(self.tokenizer.tool_parser.tool_call_start_token))

    def test_decode_stream_complex_scenario(self):
        """Test a complex scenario with mixed content and false positives."""
        self.tokenizer.decode_stream("The temperature is ")
        self.tokenizer.decode_stream("< 30 degrees")
        self.tokenizer.decode_stream("<tool_code>get_weather")
        self.tokenizer.decode_stream("{'location': 'Boston'}")
        self.tokenizer.decode_stream("</tool_code>")

        final_result = self.tokenizer.parse_buffer()
        self.assertIsNotNone(final_result)
        self.assertIsNone(final_result.content)
        self.assertIsNotNone(final_result.tool_calls)
        self.assertEqual(len(final_result.tool_calls), 1)
        self.assertEqual(final_result.tool_calls[0].function.name, "test_function")

    def test_no_extra_characters_with_tool_token(self):
        """Test that no extra characters are added when tool_call_start_token is '<tool_code>'."""
        result1 = self.tokenizer.decode_stream("Hello ")
        result2 = self.tokenizer.decode_stream("world<tool_code>test_content</tool_code>")
        final_result = self.tokenizer.parse_buffer()

        all_content = ''.join(c.content for c in [result1, result2, final_result] if c and c.content)
        self.assertEqual(all_content, "Hello world")

    def test_no_extra_characters_with_different_tool_token(self):
        """Test that no extra characters are added when tool_call_start_token is '[TOOL_USE]'."""
        tokenizer = ToolParsingChatTokenizer(Mock(spec=TokenizerWrapper))
        tokenizer.tool_parser = MockToolParser("[TOOL_USE]", "[/TOOL_USE]")

        result1 = tokenizer.decode_stream("Hello ")
        result2 = tokenizer.decode_stream("world[TOOL_USE]test_content[/TOOL_USE]")
        final_result = tokenizer.parse_buffer()

        all_content = ''.join(c.content for c in [result1, result2, final_result] if c and c.content)
        self.assertEqual(all_content, "Hello world")

    def test_no_extra_less_than_at_end(self):
        """Test that no extra '<' appears at end of output when partial tool start is buffered."""
        result1 = self.tokenizer.decode_stream("Hello world")
        result2 = self.tokenizer.decode_stream("The price is <")
        result3 = self.tokenizer.decode_stream("100 dollars")
        final_result = self.tokenizer.parse_buffer()

        all_content = ''.join(c.content for c in [result1, result2, result3, final_result] if c and c.content)
        self.assertFalse(all_content.endswith('<'))
        self.assertEqual(all_content, "Hello worldThe price is <100 dollars")


if __name__ == '__main__':
    unittest.main()
