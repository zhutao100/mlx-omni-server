import json
import unittest
from unittest.mock import Mock

from mlx_lm.tokenizer_utils import TokenizerWrapper

from mlx_omni_server.schemas.chat_schema import Role
from mlx_omni_server.services.chat.tools.qwen2 import Qwen2ChatTokenizer


class TestQwen2ChatTokenizer(unittest.TestCase):
    def setUp(self):
        mock_tokenizer = Mock(spec=TokenizerWrapper)
        self.qwen2_tokenizer = Qwen2ChatTokenizer(mock_tokenizer)

    def test_decode_single_tool_call(self):
        # Test single tool call with double quotes
        text = """<tool_call>
{"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}}
</tool_call>"""
        result = self.qwen2_tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertEqual(result.role, Role.ASSISTANT)
        self.assertIsNone(result.content)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(len(result.tool_calls), 1)

        tool_call = result.tool_calls[0]
        self.assertEqual(tool_call.function.name, "get_current_weather")
        self.assertEqual(
            json.loads(tool_call.function.arguments),
            {"location": "Boston, MA", "unit": "fahrenheit"},
        )

    def test_decode_invalid_json(self):
        # Test invalid JSON format
        text = """<tool_call>
{"name": "get_current_weather", invalid_json}
</tool_call>"""
        result = self.qwen2_tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertEqual(result.role, Role.ASSISTANT)
        self.assertEqual(
            result.content, text
        )  # Should return original text for invalid JSON
        self.assertIsNone(result.tool_calls)

    def test_decode_invalid_tool_call(self):
        # Test invalid tool call format (missing name)
        text = """<tool_call>
{"arguments": {"location": "Boston, MA"}}
</tool_call>"""
        result = self.qwen2_tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertEqual(result.role, Role.ASSISTANT)
        self.assertEqual(
            result.content, text
        )  # Should return original text for invalid format
        self.assertIsNone(result.tool_calls)

    def test_decode_non_tool_call(self):
        # Test non-tool call text
        text = "This is a regular message without any tool calls."
        result = self.qwen2_tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertEqual(result.role, Role.ASSISTANT)
        self.assertEqual(result.content, text)
        self.assertIsNone(result.tool_calls)
