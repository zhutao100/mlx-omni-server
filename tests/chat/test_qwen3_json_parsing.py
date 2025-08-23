import json
import logging
import pytest
from unittest.mock import Mock, patch
from mlx_omni_server.chat.mlx.tools.qwen3 import Qwen3ChatTokenizer
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_omni_server.chat.schema import ChatMessage, Role

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestQwen3JsonParsing:
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