
import json
import logging
import pytest
from unittest.mock import patch
from openai import OpenAI
from fastapi.testclient import TestClient
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_omni_server.main import app
from mlx_omni_server.chat.mlx.model_types import load_tools_handler, load_config
from unittest.mock import Mock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def openai_client(client):
    """Create OpenAI client configured with test server"""
    return OpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=client,
    )


class TestQwen3ToolCalls:
    @patch('mlx_omni_server.chat.mlx.model_types.load_config')
    def test_qwen3_model_type_detection(self, mock_load_config, openai_client):
        """Test that Qwen3 model type is correctly detected and handled"""
        # Mock the model config to return qwen3 model type
        mock_load_config.return_value = {"model_type": "qwen3"}
        
        # Create a mock tokenizer
        mock_tokenizer = Mock(spec=TokenizerWrapper)
        
        # Test that the correct tokenizer is loaded for qwen3
        tokenizer = load_tools_handler("qwen3", mock_tokenizer)
        from mlx_omni_server.chat.mlx.tools.qwen3 import Qwen3ChatTokenizer
        assert isinstance(tokenizer, Qwen3ChatTokenizer)
        
        # Test that the correct tokenizer is loaded for qwen3_moe
        tokenizer = load_tools_handler("qwen3_moe", mock_tokenizer)
        assert isinstance(tokenizer, Qwen3ChatTokenizer)


    @patch('mlx_omni_server.chat.mlx.model_types.load_config')
    def test_qwen3_tool_call(self, mock_load_config, openai_client):
        # Mock the model config to return qwen3 model type
        mock_load_config.return_value = {"model_type": "qwen3"}
        
        request = {
            "model": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr9e8",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Boston?"
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA"
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"]
                                }
                            },
                            "required": ["location"]
                        }
                    }
                }
            ]
        }

        response = openai_client.chat.completions.create(**request)

        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.message.tool_calls is not None
        assert len(choice.message.tool_calls) == 1
        tool_call = choice.message.tool_calls[0]
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_current_weather"
        arguments = json.loads(tool_call.function.arguments)
        assert "location" in arguments
        assert str(arguments["location"]).startswith("Boston")
        assert choice.message.content is ""

    @patch('mlx_omni_server.chat.mlx.model_types.load_config')
    def test_qwen3_tool_call_stream(self, mock_load_config, openai_client):
        # Mock the model config to return qwen3 model type
        mock_load_config.return_value = {"model_type": "qwen3"}
        
        request = {
            "model": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr9e8",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Boston?"
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA"
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"]
                                }
                            },
                            "required": ["location"]
                        }
                    }
                }
            ],
            "stream": True
        }

        tool_calls = []
        for chunk in openai_client.chat.completions.create(**request):
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.tool_calls:
                tool_calls.extend(chunk.choices[0].delta.tool_calls)

        assert len(tool_calls) == 1
        tool_call = tool_calls[0]
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_current_weather"
        # In streaming mode, the arguments are accumulated, so we can't reliably parse the JSON
        # until the end of the stream. We can, however, check that the arguments contain the
        # expected keys.
        assert "location" in tool_call.function.arguments
        assert "Boston" in tool_call.function.arguments
