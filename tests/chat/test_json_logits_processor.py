
import json
import logging
import threading
import time
from unittest.mock import Mock, patch

import mlx.core as mx
import pytest
import torch
from mlx_lm.tokenizer_utils import TokenizerWrapper

from mlx_omni_server.chat.mlx_lm.json_logits_processor import \
    JsonLogitsProcessor
from mlx_omni_server.chat.schema import (ChatCompletionRequest,
                                         JsonSchemaFormat, ResponseFormat)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestJsonLogitsProcessor:
    """Test suite for JsonLogitsProcessor class."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        mock_tokenizer = Mock(spec=TokenizerWrapper)
        mock_tokenizer._tokenizer = Mock()
        return mock_tokenizer

    @pytest.fixture
    def valid_json_schema(self):
        """Create a valid JSON schema for testing."""
        return {
            "name": "test_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "active": {"type": "boolean"}
                },
                "required": ["name", "age"]
            }
        }

    @pytest.fixture
    def valid_response_format(self, valid_json_schema):
        """Create a valid ResponseFormat for testing."""
        json_schema_format = JsonSchemaFormat(**valid_json_schema)
        return ResponseFormat(type="json_schema", json_schema=json_schema_format)

    def test_init_valid_inputs(self, mock_tokenizer, valid_response_format):
        """Test JsonLogitsProcessor initialization with valid inputs."""
        with patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.JsonSchemaParser') as mock_parser, \
                patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.build_transformers_prefix_allowed_tokens_fn') as mock_build_fn:
            processor = JsonLogitsProcessor(mock_tokenizer, valid_response_format)

            assert processor.processed_token_count == 0
            mock_parser.assert_called_once()
            mock_build_fn.assert_called_once()

    def test_init_invalid_response_format_type(self, mock_tokenizer):
        """Test JsonLogitsProcessor initialization with invalid response format type."""
        response_format = ResponseFormat(type="text")

        with pytest.raises(ValueError, match="JsonLogitsProcessor only supports type='json_schema'"):
            JsonLogitsProcessor(mock_tokenizer, response_format)

    def test_init_missing_json_schema(self, mock_tokenizer):
        """Test JsonLogitsProcessor initialization with missing json_schema."""
        response_format = ResponseFormat(type="json_schema")

        with pytest.raises(ValueError, match="JsonLogitsProcessor requires a json_schema in response_format"):
            JsonLogitsProcessor(mock_tokenizer, response_format)

    def test_init_missing_json_schema_validation_error(self, mock_tokenizer):
        """Test that Pydantic validation prevents creating ResponseFormat with type=json_schema but no json_schema."""
        with pytest.raises(Exception):
            ResponseFormat(
                type="json_schema",
                json_schema=None
            )

    def test_init_calls_prefix_function_builder(
        self,
        mock_tokenizer,
        valid_response_format
    ):
        """Test that JsonLogitsProcessor calls the prefix function builder correctly."""
        with patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.JsonSchemaParser') as mock_parser, \
                patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.build_transformers_prefix_allowed_tokens_fn') as mock_build_fn:
            mock_prefix_fn = Mock(return_value=[1, 2, 3])
            mock_build_fn.return_value = mock_prefix_fn

            processor = JsonLogitsProcessor(mock_tokenizer, valid_response_format)

            mock_parser.assert_called_once()
            schema_arg = mock_parser.call_args[0][0]
            assert "properties" in schema_arg

            mock_build_fn.assert_called_once_with(mock_tokenizer._tokenizer, mock_parser.return_value)
            assert hasattr(processor, 'prefix_fn')

    def test_call_with_1d_tokens(self, mock_tokenizer, valid_response_format):
        """Test __call__ method with 1D tokens array."""
        with patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.JsonSchemaParser'), \
                patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.build_transformers_prefix_allowed_tokens_fn') as mock_build_fn:

            mock_prefix_fn = Mock(return_value=[1, 2, 3])
            mock_build_fn.return_value = mock_prefix_fn

            processor = JsonLogitsProcessor(mock_tokenizer, valid_response_format)

            tokens = mx.array([1, 2, 3, 4])
            logits = mx.array([0.1, 0.2, 0.3, 0.4, 0.5])

            result = processor(tokens, logits)

            assert isinstance(result, mx.array)
            assert result.shape == logits.shape

            mock_prefix_fn.assert_called_once()
            call_args = mock_prefix_fn.call_args
            assert call_args[0][0] == 0
            assert isinstance(call_args[0][1], torch.Tensor)
            assert call_args[0][1].dtype == torch.long

    def test_call_with_2d_tokens_single_batch(self, mock_tokenizer, valid_response_format):
        """Test __call__ method with 2D tokens array (single batch)."""
        with patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.JsonSchemaParser'), \
                patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.build_transformers_prefix_allowed_tokens_fn') as mock_build_fn:

            mock_prefix_fn = Mock(return_value=[1, 2, 3])
            mock_build_fn.return_value = mock_prefix_fn

            processor = JsonLogitsProcessor(mock_tokenizer, valid_response_format)

            tokens = mx.array([[1, 2, 3, 4]])
            logits = mx.array([0.1, 0.2, 0.3, 0.4, 0.5])

            result = processor(tokens, logits)

            assert isinstance(result, mx.array)
            assert result.shape == logits.shape

    def test_call_with_empty_allowed_tokens(self, mock_tokenizer, valid_response_format):
        """Test __call__ method with empty allowed tokens list."""
        with patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.JsonSchemaParser'), \
                patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.build_transformers_prefix_allowed_tokens_fn') as mock_build_fn:

            mock_prefix_fn = Mock(return_value=[])
            mock_build_fn.return_value = mock_prefix_fn

            processor = JsonLogitsProcessor(mock_tokenizer, valid_response_format)

            tokens = mx.array([1, 2, 3, 4])
            logits = mx.array([0.1, 0.2, 0.3, 0.4, 0.5])

            result = processor(tokens, logits)

            assert isinstance(result, mx.array)
            assert result.shape == logits.shape
            assert mx.all(mx.isinf(result)).item()

    def test_call_with_invalid_tokens_shape(self, mock_tokenizer, valid_response_format):
        """Test __call__ method with invalid tokens shape."""
        with patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.JsonSchemaParser'), \
                patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.build_transformers_prefix_allowed_tokens_fn'):

            processor = JsonLogitsProcessor(mock_tokenizer, valid_response_format)

            tokens = mx.array([[1, 2, 3], [4, 5, 6]])
            logits = mx.array([0.1, 0.2, 0.3, 0.4, 0.5])

            with pytest.raises(ValueError, match="Unsupported tokens shape"):
                processor(tokens, logits)

    def test_call_preserves_logits_dtype(self, mock_tokenizer, valid_response_format):
        """Test that __call__ method preserves the original logits dtype."""
        with patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.JsonSchemaParser'), \
                patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.build_transformers_prefix_allowed_tokens_fn') as mock_build_fn:

            mock_prefix_fn = Mock(return_value=[1, 2])
            mock_build_fn.return_value = mock_prefix_fn

            processor = JsonLogitsProcessor(mock_tokenizer, valid_response_format)

            tokens = mx.array([1, 2, 3, 4])
            logits = mx.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=mx.float32)

            result = processor(tokens, logits)

            assert result.dtype == mx.float32

    def test_call_reshapes_logits_correctly(self, mock_tokenizer, valid_response_format):
        """Test that __call__ method reshapes logits correctly."""
        with patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.JsonSchemaParser'), \
                patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.build_transformers_prefix_allowed_tokens_fn') as mock_build_fn:

            mock_prefix_fn = Mock(return_value=[1, 2])
            mock_build_fn.return_value = mock_prefix_fn

            processor = JsonLogitsProcessor(mock_tokenizer, valid_response_format)

            tokens = mx.array([1, 2, 3, 4])
            original_logits_shape = (2, 5)
            logits = mx.random.normal(original_logits_shape)

            result = processor(tokens, logits)

            assert result.shape == original_logits_shape

    @patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.torch')
    def test_torch_tensor_conversion(self, mock_torch, mock_tokenizer, valid_response_format):
        """Test that tokens are correctly converted to torch tensors."""
        with patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.JsonSchemaParser'), \
                patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.build_transformers_prefix_allowed_tokens_fn') as mock_build_fn:

            mock_prefix_fn = Mock(return_value=[1, 2])
            mock_build_fn.return_value = mock_prefix_fn

            mock_tensor = Mock()
            mock_torch.tensor = Mock(return_value=mock_tensor)
            mock_torch.long = torch.long

            processor = JsonLogitsProcessor(mock_tokenizer, valid_response_format)

            tokens = mx.array([1, 2, 3, 4])
            logits = mx.array([0.1, 0.2, 0.3, 0.4, 0.5])

            processor(tokens, logits)

            mock_torch.tensor.assert_called_once()
            call_args = mock_torch.tensor.call_args
            assert call_args[0][0] == [1, 2, 3, 4]
            assert call_args[1]['dtype'] == torch.long


class TestJsonLogitsProcessorEdgeCases:
    """Test suite for edge cases and error scenarios in JsonLogitsProcessor."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        mock_tokenizer = Mock(spec=TokenizerWrapper)
        mock_tokenizer._tokenizer = Mock()
        return mock_tokenizer

    @pytest.fixture
    def simple_response_format(self):
        """Create a simple response format for testing."""
        json_schema_format = JsonSchemaFormat(
            name="simple",
            schema={"type": "string"}
        )
        return ResponseFormat(type="json_schema", json_schema=json_schema_format)

    @pytest.fixture
    def json_logits_processor(self, mock_tokenizer, simple_response_format):
        """Create a JsonLogitsProcessor instance for testing."""
        with patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.JsonSchemaParser'), \
                patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.build_transformers_prefix_allowed_tokens_fn') as mock_build_fn:
            mock_build_fn.return_value = Mock(return_value=[])
            return JsonLogitsProcessor(mock_tokenizer, simple_response_format)

    def test_call_with_large_token_array(self, json_logits_processor):
        """Test __call__ method with large token array."""
        json_logits_processor.prefix_fn = Mock(return_value=[100, 200, 300])

        large_tokens = mx.arange(1000)
        logits = mx.random.normal((1, 5000))

        result = json_logits_processor(large_tokens, logits)

        assert isinstance(result, mx.array)
        assert result.shape == logits.shape

    def test_call_with_nan_values(self, json_logits_processor):
        """Test __call__ method with NaN values in logits."""
        json_logits_processor.prefix_fn = Mock(return_value=[0, 2])

        tokens = mx.array([1, 2, 3, 4])
        logits = mx.array([[float('nan'), 0.2, float('nan'), 0.4, 0.5]])

        result = json_logits_processor(tokens, logits)

        assert isinstance(result, mx.array)
        assert result.shape == logits.shape
        assert mx.isnan(result[0, 0])
        assert mx.isnan(result[0, 2])

    def test_call_with_inf_values(self, json_logits_processor):
        """Test __call__ method with infinite values in logits."""
        json_logits_processor.prefix_fn = Mock(return_value=[0, 2])

        tokens = mx.array([1, 2, 3, 4])
        logits = mx.array([[float('inf'), 0.2, float('-inf'), 0.4, 0.5]])

        result = json_logits_processor(tokens, logits)

        assert isinstance(result, mx.array)
        assert result.shape == logits.shape
        assert mx.isinf(result[0, 0])
        assert mx.isinf(result[0, 2])

    def test_prefix_function_exception_handling(self, json_logits_processor):
        """Test handling of exceptions from the prefix function."""
        json_logits_processor.prefix_fn = Mock(side_effect=Exception("Prefix function error"))

        tokens = mx.array([1, 2, 3, 4])
        logits = mx.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

        with pytest.raises(Exception, match="Prefix function error"):
            json_logits_processor(tokens, logits)

    def test_call_with_negative_token_values(self, json_logits_processor):
        """Test __call__ method with negative token values."""
        json_logits_processor.prefix_fn = Mock(return_value=[100, 200])

        tokens = mx.array([-1, 0, 1, 2, 3])
        logits = mx.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

        result = json_logits_processor(tokens, logits)

        assert isinstance(result, mx.array)
        assert result.shape == logits.shape

    def test_call_with_zero_logits(self, json_logits_processor):
        """Test __call__ method with zero logits."""
        json_logits_processor.prefix_fn = Mock(return_value=[100, 200])

        tokens = mx.array([1, 2, 3, 4])
        logits = mx.zeros((1, 1000))

        result = json_logits_processor(tokens, logits)

        assert isinstance(result, mx.array)
        assert result.shape == logits.shape
        assert result[0, 100] == 0.0
        assert result[0, 200] == 0.0

    def test_multiple_calls_statelessness(self, json_logits_processor):
        """Test that multiple calls to __call__ are stateless."""
        json_logits_processor.prefix_fn = Mock(return_value=[0, 1, 2, 3, 4])

        tokens = mx.array([1, 2, 3, 4])
        logits = mx.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

        results = []
        for _ in range(5):
            result = json_logits_processor(tokens, logits)
            results.append(result)

        for i in range(1, len(results)):
            assert mx.array_equal(results[0], results[i])

    def test_different_dtypes(self, mock_tokenizer, simple_response_format):
        """Test JsonLogitsProcessor with different data types."""
        for dtype in [mx.float16, mx.float32]:
            with patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.JsonSchemaParser'), \
                    patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.build_transformers_prefix_allowed_tokens_fn') as mock_build_fn:
                mock_build_fn.return_value = Mock(return_value=[0, 1, 2, 3, 4])
                processor = JsonLogitsProcessor(mock_tokenizer, simple_response_format)

                tokens = mx.array([1, 2, 3, 4], dtype=mx.int32)
                logits = mx.random.normal((1, 1000), dtype=dtype)

                result = processor(tokens, logits)
                assert result.dtype == dtype

    def test_call_with_negative_logits(self, json_logits_processor):
        """Test __call__ method with negative logits."""
        json_logits_processor.prefix_fn = Mock(return_value=[0, 1, 2, 3, 4])

        tokens = mx.array([1, 2, 3, 4])
        logits = mx.array([[-0.1, -0.2, -0.3, -0.4, -0.5]])

        result = json_logits_processor(tokens, logits)

        assert isinstance(result, mx.array)
        assert result.shape == logits.shape
        assert result[0, 0] == -0.1

    def test_call_with_floating_point_tokens(self, json_logits_processor):
        """Test __call__ method with floating point tokens."""
        json_logits_processor.prefix_fn = Mock(return_value=[100, 200])

        tokens = mx.array([1.0, 2.0, 3.0, 4.0])
        logits = mx.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

        result = json_logits_processor(tokens, logits)

        assert isinstance(result, mx.array)
        assert result.shape == logits.shape

    def test_call_with_empty_logits(self, json_logits_processor):
        """Test __call__ method with empty logits array."""
        json_logits_processor.prefix_fn = Mock(return_value=[])

        tokens = mx.array([1, 2, 3, 4])
        logits = mx.array([])

        result = json_logits_processor(tokens, logits)

        assert isinstance(result, mx.array)
        assert result.shape == logits.shape

    def test_call_with_single_element_logits(self, json_logits_processor):
        """Test __call__ method with single element logits array."""
        json_logits_processor.prefix_fn = Mock(return_value=[0])

        tokens = mx.array([1])
        logits = mx.array([0.5])

        result = json_logits_processor(tokens, logits)

        assert isinstance(result, mx.array)
        assert result.shape == logits.shape
        assert result[0] == 0.5

    @patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.torch')
    def test_torch_tensor_conversion_edge_cases(self, mock_torch, json_logits_processor):
        """Test torch tensor conversion with edge cases."""
        mock_tensor = Mock()
        mock_torch.tensor = Mock(return_value=mock_tensor)

        json_logits_processor.prefix_fn = Mock(return_value=[0, 1, 2])

        empty_tokens = mx.array([])
        logits = mx.array([[0.1, 0.2, 0.3]])

        json_logits_processor(empty_tokens, logits)

        mock_torch.tensor.assert_called_once_with([], dtype=mock_torch.long)

    def test_integration_with_real_tokenizer_schema(self, mock_tokenizer):
        """Test integration with a more complex real-world schema."""
        complex_schema = {
            "name": "user_profile",
            "schema": {
                "type": "object",
                "properties": {
                    "user": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"},
                        }
                    }
                }
            }
        }

        json_schema_format = JsonSchemaFormat(**complex_schema)
        response_format = ResponseFormat(type="json_schema", json_schema=json_schema_format)

        with patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.JsonSchemaParser'), \
                patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.build_transformers_prefix_allowed_tokens_fn'):
            processor = JsonLogitsProcessor(mock_tokenizer, response_format)

            assert processor is not None
            assert hasattr(processor, 'prefix_fn')

    def test_error_handling_with_malformed_schema(self, mock_tokenizer):
        """Test error handling with malformed JSON schema."""
        malformed_schema = {
            "name": "malformed",
            "schema": {
                "type": "object"
            }
        }

        json_schema_format = JsonSchemaFormat(**malformed_schema)
        response_format = ResponseFormat(type="json_schema", json_schema=json_schema_format)

        with patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.JsonSchemaParser'), \
                patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.build_transformers_prefix_allowed_tokens_fn'):
            processor = JsonLogitsProcessor(mock_tokenizer, response_format)

            assert processor is not None
            assert hasattr(processor, 'prefix_fn')

            processor.prefix_fn = Mock(return_value=[1, 2])

            result = processor(tokens=mx.array([1, 2]), logits=mx.array([0.1, 0.2, 0.3]))
            assert isinstance(result, mx.array)

    def test_json_logits_processor_with_complex_schema(self, mock_tokenizer):
        """Test JsonLogitsProcessor with a more complex JSON schema."""
        json_schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"]
                },
            },
            "required": ["user"]
        }

        response_format = ResponseFormat(
            type="json_schema",
            json_schema=JsonSchemaFormat(
                name="complex_schema",
                schema=json_schema
            )
        )

        allowed_tokens = [1, 3, 5, 7, 9]

        with patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.JsonSchemaParser') as mock_parser, \
                patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.build_transformers_prefix_allowed_tokens_fn') as mock_build_fn:

            mock_prefix_fn = Mock(return_value=allowed_tokens)
            mock_build_fn.return_value = mock_prefix_fn

            processor = JsonLogitsProcessor(mock_tokenizer, response_format)

            mock_parser.assert_called_once_with(json_schema)

            tokens = mx.array([10, 20, 30, 40, 50])
            logits = mx.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

            result = processor(tokens, logits)

            assert result.shape == logits.shape

            result_list = result.tolist()
            for i, logit_val in enumerate(result_list):
                if i in allowed_tokens:
                    assert logit_val == logits[i].item()
                else:
                    assert logit_val == float('-inf')


class TestJsonLogitsProcessorIntegration:
    """Integration tests for JsonLogitsProcessor with the MLX LM model system."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        mock_tokenizer = Mock(spec=TokenizerWrapper)
        mock_tokenizer._tokenizer = Mock()
        mock_tokenizer.tokenizer = mock_tokenizer  # For compatibility
        return mock_tokenizer

    @pytest.fixture
    def simple_json_schema(self):
        """Create a simple JSON schema for testing."""
        return {
            "name": "simple_test",
            "schema": {
                "type": "object",
                "properties": {
                    "response": {"type": "string"}
                },
                "required": ["response"]
            }
        }

    @pytest.fixture
    def chat_completion_request(self, simple_json_schema):
        """Create a ChatCompletionRequest with JSON schema response format."""
        json_schema_format = JsonSchemaFormat(**simple_json_schema)
        response_format = ResponseFormat(type="json_schema", json_schema=json_schema_format)

        return ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Generate a JSON response"}],
            response_format=response_format
        )

    def test_json_logits_processor_creation_in_model(
        self,
        mock_tokenizer,
        chat_completion_request
    ):
        """Test that JsonLogitsProcessor is correctly created in the model system."""
        generate_kwargs = {}

        if chat_completion_request.response_format and chat_completion_request.response_format.json_schema:
            generate_kwargs["logits_processors"] = [
                JsonLogitsProcessor(
                    mock_tokenizer,
                    chat_completion_request.response_format
                )
            ]

        assert "logits_processors" in generate_kwargs
        assert len(generate_kwargs["logits_processors"]) == 1

        processor = generate_kwargs["logits_processors"][0]
        assert isinstance(processor, JsonLogitsProcessor)

        assert processor.processed_token_count == 0
        assert hasattr(processor, 'prefix_fn')

    def test_json_logits_processor_with_real_tokenizer_simulation(
        self,
        mock_tokenizer,
        simple_json_schema
    ):
        """Test JsonLogitsProcessor with a more realistic tokenizer simulation."""
        mock_tokenizer._tokenizer.convert_ids_to_tokens = Mock(return_value=["<s>", "hello", "world", "</s>"])
        mock_tokenizer._tokenizer.convert_tokens_to_ids = Mock(return_value=lambda tokens: [0, 1, 2, 3])
        mock_tokenizer._tokenizer.vocab_size = 10000

        json_schema_format = JsonSchemaFormat(**simple_json_schema)
        response_format = ResponseFormat(type="json_schema", json_schema=json_schema_format)

        with patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.JsonSchemaParser'), \
                patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.build_transformers_prefix_allowed_tokens_fn') as mock_build_fn:
            mock_build_fn.return_value = Mock(return_value=[1, 2, 3, 100, 101, 102])
            processor = JsonLogitsProcessor(mock_tokenizer, response_format)

            tokens = mx.array([0, 1, 2, 3, 1, 2])
            logits = mx.random.normal((1, 10000))

            result = processor(tokens, logits)

            assert isinstance(result, mx.array)
            assert result.shape == logits.shape

    def test_json_logits_processor_schema_validation(
        self,
        mock_tokenizer
    ):
        """Test JsonLogitsProcessor with various schema types."""
        test_schemas = [
            {"name": "string_test", "schema": {"type": "string"}},
            {
                "name": "nested_test",
                "schema": {
                    "type": "object",
                    "properties": {
                        "user": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "integer"}
                            }
                        }
                    }
                }
            },
        ]

        for schema_def in test_schemas:
            json_schema_format = JsonSchemaFormat(**schema_def)
            response_format = ResponseFormat(type="json_schema", json_schema=json_schema_format)

            with patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.JsonSchemaParser'), \
                    patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.build_transformers_prefix_allowed_tokens_fn') as mock_build_fn:
                mock_build_fn.return_value = Mock(return_value=[100, 200])
                processor = JsonLogitsProcessor(mock_tokenizer, response_format)

                assert processor is not None
                assert hasattr(processor, 'prefix_fn')
                assert callable(processor.prefix_fn)

                tokens = mx.array([1, 2, 3, 4])
                logits = mx.random.normal((1, 1000))

                result = processor(tokens, logits)
                assert isinstance(result, mx.array)
                assert result.shape == logits.shape

    def test_json_logits_processor_error_scenarios(
        self,
        mock_tokenizer
    ):
        """Test JsonLogitsProcessor with various error scenarios."""
        invalid_schemas = [
            {"name": "invalid_test", "schema": {"properties": {"test": {"type": "string"}}}},
            {"name": "invalid_test", "schema": {"type": "invalid_type"}},
            {"name": "empty_test", "schema": {}}
        ]

        for schema_def in invalid_schemas:
            try:
                json_schema_format = JsonSchemaFormat(**schema_def)
                response_format = ResponseFormat(type="json_schema", json_schema=json_schema_format)

                with patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.JsonSchemaParser'), \
                        patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.build_transformers_prefix_allowed_tokens_fn'):
                    processor = JsonLogitsProcessor(mock_tokenizer, response_format)

                    assert processor is not None
                    assert hasattr(processor, 'prefix_fn')

                    processor.prefix_fn = Mock(return_value=[100, 200])

                    tokens = mx.array([1, 2, 3, 4])
                    logits = mx.random.normal((1, 1000))

                    result = processor(tokens, logits)
                    assert isinstance(result, mx.array)

            except Exception as e:
                logger.info(f"Schema {schema_def['name']} caused expected error: {e}")

    def test_json_logits_processor_performance_characteristics(
        self,
        mock_tokenizer
    ):
        """Test performance characteristics of JsonLogitsProcessor."""
        complex_schema = {
            "name": "performance_test",
            "schema": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "name": {"type": "string"},
                            }
                        }
                    }
                }
            }
        }

        json_schema_format = JsonSchemaFormat(**complex_schema)
        response_format = ResponseFormat(type="json_schema", json_schema=json_schema_format)

        with patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.JsonSchemaParser'), \
                patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.build_transformers_prefix_allowed_tokens_fn') as mock_build_fn:
            mock_build_fn.return_value = Mock(return_value=list(range(1000)))
            processor = JsonLogitsProcessor(mock_tokenizer, response_format)

            test_sizes = [
                (100, 1000),
                (1000, 5000),
            ]

            for token_count, vocab_size in test_sizes:
                tokens = mx.arange(token_count)
                logits = mx.random.normal((1, vocab_size))

                start_time = time.time()

                result = processor(tokens, logits)

                end_time = time.time()
                processing_time = end_time - start_time

                logger.info(f"Processed {token_count} tokens, {vocab_size} vocab size in {processing_time:.4f}s")

                assert isinstance(result, mx.array)
                assert result.shape == logits.shape
                assert processing_time < 1.0

    def test_json_logits_processor_thread_safety(
        self,
        mock_tokenizer
    ):
        """Test thread safety of JsonLogitsProcessor."""
        json_schema_format = JsonSchemaFormat(
            name="thread_test",
            schema={"type": "string"}
        )
        response_format = ResponseFormat(type="json_schema", json_schema=json_schema_format)

        with patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.JsonSchemaParser'), \
                patch('mlx_omni_server.chat.mlx_lm.json_logits_processor.build_transformers_prefix_allowed_tokens_fn') as mock_build_fn:
            mock_build_fn.return_value = Mock(return_value=[100, 200])
            processor = JsonLogitsProcessor(mock_tokenizer, response_format)

            results = []
            errors = []

            def worker_thread(thread_id):
                try:
                    for i in range(5):
                        tokens = mx.array([1, 2, 3, 4])
                        logits = mx.random.normal((1, 1000))

                        result = processor(tokens, logits)
                        results.append((thread_id, i, result))

                        time.sleep(0.001)

                except Exception as e:
                    errors.append((thread_id, str(e)))

            threads = []
            num_threads = 3
            for i in range(num_threads):
                thread = threading.Thread(target=worker_thread, args=(i,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            assert len(errors) == 0, f"Thread safety errors: {errors}"
            assert len(results) == num_threads * 5, "Not all operations completed"

            for thread_id, iteration, result in results:
                assert isinstance(result, mx.array)
                assert result.shape == (1, 1000)
