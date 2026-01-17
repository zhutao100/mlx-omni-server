from unittest.mock import Mock

from mlx_lm.tokenizer_utils import TokenizerWrapper

from mlx_omni_server.chat.schema import ChatCompletionRequest, Role, Tool, ToolType
from mlx_omni_server.chat.tools.hugging_face import HuggingFaceChatTokenizer


def test_chat_completion_request_coerces_developer_to_system() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [
                {"role": "developer", "content": "Follow these instructions."},
                {"role": "user", "content": "Hello"},
            ],
        }
    )

    assert request.messages[0].role == Role.SYSTEM
    assert request.messages[1].role == Role.USER


def test_chat_completion_request_coerces_custom_tool_to_function_tool_shim() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [
                {
                    "type": "custom",
                    "custom": {
                        "name": "apply_patch",
                        "description": "Apply a patch.",
                        "format": {"type": "grammar", "syntax": "lark"},
                    },
                }
            ],
        }
    )

    assert request.tools is not None
    assert len(request.tools) == 1

    tool = request.tools[0]
    assert isinstance(tool, Tool)
    assert tool.type == ToolType.FUNCTION
    assert tool.function.name == "apply_patch"
    assert tool.function.description == "Apply a patch."
    assert tool.function.parameters is not None
    assert tool.function.parameters.type == "object"
    assert tool.function.parameters.properties is not None
    assert tool.function.parameters.properties["input"]["type"] == "string"
    assert tool.function.parameters.required == ["input"]

    dumped = tool.model_dump(exclude_none=True)
    assert dumped["format"]["syntax"] == "lark"


def test_custom_tool_shim_is_passed_to_chat_template() -> None:
    mock_tokenizer = Mock(spec=TokenizerWrapper)
    mock_tokenizer.apply_chat_template = Mock(return_value="PROMPT")
    tokenizer = HuggingFaceChatTokenizer(mock_tokenizer)

    request = ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [{"type": "custom", "custom": {"name": "apply_patch"}}],
        }
    )

    prompt = tokenizer.encode(request.messages, tools=request.tools)
    assert prompt == "PROMPT"

    call = mock_tokenizer.apply_chat_template.call_args
    assert call is not None
    assert call.kwargs["tools"][0]["type"] == "function"
    assert call.kwargs["tools"][0]["function"]["name"] == "apply_patch"
    assert call.kwargs["tools"][0]["function"]["parameters"]["required"] == ["input"]
