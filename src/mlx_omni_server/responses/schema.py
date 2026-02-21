from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field

from ..chat.schema import MultimodalContentItem, Role


class ResponseStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    QUEUED = "queued"
    INCOMPLETE = "incomplete"


class ResponseOutputItemStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"


class ResponseInputMessage(BaseModel):
    role: Role
    content: Union[str, list[MultimodalContentItem]]


class ResponseToolCallDelta(BaseModel):
    id: Optional[str] = None
    type: str = "function"
    name: Optional[str] = None
    arguments: Optional[str] = None


class ResponseOutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str


class ResponseOutputToolCall(BaseModel):
    type: Literal["output_tool_call"] = "output_tool_call"
    id: str
    function: dict[str, Any]


class ResponseOutputContentItem(BaseModel):
    """Represents a single output content block."""

    type: str
    text: Optional[str] = None
    annotations: Optional[list[Any]] = None
    id: Optional[str] = None
    function: Optional[dict[str, Any]] = None


class ResponseOutputMessage(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Role = Role.ASSISTANT
    status: ResponseOutputItemStatus = ResponseOutputItemStatus.COMPLETED
    content: list[ResponseOutputContentItem]
    created_at: Optional[int] = None


class ResponseOutputFunctionCall(BaseModel):
    id: str
    type: Literal["function_call"] = "function_call"
    status: ResponseOutputItemStatus = ResponseOutputItemStatus.COMPLETED
    name: str
    arguments: str
    call_id: str


class ResponseOutputReasoning(BaseModel):
    id: str
    type: Literal["reasoning"] = "reasoning"
    status: ResponseOutputItemStatus = ResponseOutputItemStatus.COMPLETED
    content: Optional[list[dict[str, Any]]] = None
    summary: Optional[list[dict[str, Any]]] = None
    encrypted_content: Optional[str] = None


ResponseOutputItem = Annotated[
    Union[ResponseOutputMessage, ResponseOutputFunctionCall, ResponseOutputReasoning],
    Field(discriminator="type"),
]


class ResponseUsageInputTokensDetails(BaseModel):
    cached_tokens: int = 0


class ResponseUsageOutputTokensDetails(BaseModel):
    reasoning_tokens: int = 0


class ResponseUsage(BaseModel):
    input_tokens: int
    input_tokens_details: ResponseUsageInputTokensDetails
    output_tokens: int
    output_tokens_details: ResponseUsageOutputTokensDetails
    total_tokens: int


class ResponseResponse(BaseModel):
    id: str
    object: Literal["response"] = "response"
    created_at: int
    error: Optional[dict[str, Any]] = None
    incomplete_details: Optional[dict[str, Any]] = None
    model: str
    output: list[ResponseOutputItem]
    status: ResponseStatus = ResponseStatus.COMPLETED
    usage: Optional[ResponseUsage] = None
    metadata: Optional[dict[str, Any]] = None
    parallel_tool_calls: bool = True
    tool_choice: Any = "auto"
    tools: list[Any] = Field(default_factory=list)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    truncation: Optional[str] = None
    store: Optional[bool] = None
    max_output_tokens: Optional[int] = None
    instructions: Optional[Any] = None
    reasoning: Optional[Any] = None
    service_tier: Optional[str] = None
    previous_response_id: Optional[str] = None
    prompt: Optional[Any] = None
    text: Optional[Any] = None
    background: Optional[bool] = None
    max_tool_calls: Optional[int] = None
    prompt_cache_key: Optional[str] = None
    safety_identifier: Optional[str] = None


class ResponseTextConfig(BaseModel):
    format: Optional[dict[str, Any]] = None
    verbosity: Optional[str] = None

    class Config:
        extra = "allow"


class ResponseRequest(BaseModel):
    model: str
    input: Any
    include: Optional[list[str]] = None
    instructions: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    modalities: Optional[list[str]] = None
    previous_response_id: Optional[str] = None
    background: Optional[bool] = None
    store: Optional[bool] = None
    parallel_tool_calls: Optional[bool] = None
    tools: Optional[list[Any]] = None
    tool_choice: Optional[Any] = None
    truncation: Optional[str] = None
    text: Optional[ResponseTextConfig] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    prompt_cache_key: Optional[str] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[dict[str, float]] = None
    repetition_penalty: Optional[float] = Field(None, gt=0.0)
    repetition_context_size: Optional[int] = Field(None, ge=1)
    stop: Optional[Union[str, list[str]]] = None
    stream: Optional[bool] = False
    response_format: Optional[dict[str, Any]] = None
    extra_headers: Optional[dict[str, Any]] = None

    class Config:
        extra = "allow"


class ResponseStreamEvent(BaseModel):
    event: str
    data: dict[str, Any]
