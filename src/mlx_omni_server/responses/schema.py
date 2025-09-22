from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel

from ..chat.schema import MultimodalContentItem, Role


class ResponseStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


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
    status: ResponseStatus = ResponseStatus.COMPLETED
    content: list[ResponseOutputContentItem]
    created_at: Optional[int] = None


class ResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ResponseResponse(BaseModel):
    id: str
    object: Literal["response"] = "response"
    created: int
    model: str
    output: list[ResponseOutputMessage]
    status: ResponseStatus = ResponseStatus.COMPLETED
    usage: ResponseUsage
    metadata: Optional[dict[str, Any]] = None


class ResponseRequest(BaseModel):
    model: str
    input: Any
    instructions: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    modalities: Optional[list[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop: Optional[Union[str, list[str]]] = None
    stream: Optional[bool] = False
    response_format: Optional[dict[str, Any]] = None
    extra_headers: Optional[dict[str, Any]] = None

    class Config:
        extra = "allow"


class ResponseStreamEvent(BaseModel):
    event: str
    data: dict[str, Any]
