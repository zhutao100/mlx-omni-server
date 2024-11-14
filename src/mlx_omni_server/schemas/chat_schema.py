from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: Role
    content: str
    name: Optional[str] = None

    class Config:
        json_encoders = {
            # 自定义编码器
            bytes: lambda v: v.decode()
        }


class ChatCompletionUsageDetails(BaseModel):
    reasoning_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: Optional[ChatCompletionUsageDetails] = None


class ChatCompletionMessage(BaseModel):
    role: Role
    content: str
    name: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str
    logprobs: Optional[Any] = None


class ChatCompletion(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage
    system_fingerprint: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]
    system_fingerprint: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    messages: List[Message]
    temperature: Optional[float] = Field(1.0, ge=0, le=2)
    top_p: Optional[float] = Field(1.0, ge=0, le=1)
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    n: Optional[int] = Field(1, ge=1, le=10)

    @field_validator("temperature")
    def validate_temperature(cls, v):
        if v is not None and (v < 0 or v > 2):
            raise ValueError("Temperature must be between 0 and 2")
        return v

    @field_validator("top_p")
    def validate_top_p(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Top_p must be between 0 and 1")
        return v
