from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, field_validator


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: Role
    content: str
    name: Optional[str] = None

    class Config:
        json_encoders = {bytes: lambda v: v.decode()}


class TokenLogprobs(BaseModel):
    token: str
    logprob: float
    bytes: Optional[List[int]]
    top_logprobs: Optional[List[Dict[str, Any]]]


class LogprobsResult(BaseModel):
    content: List[TokenLogprobs]


class ChatCompletionUsageDetails(BaseModel):
    reasoning_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: Optional[ChatCompletionUsageDetails] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str
    logprobs: Optional[LogprobsResult] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]
    system_fingerprint: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage
    system_fingerprint: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    # Standard OpenAI API fields
    model: str = Field(..., description="ID of the model to use")
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(1.0, ge=0, le=2)
    top_p: Optional[float] = Field(1.0, ge=0, le=1)
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = Field(
        None,
        ge=0,
        le=20,
    )
    n: Optional[int] = Field(1, ge=1, le=10)

    # Allow any additional fields
    class Config:
        extra = "allow"  # This allows additional fields not defined in the model

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

    def get_extra_params(self) -> Dict[str, Any]:
        """Get all extra parameters that aren't part of the standard OpenAI API."""
        standard_fields: Set[str] = {
            "model",
            "messages",
            "temperature",
            "top_p",
            "max_tokens",
            "stream",
            "seed",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "n",
        }
        return {k: v for k, v in self.model_dump().items() if k not in standard_fields}
