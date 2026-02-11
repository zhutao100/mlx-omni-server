from typing import Any

from pydantic import BaseModel, Field

from ..schema_utils import extract_extra_params


class EmbeddingRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    input: str | list[str] = Field(..., description="Input text to get embeddings for")
    encoding_format: str | None = Field("float", description="The format of the embeddings")
    user: str | None = None
    dimensions: int | None = None

    # Allow any additional fields
    class Config:
        extra = "allow"  # This allows additional fields not defined in the model

    def get_extra_params(self) -> dict[str, Any]:
        """Get all extra parameters that aren't part of the standard OpenAI API."""
        standard_fields = frozenset({"model", "input", "encoding_format", "user", "dimensions"})
        return extract_extra_params(self, standard_fields)


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage
