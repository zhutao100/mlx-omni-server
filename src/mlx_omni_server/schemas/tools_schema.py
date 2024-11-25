from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ToolType(str, Enum):
    FUNCTION = "function"


class FunctionParameters(BaseModel):
    type: str = "object"
    properties: Dict
    required: Optional[List[str]] = None


class Function(BaseModel):
    name: str = Field(..., max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    description: Optional[str] = None
    parameters: Optional[FunctionParameters] = None


class Tool(BaseModel):
    type: ToolType = ToolType.FUNCTION
    function: Function


class ToolChoice(str, Enum):
    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


class SpecificToolChoice(BaseModel):
    type: ToolType = ToolType.FUNCTION
    function: Dict[str, str]


ToolChoiceType = Union[ToolChoice, SpecificToolChoice]


class ToolCall(BaseModel):
    id: str
    type: ToolType = ToolType.FUNCTION
    function: Dict[str, str]
