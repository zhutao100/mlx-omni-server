from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional

from ...schemas.chat_schema import ChatCompletionRequest
from ...schemas.tools_schema import ToolCall


@dataclass
class GenerateResult:
    """Result from generate step"""

    text: str
    token: int
    finished: bool
    logprobs: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[ToolCall]] = None


class BaseMLXModel(ABC):
    """Base class for chat models"""

    @abstractmethod
    async def generate(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[GenerateResult, None]:
        """Generate completion text with parameters from request"""
        pass

    @abstractmethod
    async def token_count(self, prompt: str) -> int:
        """Count the number of tokens in the text"""
        pass
