from abc import ABC, abstractmethod
from typing import AsyncGenerator, Tuple

from ...schemas.chat_schema import ChatCompletionRequest


class BaseMLXModel(ABC):
    """Base class for chat models"""

    @abstractmethod
    async def generate(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[Tuple[str, bool], None]:
        """Generate completion text with parameters from request"""
        pass

    @abstractmethod
    async def token_count(self, prompt: str) -> int:
        """Generate completion text with parameters from request"""
        pass
