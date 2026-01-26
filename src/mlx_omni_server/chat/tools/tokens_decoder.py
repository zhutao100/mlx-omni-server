import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict

from mlx_lm.tokenizer_utils import TokenizerWrapper


class TokensDecoder(ABC):
    """Base class for tokens decoders."""

    @abstractmethod
    def decode(self, text: str) -> Dict[str, Any] | None:
        """Parse tool calls from model output."""
        pass

    def stream_decode(self, text: str) -> Dict[str, Any] | None:
        """Parse tool calls from model output."""
        return self.decode(text)


class ReasoningDecoder(TokensDecoder):
    """Decoder that extracts reasoning steps enclosed in specific tags from model output."""
    def __init__(self, thinking_tag) -> None:
        self.thinking_tag = thinking_tag
        self.thinking_start_tag = f"<{self.thinking_tag}>"
        self.thinking_end_tag = f"</{self.thinking_tag}>"
        self.accumulated_text = ""
        self.enable_thinking: bool = False
        self.add_thinking_prefix: bool = False
        self._stream_buffer = ""
        self._in_thinking = False

    def set_thinking_prefix(self, add_thinking_prefix: bool) -> None:
        self.add_thinking_prefix = add_thinking_prefix
        self.accumulated_text = ""
        self._stream_buffer = ""
        self._in_thinking = add_thinking_prefix
        if add_thinking_prefix:
            self.accumulated_text = self.thinking_start_tag

    def _split_partial_suffix(self, text: str, tag: str) -> tuple[str, str]:
        if not text:
            return "", ""

        max_prefix_len = min(len(text), len(tag) - 1)
        for suffix_len in range(max_prefix_len, 0, -1):
            suffix = text[-suffix_len:]
            if tag.startswith(suffix):
                return text[:-suffix_len], suffix
        return text, ""

    def _parse_stream_response(self, text: str) -> dict[str, Any]:
        if not text:
            return {"delta_content": None, "delta_reasoning": None}

        self.accumulated_text += text

        chunk = f"{self._stream_buffer}{text}"
        self._stream_buffer = ""

        saw_start_tag = False
        saw_end_tag = False
        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        remaining = chunk
        while remaining:
            if self._in_thinking:
                end_pos = remaining.find(self.thinking_end_tag)
                if end_pos < 0:
                    emit, buffer = self._split_partial_suffix(remaining, self.thinking_end_tag)
                    if emit:
                        reasoning_parts.append(emit)
                    self._stream_buffer = buffer
                    remaining = ""
                    continue

                before = remaining[:end_pos]
                if before:
                    reasoning_parts.append(before)
                remaining = remaining[end_pos + len(self.thinking_end_tag) :]
                self._in_thinking = False
                saw_end_tag = True
                continue

            start_pos = remaining.find(self.thinking_start_tag)
            if start_pos < 0:
                emit, buffer = self._split_partial_suffix(remaining, self.thinking_start_tag)
                if emit:
                    content_parts.append(emit)
                self._stream_buffer = buffer
                remaining = ""
                continue

            before = remaining[:start_pos]
            if before:
                content_parts.append(before)
            remaining = remaining[start_pos + len(self.thinking_start_tag) :]
            self._in_thinking = True
            saw_start_tag = True

        delta_reasoning = "".join(reasoning_parts) if reasoning_parts else None
        delta_content = "".join(content_parts) if content_parts else None

        if saw_start_tag and delta_reasoning is None and delta_content is None:
            delta_reasoning = ""

        if saw_end_tag and delta_reasoning is None and delta_content is None:
            delta_content = ""

        return {"delta_content": delta_content, "delta_reasoning": delta_reasoning}

    def stream_decode(self, text: str) -> Dict[str, Any] | None:
        """Parse tool calls from model output."""
        parsed = self._parse_stream_response(text)
        if self.enable_thinking:
            return parsed
        return {"delta_content": parsed.get("delta_content"), "delta_reasoning": None}

    def _parse_response(self, response: str):
        # First check for complete thinking tag pattern
        reasoning_regex = fr"{self.thinking_start_tag}([\s\S]*?){self.thinking_end_tag}"
        reasoning_match = re.search(reasoning_regex, response)

        if reasoning_match:
            # Extract thinking content
            reasoning_content = reasoning_match.group(1).strip()

            # Get final content by replacing thinking tag and its content
            content = re.sub(reasoning_regex, "", response, count=1).strip()

            return {
                "content": content,
                "reasoning": reasoning_content,
            }
        else:
            # Check if only end tag exists (missing start tag case)
            if self.thinking_end_tag in response:
                # Split response using end tag
                parts = response.split(self.thinking_end_tag, 1)
                if len(parts) == 2:
                    reasoning_content = parts[0].strip()
                    content = parts[1].strip()
                    return {
                        "content": content,
                        "reasoning": reasoning_content,
                    }

            # If no tags exist, the entire response is the content
            return {
                "content": response.strip(),
                "reasoning": None,
            }

    def decode(self, text: str) -> Dict[str, Any] | None:
        """Parse thinking content from model output"""
        parsed = self._parse_response(text)
        if self.enable_thinking:
            return parsed
        return {"content": parsed.get("content")}
