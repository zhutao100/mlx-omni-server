import json
import time
import uuid
from collections.abc import Callable
from typing import Optional

from fastapi import Request, Response
from rich.markup import escape
from starlette.background import BackgroundTasks
from starlette.middleware.base import BaseHTTPMiddleware

from ..utils.logger import logger

MAX_LOG_BODY_BYTES = 16 * 1024


def format_body(body: str) -> str:
    """Format body content for logging."""
    try:
        # Try to parse as JSON and format it
        parsed = json.loads(body)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        # If not JSON, return as is
        return body


class RequestResponseLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        *,
        exclude_paths: Optional[list[str]] = None,
    ):
        """Initialize the middleware with custom log levels.

        Args:
            app: The ASGI application
            request_level: Logging level for requests (default: INFO)
            response_level: Logging level for responses (default: INFO)
            exclude_paths: List of paths to exclude from logging (default: None)
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or []

    def should_log(self, path: str) -> bool:
        """Check if the path should be logged."""
        return not any(path.startswith(exclude) for exclude in self.exclude_paths)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.should_log(request.url.path):
            return await call_next(request)

        request_id = uuid.uuid4().hex
        request_body, request_body_truncated = await self._get_request_body_for_logging(request)

        logger.info(
            f"Request [{request_id}]: {request.method} {request.url}\n"
            f"Headers:\n{json.dumps(dict(request.headers), indent=2)}\n"
            f"Body:\n{escape(self._format_body_for_logging(request_body, request_body_truncated))}",
        )

        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time

        media_type = self._response_media_type(response)
        if self._should_skip_response_body_logging(response, media_type):
            logger.info(
                f"Response [{request_id}] took {process_time:.2f}s:\n"
                f"Status: {response.status_code}\n"
                f"Headers:\n{json.dumps(dict(response.headers), indent=2)}\n"
            )
            return response

        captured = bytearray()
        total_bytes = 0
        truncated = False
        original_iterator = response.body_iterator

        async def wrapped_body_iterator():
            nonlocal total_bytes, truncated
            async for chunk in original_iterator:
                total_bytes += len(chunk)
                if len(captured) < MAX_LOG_BODY_BYTES:
                    remaining = MAX_LOG_BODY_BYTES - len(captured)
                    captured.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        truncated = True
                else:
                    truncated = True
                yield chunk

        response.body_iterator = wrapped_body_iterator()

        def log_response_body() -> None:
            body_text = self._decode_body_preview_for_logging(
                bytes(captured),
                media_type,
            )
            if truncated:
                body_text = f"{body_text}\n\n<Truncated after {MAX_LOG_BODY_BYTES} bytes (total={total_bytes} bytes)>"
            logger.info(
                f"Response [{request_id}] took {process_time:.2f}s:\n"
                f"Status: {response.status_code}\n"
                f"Headers:\n{json.dumps(dict(response.headers), indent=2)}\n"
                f"Body:\n{escape(self._format_body_for_logging(body_text, truncated))}",
            )

        if response.background is None:
            response.background = BackgroundTasks()
        if isinstance(response.background, BackgroundTasks):
            response.background.add_task(log_response_body)
        else:
            existing = response.background
            tasks = BackgroundTasks()
            tasks.add_task(existing)
            tasks.add_task(log_response_body)
            response.background = tasks

        return response

    def _request_media_type(self, request: Request) -> str:
        content_type = request.headers.get("content-type", "")
        return content_type.split(";", 1)[0].strip().lower()

    def _response_media_type(self, response: Response) -> str:
        content_type = response.headers.get("content-type", "")
        return content_type.split(";", 1)[0].strip().lower()

    def _is_textual_media_type(self, media_type: str) -> bool:
        if not media_type:
            return True
        if media_type == "text/event-stream":
            return False
        if media_type.startswith("text/"):
            return True
        return media_type in {
            "application/json",
            "application/problem+json",
            "application/xml",
        }

    def _is_binary_media_type(self, media_type: str) -> bool:
        if not media_type:
            return False
        if media_type.startswith(("audio/", "image/", "video/")):
            return True
        return media_type in {
            "application/octet-stream",
            "application/pdf",
            "application/zip",
        }

    def _should_skip_response_body_logging(self, response: Response, media_type: str) -> bool:
        if media_type == "text/event-stream":
            return True
        if self._is_binary_media_type(media_type):
            return True
        content_disposition = response.headers.get("content-disposition", "").lower()
        if content_disposition.startswith("attachment"):
            return True
        return False

    def _format_body_for_logging(self, body: str, truncated: bool) -> str:
        if not body:
            return ""
        if truncated:
            return body
        formatted = format_body(body)
        if len(formatted) <= MAX_LOG_BODY_BYTES:
            return formatted
        return formatted[:MAX_LOG_BODY_BYTES] + "\n\n<Truncated>"

    def _decode_body_preview_for_logging(self, body: bytes, media_type: str) -> str:
        if not body:
            return ""
        if self._is_binary_media_type(media_type) or not self._is_textual_media_type(media_type):
            return "<Binary Content>"
        return body.decode(errors="replace")

    async def _get_request_body_for_logging(self, request: Request) -> tuple[str, bool]:
        media_type = self._request_media_type(request)

        content_length = request.headers.get("content-length")
        try:
            length = int(content_length) if content_length is not None else None
        except ValueError:
            length = None

        if not self._is_textual_media_type(media_type):
            return f"<Skipped: content-type={media_type or 'unknown'}>", False
        if length is not None and length > MAX_LOG_BODY_BYTES and media_type != "application/json":
            return (
                f"<Skipped: content-type={media_type or 'unknown'}, content-length={length}>",
                False,
            )

        try:
            raw = await request.body()
        except Exception:
            return "", False

        truncated = len(raw) > MAX_LOG_BODY_BYTES
        preview = raw[:MAX_LOG_BODY_BYTES]
        text = preview.decode(errors="replace")
        if truncated:
            text = f"{text}\n\n<Truncated after {MAX_LOG_BODY_BYTES} bytes (content-length={len(raw)} bytes)>"
        return text, truncated
