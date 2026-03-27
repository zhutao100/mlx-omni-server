import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import Callable
from typing import Optional

from fastapi import Request, Response
from rich.markup import escape
from starlette.background import BackgroundTasks
from starlette.middleware.base import BaseHTTPMiddleware

from ..utils import log_artifacts
from ..utils.logger import logger
from ..utils.request_context import reset_request_id, set_request_id

MAX_LOG_BODY_BYTES = 16 * 1024
MAX_LOG_BODY_HEAD_BYTES = MAX_LOG_BODY_BYTES // 2
MAX_LOG_BODY_TAIL_BYTES = MAX_LOG_BODY_BYTES - MAX_LOG_BODY_HEAD_BYTES


def _format_head_tail_preview(
    *,
    head_text: str,
    tail_text: str,
    head_bytes: int,
    tail_bytes: int,
    total_bytes: int,
) -> str:
    preview_bytes = head_bytes + tail_bytes
    return (
        f"{head_text}\n\n<...snip...>\n\n{tail_text}\n\n"
        f"<Truncated preview={preview_bytes} bytes (head={head_bytes} tail={tail_bytes}) total={total_bytes} bytes>"
    )


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
        request_id_token = set_request_id(request_id)
        try:
            (
                request_body,
                request_body_truncated,
                request_body_sha256,
                request_body_artifact,
            ) = await self._get_request_body_for_logging(request, request_id=request_id)

            request_body_log = self._format_body_for_logging(request_body, request_body_truncated)
            if request_body_sha256:
                request_body_log += f"\n\n<Body sha256={request_body_sha256}>"
            if request_body_artifact:
                request_body_log += f"\n<Body artifact={request_body_artifact}>"

            logger.info(
                f"Request [{request_id}]: {request.method} {request.url}\n"
                f"Headers:\n{json.dumps(dict(request.headers), indent=2)}\n"
                f"Body:\n{escape(request_body_log)}",
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
            tail = bytearray()
            total_bytes = 0
            truncated = False
            response_body_sha256: str | None = None
            response_body_artifact: str | None = None
            response_body_hasher = None
            response_artifact_handle = None

            if log_artifacts.artifacts_enabled_http():
                response_body_hasher = hashlib.sha256()
                gzip_enabled = log_artifacts.artifacts_gzip_enabled()
                extension = "json" if media_type == "application/json" else "txt"
                artifact_path = log_artifacts.http_response_artifact_path(
                    request_id=request_id,
                    extension=extension,
                )
                effective_path = (
                    artifact_path.with_name(artifact_path.name + ".gz")
                    if gzip_enabled and not artifact_path.name.endswith(".gz")
                    else artifact_path
                )
                try:
                    response_artifact_handle = log_artifacts.open_artifact_writer(
                        artifact_path,
                        gzip_enabled=gzip_enabled,
                    )
                    response_body_artifact = log_artifacts.redact_home_path(effective_path)
                except Exception:
                    response_artifact_handle = None
                    response_body_artifact = None

            original_iterator = response.body_iterator

            async def wrapped_body_iterator():
                nonlocal response_body_sha256, total_bytes, truncated
                try:
                    async for chunk in original_iterator:
                        total_bytes += len(chunk)
                        if response_artifact_handle is not None:
                            response_artifact_handle.write(chunk)
                        if response_body_hasher is not None:
                            response_body_hasher.update(chunk)
                        if len(captured) < MAX_LOG_BODY_BYTES:
                            remaining = MAX_LOG_BODY_BYTES - len(captured)
                            captured.extend(chunk[:remaining])
                            if len(chunk) > remaining:
                                truncated = True
                        else:
                            truncated = True
                        if MAX_LOG_BODY_TAIL_BYTES > 0:
                            tail.extend(chunk)
                            if len(tail) > MAX_LOG_BODY_TAIL_BYTES:
                                del tail[:-MAX_LOG_BODY_TAIL_BYTES]
                        yield chunk
                finally:
                    if response_artifact_handle is not None:
                        try:
                            response_artifact_handle.close()
                        except Exception:
                            logger.debug(
                                "Failed to close response body artifact handle",
                                exc_info=True,
                            )
                    if response_body_hasher is not None:
                        response_body_sha256 = response_body_hasher.hexdigest()

            response.body_iterator = wrapped_body_iterator()

            def log_response_body() -> None:
                if truncated:
                    head_bytes = min(len(captured), MAX_LOG_BODY_HEAD_BYTES)
                    head_text = self._decode_body_preview_for_logging(
                        bytes(captured[:head_bytes]),
                        media_type,
                    )
                    tail_text = self._decode_body_preview_for_logging(
                        bytes(tail),
                        media_type,
                    )
                    body_text = _format_head_tail_preview(
                        head_text=head_text,
                        tail_text=tail_text,
                        head_bytes=head_bytes,
                        tail_bytes=len(tail),
                        total_bytes=total_bytes,
                    )
                else:
                    body_text = self._decode_body_preview_for_logging(
                        bytes(captured),
                        media_type,
                    )

                response_body_log = self._format_body_for_logging(body_text, truncated)
                if response_body_sha256:
                    response_body_log += f"\n\n<Body sha256={response_body_sha256}>"
                if response_body_artifact:
                    response_body_log += f"\n<Body artifact={response_body_artifact}>"

                logger.info(
                    f"Response [{request_id}] took {process_time:.2f}s:\n"
                    f"Status: {response.status_code}\n"
                    f"Headers:\n{json.dumps(dict(response.headers), indent=2)}\n"
                    f"Body:\n{escape(response_body_log)}",
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
        finally:
            reset_request_id(request_id_token)

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

    async def _get_request_body_for_logging(
        self, request: Request, *, request_id: str
    ) -> tuple[str, bool, str | None, str | None]:
        media_type = self._request_media_type(request)

        content_length = request.headers.get("content-length")
        try:
            length = int(content_length) if content_length is not None else None
        except ValueError:
            length = None

        if not self._is_textual_media_type(media_type):
            return f"<Skipped: content-type={media_type or 'unknown'}>", False, None, None
        if length is not None and length > MAX_LOG_BODY_BYTES and media_type != "application/json":
            return (
                f"<Skipped: content-type={media_type or 'unknown'}, content-length={length}>",
                False,
                None,
                None,
            )

        try:
            raw = await request.body()
        except Exception:
            return "", False, None, None

        truncated = len(raw) > MAX_LOG_BODY_BYTES
        sha256: str | None = None
        artifact: str | None = None
        if log_artifacts.artifacts_enabled_http():
            sha256 = hashlib.sha256(raw).hexdigest()
            gzip_enabled = log_artifacts.artifacts_gzip_enabled()
            extension = "json" if media_type == "application/json" else "txt"
            artifact_path = log_artifacts.http_request_artifact_path(
                request_id=request_id,
                extension=extension,
            )
            try:
                written = await asyncio.to_thread(
                    log_artifacts.write_artifact_bytes,
                    artifact_path,
                    raw,
                    gzip_enabled=gzip_enabled,
                )
                artifact = log_artifacts.redact_home_path(written)
            except Exception:
                artifact = None
        if not truncated:
            return raw.decode(errors="replace"), False, sha256, artifact

        head = raw[:MAX_LOG_BODY_HEAD_BYTES]
        tail = raw[-MAX_LOG_BODY_TAIL_BYTES:] if MAX_LOG_BODY_TAIL_BYTES else b""
        return (
            _format_head_tail_preview(
                head_text=head.decode(errors="replace"),
                tail_text=tail.decode(errors="replace"),
                head_bytes=len(head),
                tail_bytes=len(tail),
                total_bytes=len(raw),
            ),
            True,
            sha256,
            artifact,
        )
