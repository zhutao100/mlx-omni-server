import argparse
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from .chat.generation_service import background_cache_cleanup
from .middleware.logging import RequestResponseLoggingMiddleware
from .optional_features import is_available
from .routers import api_router
from .utils.logger import configure_logging, default_log_dir, logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events for the application."""
    # Startup
    background_tasks = [
        asyncio.create_task(background_cache_cleanup(), name="chat-cache-cleanup"),
    ]

    if is_available("images"):
        try:
            from .images.images import get_images_service
            from .images.images_service import background_url_image_cleanup

            images_service = get_images_service()
            background_tasks.append(
                asyncio.create_task(
                    background_url_image_cleanup(images_service.output_dir),
                    name="image-url-cleanup",
                )
            )
        except Exception:
            # Images are an optional extra; avoid failing app startup if something is missing.
            logger.exception(
                "Failed to initialize images background tasks; continuing without images."
            )
    yield
    # Shutdown
    for task in background_tasks:
        task.cancel()
    for task in background_tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="MLX Omni Server", lifespan=lifespan)

# Add request/response logging middleware with custom levels
app.add_middleware(
    RequestResponseLoggingMiddleware,
    # exclude_paths=["/health"]
)

app.include_router(api_router)


def build_parser():
    """Create and configure the argument parser for the server."""
    parser = argparse.ArgumentParser(description="MLX Omni Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to, defaults to 0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10240,
        help="Port to bind the server to, defaults to 10240",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers to use, defaults to 1",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set the logging level, defaults to info",
    )
    parser.add_argument(
        "--log-file",
        action="store_true",
        help="Enable on-disk logging",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=default_log_dir(),
        help="Directory for on-disk logs (used with --log-file)",
    )
    parser.add_argument(
        "--log-file-format",
        type=str,
        default="jsonl",
        choices=["text", "jsonl"],
        help="On-disk log format (used with --log-file), defaults to jsonl",
    )
    return parser


def start():
    """Start the MLX Omni Server."""
    parser = build_parser()
    args = parser.parse_args()

    log_config = configure_logging(
        log_level=args.log_level,
        log_file=args.log_file,
        log_dir=args.log_dir,
        log_file_format=args.log_file_format,
    )

    # Start server with uvicorn
    uvicorn.run(
        "mlx_omni_server.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        log_config=log_config,
        use_colors=True,
        workers=args.workers,
    )


if __name__ == "__main__":
    start()
