import logging

import uvicorn
from fastapi import FastAPI

from .api.routers import api_router
from .middleware.logging import RequestResponseLoggingMiddleware

app = FastAPI(title="MLX Omni Server")

# Add request/response logging middleware with custom levels
app.add_middleware(
    RequestResponseLoggingMiddleware,
    request_level=logging.DEBUG,
    response_level=logging.DEBUG,
    # exclude_paths=["/health"]
)

app.include_router(api_router)


def start():
    uvicorn.run("mlx_omni_server.main:app", host="0.0.0.0", port=10240, reload=True)
