import uvicorn
from fastapi import FastAPI

from .api.routers import api_router

app = FastAPI(title="MLX Omni Server")
app.include_router(api_router)


def start():
    uvicorn.run("mlx_omni_server.main:app", host="0.0.0.0", port=10240, reload=True)
