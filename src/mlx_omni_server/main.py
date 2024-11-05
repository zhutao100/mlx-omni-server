from fastapi import FastAPI
from .api import stt

app = FastAPI(title="MLX Omni Server")

# Register the STT router
app.include_router(stt.router)
