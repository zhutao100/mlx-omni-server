from fastapi import FastAPI
from .api import stt, tts
import sys
import uvicorn

app = FastAPI(title="MLX Omni Server")

# Register the STT router
app.include_router(stt.router)
app.include_router(tts.router)



def start():
    uvicorn.run("mlx_omni_server.main:app", host="0.0.0.0", port=8000, reload=True)
