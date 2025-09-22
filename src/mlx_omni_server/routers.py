from fastapi import APIRouter

from .chat import router as chat_router
from .chat.models import router as models_router
from .embeddings import router as embeddings_router
from .images import images
from .responses import router as responses_router
from .stt import stt as stt_router
from .tts import tts as tts_router

api_router = APIRouter()
api_router.include_router(stt_router.router)
api_router.include_router(tts_router.router)
api_router.include_router(models_router.router)
api_router.include_router(images.router)
api_router.include_router(chat_router.router)
api_router.include_router(embeddings_router.router)
api_router.include_router(responses_router.router)
