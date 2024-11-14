from fastapi import APIRouter

from .endpoints import chat, images, models, stt, tts

api_router = APIRouter()
api_router.include_router(stt.router)
api_router.include_router(tts.router)
api_router.include_router(models.router)
api_router.include_router(images.router)
api_router.include_router(chat.router)
