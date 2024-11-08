from fastapi import APIRouter

from .endpoints import stt, tts

api_router = APIRouter()
api_router.include_router(stt.router)
api_router.include_router(tts.router)
