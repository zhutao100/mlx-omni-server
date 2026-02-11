import time
from typing import Any

from fastapi import APIRouter, HTTPException

from ..inference.runtime import run_mlx
from ..optional_features import ensure_extra_available
from .schema import ImageGenerationRequest, ImageGenerationResponse

router = APIRouter(tags=["images"])

_images_service: Any | None = None


def get_images_service() -> Any:
    global _images_service
    if _images_service is None:
        from .images_service import ImagesService

        _images_service = ImagesService()
    return _images_service


@router.post("/images/generations")
@router.post("/v1/images/generations")
async def create_image(request: ImageGenerationRequest) -> ImageGenerationResponse:
    """
    Creates an image given a prompt.
    """
    ensure_extra_available("images")

    try:
        images = await run_mlx(get_images_service().generate_images, request)
        return ImageGenerationResponse(created=int(time.time()), data=images)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
