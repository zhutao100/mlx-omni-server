import time

from fastapi import APIRouter, HTTPException

from ...schemas.images_schema import ImageGenerationRequest, ImageGenerationResponse
from ...services.images_service import ImagesService

router = APIRouter(tags=["images"])


@router.post("/images/generations")
@router.post("/v1/images/generations")
async def create_image(request: ImageGenerationRequest) -> ImageGenerationResponse:
    """
    Creates an image given a prompt.
    """
    try:
        service = ImagesService()

        # Generate images
        images = service.generate_images(request)

        # Create response
        return ImageGenerationResponse(created=int(time.time()), data=images)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
