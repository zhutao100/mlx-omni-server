import io

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..optional_features import ensure_extra_available
from .schema import AudioFormat, TTSRequest

router = APIRouter(tags=["text-to-speech"])

@router.post("/audio/speech")
@router.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """
    Generate audio from input text.

    Returns:
        StreamingResponse: Audio file content in the requested format
    """
    ensure_extra_available("tts")

    from .tts_service import TTSService

    tts_service = TTSService(request.model)

    try:
        audio_content = await tts_service.generate_speech(
            request=request,
        )

        response_format = request.response_format or AudioFormat.WAV

        # Create content type mapping
        content_type_mapping = {
            AudioFormat.MP3: "audio/mpeg",
            AudioFormat.OPUS: "audio/opus",
            AudioFormat.AAC: "audio/aac",
            AudioFormat.FLAC: "audio/flac",
            AudioFormat.WAV: "audio/wav",
            AudioFormat.PCM: "audio/pcm",
        }

        # Create response
        return StreamingResponse(
            io.BytesIO(audio_content),
            media_type=content_type_mapping[response_format],
            headers={
                "Content-Disposition": f'attachment; filename="speech.{response_format.value}"'
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
