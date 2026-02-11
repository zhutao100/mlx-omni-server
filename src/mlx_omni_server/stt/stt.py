from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, Response
from starlette.responses import PlainTextResponse

from ..optional_features import ensure_extra_available
from .schema import ResponseFormat, STTRequestForm, TranscriptionResponse

router = APIRouter(tags=["speech-to-text"])

@router.post("/audio/transcriptions", response_model=TranscriptionResponse)
@router.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def create_transcription(request: STTRequestForm = Depends()):
    """
    Transcribe audio file to text.
    """
    ensure_extra_available("stt")
    from .whisper_model import STTService

    stt_service = STTService()
    try:
        result = await stt_service.transcribe(request)

        # Return appropriate response based on format
        if request.response_format == ResponseFormat.TEXT:
            return PlainTextResponse(content=result)
        if request.response_format in (ResponseFormat.SRT, ResponseFormat.VTT):
            return Response(
                content=result,
                media_type="text/plain",
                headers={
                    "Content-Disposition": (
                        'attachment; filename="transcription.'
                        f'{request.response_format.value.lower()}"'
                    ),
                },
            )

        # JSON and VERBOSE_JSON
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
