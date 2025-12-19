from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.responses import PlainTextResponse

from ..optional_features import missing_packages, not_installed_detail

router = APIRouter(tags=["speech-to-text"])

_MISSING_DEPS = missing_packages("stt")


if not _MISSING_DEPS:
    from .schema import ResponseFormat, STTRequestForm, TranscriptionResponse
    from .whisper_model import STTService

    @router.post("/audio/transcriptions", response_model=TranscriptionResponse)
    @router.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
    async def create_transcription(request: STTRequestForm = Depends()):
        """
        Transcribe audio file to text.
        """
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

else:

    @router.post("/audio/transcriptions")
    @router.post("/v1/audio/transcriptions")
    async def create_transcription_unavailable(_request: Request):
        raise HTTPException(
            status_code=501,
            detail=not_installed_detail("stt", missing=_MISSING_DEPS),
        )
