from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends

from ..models.stt import ResponseFormat, TimestampGranularity, TranscriptionResponse, STTRequestForm
from ..services.stt_service import STTService

router = APIRouter(tags=["speech-to-text"])


@router.post("/audio/transcriptions", response_model=TranscriptionResponse)
@router.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def create_transcription(
    request: STTRequestForm = Depends()  # Use FastAPI's Depends to inject and validate the form
):
    """
    Transcribe audio file to text.
    """
    # Validate file extension
    allowed_extensions = {'flac', 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'ogg', 'wav', 'webm'}
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Must be one of: {', '.join(allowed_extensions)}"
        )

    # Process transcription
    stt_service = STTService()
    try:
        result = await stt_service.transcribe(
            file=file,
            model=request.model,
            language=request.language,
            prompt=request.prompt,
            response_format=request.response_format,
            temperature=request.temperature,
            timestamp_granularities=request.timestamp_granularities
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
