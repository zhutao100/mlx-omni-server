from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Optional
from ..models.stt import ResponseFormat, TimestampGranularity, TranscriptionResponse
from ..services.stt_service import STTService

router = APIRouter(tags=["speech-to-text"])

@router.post("/transcriptions", response_model=TranscriptionResponse)
@router.post("/v1/transcriptions", response_model=TranscriptionResponse)
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: ResponseFormat = Form(ResponseFormat.json),
    temperature: float = Form(0.0),
    timestamp_granularities: Optional[List[TimestampGranularity]] = Form(None)
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

    # Validate model
    if model != "whisper-1":
        raise HTTPException(
            status_code=400,
            detail="Only whisper-1 model is currently supported"
        )

    # Validate temperature
    if not 0 <= temperature <= 1:
        raise HTTPException(
            status_code=400,
            detail="Temperature must be between 0 and 1"
        )

    # Process transcription
    stt_service = STTService()
    try:
        result = await stt_service.transcribe(
            file=file,
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
