from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, validator
from fastapi import Form, File, UploadFile, HTTPException

class ResponseFormat(str, Enum):
    JSON = "json"
    TEXT = "text"
    SRT = "srt"
    VERBOSE_JSON = "verbose_json"
    VTT = "vtt"

class TimestampGranularity(str, Enum):
    WORD = "word"
    SEGMENT = "segment"

class TranscriptionWord(BaseModel):
    word: str
    start: float
    end: float

class TranscriptionResponse(BaseModel):
    task: str = "transcribe"
    language: str
    duration: float
    text: str
    words: Optional[List[TranscriptionWord]] = None

# Form model for request validation
class STTRequestForm:
    def __init__(
        self,
        file: UploadFile = File(..., description="The audio file to transcribe"),
        model: str = Form(..., description="ID of the model to use"),
        language: Optional[str] = Form(None, description="The language of the input audio in ISO-639-1 format"),
        prompt: Optional[str] = Form(None, description="An optional text to guide the model's style"),
        response_format: ResponseFormat = Form(ResponseFormat.JSON, description="The format of the transcription output"),
        temperature: float = Form(0.0, description="The sampling temperature, between 0 and 1"),
        timestamp_granularities: Optional[List[TimestampGranularity]] = Form(
            None,
            description="The timestamp granularities to populate (word or segment)"
        )
    ):
        self.file = file
        self.model = model
        self.language = language
        self.prompt = prompt
        self.response_format = response_format
        self.temperature = temperature
        self.timestamp_granularities = timestamp_granularities or [TimestampGranularity.SEGMENT]

        self.validate()

    def validate(self):
        # Validate file extension
        allowed_extensions = {'flac', 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'ogg', 'wav', 'webm'}
        file_extension = self.file.filename.split('.')[-1].lower()
        if file_extension not in allowed_extensions:
            raise ValueError(
                f"File type not supported. Must be one of: {', '.join(allowed_extensions)}"
            )

        # Validate model
        valid_models = ['whisper-1']
        if self.model not in valid_models:
            raise ValueError(f'Model must be one of: {", ".join(valid_models)}')

        # Validate temperature
        if not 0 <= self.temperature <= 1:
            raise ValueError('Temperature must be between 0 and 1')

        # Validate language code if provided
        if self.language and len(self.language) != 2:
            raise ValueError('Language code must be a 2-letter ISO-639-1 code')

        # Validate timestamp_granularities when using verbose_json
        if (self.timestamp_granularities and
            self.response_format != ResponseFormat.VERBOSE_JSON):
            raise ValueError('timestamp_granularities requires response_format=verbose_json')
