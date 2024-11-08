from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class AudioFormat(str, Enum):
    MP3 = "mp3"
    OPUS = "opus"
    AAC = "aac"
    FLAC = "flac"
    WAV = "wav"
    PCM = "pcm"


class TTSRequest(BaseModel):
    model: str = Field(..., description="TTS model to use")
    input: str = Field(..., max_length=4096)
    voice: str = Field(..., description="Voice to use (e.g. alloy, echo, fable, onyx, nova, shimmer)")
    response_format: Optional[AudioFormat] = Field(default=AudioFormat.WAV)
    speed: Optional[float] = Field(default=1.0)

    @field_validator('speed')
    def validate_speed(cls, v):
        if v < 0.25 or v > 4.0:
            raise ValueError('Speed must be between 0.25 and 4.0')
        return v

    @field_validator('model')
    def validate_model(cls, v):
        valid_models = ['lucasnewman/f5-tts-mlx']
        if v not in valid_models:
            raise ValueError(f'Model must be one of: {", ".join(valid_models)}')
        return v
