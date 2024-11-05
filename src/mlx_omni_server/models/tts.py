from enum import Enum
from pydantic import BaseModel, Field, validator
from typing import Optional

class AudioFormat(str, Enum):
    MP3 = "mp3"
    OPUS = "opus"
    AAC = "aac"
    FLAC = "flac"
    WAV = "wav"
    PCM = "pcm"

class TTSRequest(BaseModel):
    model: str = Field(..., description="TTS model to use (e.g. tts-1, tts-1-hd)")
    input: str = Field(..., max_length=4096)
    voice: str = Field(..., description="Voice to use (e.g. alloy, echo, fable, onyx, nova, shimmer)")
    response_format: Optional[AudioFormat] = Field(default=AudioFormat.MP3)
    speed: Optional[float] = Field(default=1.0)

    @validator('speed')
    def validate_speed(cls, v):
        if v < 0.25 or v > 4.0:
            raise ValueError('Speed must be between 0.25 and 4.0')
        return v

    @validator('model')
    def validate_model(cls, v):
        valid_models = ['tts-1', 'tts-1-hd']
        if v not in valid_models:
            raise ValueError(f'Model must be one of: {", ".join(valid_models)}')
        return v

    @validator('voice')
    def validate_voice(cls, v):
        valid_voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
        if v not in valid_voices:
            raise ValueError(f'Voice must be one of: {", ".join(valid_voices)}')
        return v
