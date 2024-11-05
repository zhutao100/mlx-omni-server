from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class ResponseFormat(str, Enum):
    json = "json"
    text = "text"
    srt = "srt"
    verbose_json = "verbose_json"
    vtt = "vtt"

class TimestampGranularity(str, Enum):
    word = "word"
    segment = "segment"

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
