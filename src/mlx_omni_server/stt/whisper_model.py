import os
import tempfile
from pathlib import Path
from typing import Union

from mlx_whisper import transcribe
from mlx_whisper.writers import WriteSRT, WriteVTT

from ..inference.runtime import run_blocking, run_mlx
from .schema import (
    ResponseFormat,
    STTRequestForm,
    TranscriptionResponse,
    TranscriptionWord,
)


class WhisperModel:

    async def _save_upload_file(self, file) -> str:
        suffix = Path(file.filename).suffix
        content = await file.read()

        def write_temp() -> str:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                return tmp.name

        return await run_blocking(write_temp)

    def generate(self, audio_path: str, request: STTRequestForm):
        word_timestamps = False
        if request.timestamp_granularities:
            word_timestamps = "word" in [
                g.value for g in request.timestamp_granularities
            ]

        result = transcribe(
            audio=audio_path,
            path_or_hf_repo=request.model,
            temperature=request.temperature,
            initial_prompt=request.prompt,
            language=request.language,
            word_timestamps=word_timestamps,
            verbose=False,
            condition_on_previous_text=True,
        )
        return result

    def _generate_subtitle_file(self, result: dict, format: str) -> str:
        temp_dir = None
        temp_file = None
        try:
            temp_dir = tempfile.mkdtemp()
            temp_file = os.path.join(temp_dir, f"temp.{format}")

            if format == "srt":
                writer = WriteSRT(temp_dir)
            else:  # vtt
                writer = WriteVTT(temp_dir)

            writer(result, temp_file)

            with open(temp_file, "r", encoding="utf-8") as f:
                return f.read()

        finally:
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
            if temp_dir and os.path.exists(temp_dir):
                os.rmdir(temp_dir)

    def _format_response(
        self, result: dict, request: STTRequestForm
    ) -> Union[dict, str, TranscriptionResponse]:
        if request.response_format == ResponseFormat.TEXT:
            return result["text"]

        elif request.response_format == ResponseFormat.SRT:
            return self._generate_subtitle_file(result, "srt")

        elif request.response_format == ResponseFormat.VTT:
            return self._generate_subtitle_file(result, "vtt")

        elif request.response_format == ResponseFormat.VERBOSE_JSON:
            return result

        elif request.response_format == ResponseFormat.JSON:
            return {"text": result["text"]}

        else:
            text = result.get("text", "")
            language = result.get("language", "en")

            duration = 0
            if "segments" in result:
                for segment in result["segments"]:
                    if "end" in segment:
                        duration = max(duration, segment["end"])

            words = []
            if request.timestamp_granularities and "word" in [
                g.value for g in request.timestamp_granularities
            ]:
                for segment in result.get("segments", []):
                    for word_data in segment.get("words", []):
                        word = TranscriptionWord(
                            word=word_data["word"],
                            start=word_data["start"],
                            end=word_data["end"],
                        )
                        words.append(word)

            return TranscriptionResponse(
                task="transcribe",
                language=language,
                duration=duration,
                text=text,
                words=words if words else None,
            )


class STTService:
    def __init__(self):
        self.model = WhisperModel()

    async def transcribe(
        self,
        request: STTRequestForm,
    ) -> Union[dict, str, TranscriptionResponse]:
        audio_path: str | None = None
        try:
            audio_path = await self.model._save_upload_file(request.file)
            result = await run_mlx(self.model.generate, audio_path, request)
            return await run_blocking(self.model._format_response, result, request)
        finally:
            if audio_path:
                Path(audio_path).unlink(missing_ok=True)
