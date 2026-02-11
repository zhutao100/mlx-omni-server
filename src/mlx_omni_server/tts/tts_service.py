import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from f5_tts_mlx.generate import generate
from pydantic import BaseModel, Field  # , PrivateAttr
from typing_extensions import override

from ..inference.runtime import run_blocking, run_mlx
from .schema import AudioFormat, TTSRequest


class TTSModelAdapter(BaseModel):
    """Base class to adapt different TTS models to support the audio endpoint."""

    path_or_hf_repo: str | Path | None = Field(
        None, title="The path or the huggingface repository to load the model from."
    )

    def generate_audio(self, request: TTSRequest, output_path: str | Path) -> bool:
        """
        Generate audio from input text.
        ¨
        Args:
            request (TTSRequest): The request object containing the input text and other parameters.
            output_path (str | Path): The path to save the generated audio file.

        Returns:
            bool: True if the audio was generated successfully, False otherwise.
        """
        raise NotImplementedError

    @classmethod
    def from_path_or_hf_repo(cls, path_or_hf_repo: str | Path) -> "TTSModelAdapter":
        model_id = str(path_or_hf_repo)
        if model_id == "lucasnewman/f5-tts-mlx":
            return F5Model(path_or_hf_repo=model_id)
        return MlxAudioModel(path_or_hf_repo=model_id)


class F5Model(TTSModelAdapter):

    @override
    def generate_audio(self, request: TTSRequest, output_path: str | Path) -> bool:
        self.path_or_hf_repo = request.model
        generate(
            model_name=request.model,
            generation_text=request.input,
            speed=request.speed,
            output_path=str(output_path),
            **(request.get_extra_params() or {}),
        )
        return Path(output_path).exists()


class MlxAudioModel(TTSModelAdapter):
    path_or_hf_repo: str = Field("mlx-community/Kokoro-82M-4bit")

    @override
    def generate_audio(self, request: TTSRequest, output_path: str | Path) -> bool:
        self.path_or_hf_repo = request.model
        model_id_lower = str(self.path_or_hf_repo).lower()
        is_kokoro = "kokoro" in model_id_lower

        voice = request.voice if hasattr(request, "voice") else "af_sky"
        lang_code = voice[:1] if is_kokoro else "en"

        extra_params = request.get_extra_params() or {}
        for reserved_key in (
            "text",
            "model",
            "voice",
            "speed",
            "lang_code",
            "file_prefix",
            "audio_format",
            "join_audio",
            "verbose",
        ):
            extra_params.pop(reserved_key, None)

        file_prefix = str(output_path).rsplit(".", 1)[0]
        audio_format = request.response_format.value

        # Kokoro crashes the interpreter on shutdown in some environments (SIGBUS on macOS),
        # so we run it in a subprocess and hard-exit there after writing the output.
        if is_kokoro:
            return self._generate_audio_subprocess(
                model=str(self.path_or_hf_repo),
                text=request.input,
                voice=voice,
                speed=request.speed,
                lang_code=lang_code,
                file_prefix=file_prefix,
                audio_format=audio_format,
                extra_params=extra_params,
            )

        from mlx_audio.tts.generate import generate_audio

        generate_audio(
            text=request.input,
            model=str(self.path_or_hf_repo),
            voice=voice,
            speed=request.speed,
            lang_code=lang_code,
            file_prefix=file_prefix,
            audio_format=audio_format,
            join_audio=True,
            verbose=False,
            **extra_params,
        )

        return Path(output_path).exists()

    @staticmethod
    def _generate_audio_subprocess(
        *,
        model: str,
        text: str,
        voice: str,
        speed: float,
        lang_code: str,
        file_prefix: str,
        audio_format: str,
        extra_params: dict,
    ) -> bool:
        cmd = [
            sys.executable,
            "-m",
            "mlx_omni_server.tts.mlx_audio_worker",
            "--model",
            model,
            "--text",
            text,
            "--voice",
            voice,
            "--speed",
            str(speed),
            "--lang-code",
            lang_code,
            "--file-prefix",
            file_prefix,
            "--audio-format",
            audio_format,
            "--join-audio",
            "--extra-params-json",
            json.dumps(extra_params),
        ]
        env = {
            **os.environ,
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "PYTHONWARNINGS": "ignore::UserWarning:multiprocessing.resource_tracker",
        }
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        expected = Path(f"{file_prefix}.{audio_format}")
        if expected.exists():
            return True
        if result.returncode != 0 and (result.stdout or result.stderr):
            raise RuntimeError(
                "mlx-audio subprocess failed: "
                f"returncode={result.returncode}, stdout={result.stdout!r}, stderr={result.stderr!r}"
            )
        return False


class TTSService:
    model: TTSModelAdapter

    def __init__(self, path_or_hf_repo: str | Path | None = None):
        if path_or_hf_repo is None:
            self.model = MlxAudioModel()
        else:
            self.model = TTSModelAdapter.from_path_or_hf_repo(path_or_hf_repo)

    async def generate_speech(
        self,
        request: TTSRequest,
    ) -> bytes:
        response_format = request.response_format or AudioFormat.WAV
        if isinstance(self.model, F5Model) and response_format is not AudioFormat.WAV:
            raise ValueError("lucasnewman/f5-tts-mlx only supports response_format=wav")

        suffix = response_format.value
        with TemporaryDirectory(prefix="mlx_omni_tts_") as tmp_dir:
            expected_path = Path(tmp_dir) / f"speech.{suffix}"

            generated = await run_mlx(self.model.generate_audio, request, expected_path)
            if not generated or not expected_path.exists():
                candidates = sorted(Path(tmp_dir).glob("speech.*"))
                if len(candidates) == 1:
                    expected_path = candidates[0]
                else:
                    raise FileNotFoundError(
                        f"Expected TTS output at {expected_path}, but it was not created"
                    )

            return await run_blocking(expected_path.read_bytes)
