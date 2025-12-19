import logging
import os

import pytest

from mlx_omni_server.optional_features import (
    get_optional_extra,
    install_instructions,
    missing_packages,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_MISSING_TTS = missing_packages("tts")
_MISSING_STT = missing_packages("stt")


def _skip_if_missing(extra: str, missing: tuple[str, ...]) -> None:
    if not missing:
        return
    feature_label = get_optional_extra(extra).feature_label
    missing_text = ", ".join(missing)
    pytest.skip(
        f"{feature_label} extra is not installed; install with {install_instructions(extra)}. "
        f"Missing: {missing_text}"
    )


def test_speech(openai_client):
    """Test text-to-speech functionality using OpenAI client"""
    _skip_if_missing("tts", _MISSING_TTS)
    try:
        model = "lucasnewman/f5-tts-mlx"
        response = openai_client.audio.speech.create(
            model=model,
            input="The quick brown fox jumped over the lazy dog.",
            voice="alloy",
        )
        logger.info(f"Speech Response: {response}")

        # Validate response
        assert response is not None, "No response received"
        # If further validation of audio content is needed, add more assertions

    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        raise


def test_mlx_audio_kokoro_speech(openai_client):
    """Test text-to-speech functionality using OpenAI client"""
    _skip_if_missing("tts", _MISSING_TTS)
    try:
        model = "mlx-community/Kokoro-82M-4bit"
        response = openai_client.audio.speech.create(
            model=model,
            input="The quick brown fox jumped over the lazy dog.",
            voice="af_sky",
        )
        logger.info(f"MLX Audio Speech Response: {response}")

        # Validate response
        assert response is not None, "No response received"
        # If further validation of audio content is needed, add more assertions

    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        raise


def test_mlx_audio_dia_speech(openai_client):
    """Test text-to-speech functionality using OpenAI client"""
    _skip_if_missing("tts", _MISSING_TTS)
    try:
        model = "mlx-community/Dia-1.6B-4bit"
        response = openai_client.audio.speech.create(
            model=model,
            input="The quick brown fox jumped over the lazy dog.",
            voice="demo",
        )
        logger.info(f"MLX Audio Speech Response: {response}")

        # Validate response
        assert response is not None, "No response received"
        # If further validation of audio content is needed, add more assertions

    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        raise


def test_transcription(openai_client):
    """Test audio transcription functionality using OpenAI client"""
    _skip_if_missing("stt", _MISSING_STT)
    try:
        audio_file_path = "tests/test_audio.wav"

        if not os.path.exists(audio_file_path):
            pytest.skip(f"Audio file {audio_file_path} does not exist")

        model = "mlx-community/whisper-large-v3-turbo"

        with open(audio_file_path, "rb") as audio_file:
            response = openai_client.audio.transcriptions.create(
                model=model,
                file=audio_file,
            )

        logger.info(f"Transcription response: {response}")

        # Validate response
        assert response is not None, "No response received"
        assert hasattr(response, "text"), "Response does not contain text field"
        assert "MLX" in response.text, "Transcription text is empty"

    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        raise
