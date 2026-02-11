from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subprocess wrapper for mlx-audio TTS generation.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--lang-code", default="en")
    parser.add_argument("--file-prefix", required=True)
    parser.add_argument("--audio-format", default="wav")
    parser.add_argument("--join-audio", action="store_true")
    parser.add_argument("--extra-params-json", default="{}")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    extra_params = json.loads(args.extra_params_json)

    from mlx_audio.tts.generate import generate_audio

    generate_audio(
        text=args.text,
        model=args.model,
        voice=args.voice,
        speed=args.speed,
        lang_code=args.lang_code,
        file_prefix=args.file_prefix,
        audio_format=args.audio_format,
        join_audio=args.join_audio,
        verbose=False,
        **extra_params,
    )

    expected = Path(f"{args.file_prefix}.{args.audio_format}")
    os._exit(0 if expected.exists() else 2)


if __name__ == "__main__":
    main()
