# Audio (TTS + STT)

Audio routes are packaged as **optional extras**:

- TTS (`/v1/audio/speech`): `pip install ".[tts]"`
- STT (`/v1/audio/transcriptions`): `pip install ".[stt]"`

If an extra is not installed, the route remains registered but returns `501 Not Implemented` with an install hint.

> Note: audio routes are available with and without the `/v1` prefix (`/audio/...` and `/v1/audio/...`).

## Text-to-Speech (TTS) — `POST /v1/audio/speech`

### Request notes

- `model` is required.
- `voice` is backend/model-specific.
  - For `mlx-audio` models (for example `mlx-community/Kokoro-82M-4bit`), voices typically look like `af_*` (default is `af_sky`).
  - For `lucasnewman/f5-tts-mlx`, `voice` is currently ignored.
- `response_format` defaults to `wav`.
  - `lucasnewman/f5-tts-mlx` only supports `response_format=wav`.

### cURL example (mlx-audio / Kokoro)

```bash
curl -X POST "http://localhost:10240/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Kokoro-82M-4bit",
    "input": "Hello from MLX Omni Server",
    "voice": "af_sky",
    "response_format": "wav"
  }' \
  --output speech.wav
```

### cURL example (F5; wav-only)

```bash
curl -X POST "http://localhost:10240/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lucasnewman/f5-tts-mlx",
    "input": "Hello from MLX Omni Server",
    "response_format": "wav"
  }' \
  --output speech.wav
```

### Python example (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:10240/v1", api_key="not-needed")

response = client.audio.speech.create(
    model="mlx-community/Kokoro-82M-4bit",
    input="Hello from MLX Omni Server",
    voice="af_sky",
    response_format="wav",
)
response.stream_to_file("speech.wav")
```

## Speech-to-Text (STT) — `POST /v1/audio/transcriptions`

### Request notes

- This endpoint uses `multipart/form-data`.
- `timestamp_granularities[]=word` requires `response_format=verbose_json`.
- Supported `response_format`: `json`, `text`, `srt`, `vtt`, `verbose_json`.

### cURL example (default JSON)

```bash
curl -X POST "http://localhost:10240/v1/audio/transcriptions" \
  -F "file=@tests/test_audio.wav" \
  -F "model=mlx-community/whisper-large-v3-turbo"
```

### cURL example (plain text)

```bash
curl -X POST "http://localhost:10240/v1/audio/transcriptions" \
  -F "file=@tests/test_audio.wav" \
  -F "model=mlx-community/whisper-large-v3-turbo" \
  -F "response_format=text"
```

### cURL example (word timestamps; verbose JSON)

```bash
curl -X POST "http://localhost:10240/v1/audio/transcriptions" \
  -F "file=@tests/test_audio.wav" \
  -F "model=mlx-community/whisper-large-v3-turbo" \
  -F "timestamp_granularities[]=word" \
  -F "response_format=verbose_json"
```

### Python example (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:10240/v1", api_key="not-needed")

with open("tests/test_audio.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="mlx-community/whisper-large-v3-turbo",
        file=audio_file,
    )
print(transcript.text)
```
