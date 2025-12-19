import asyncio
import time
from pathlib import Path

import pytest

from mlx_omni_server.embeddings.schema import (
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingUsage,
)
from mlx_omni_server.optional_features import install_instructions, missing_packages

_missing_tts_deps = missing_packages("tts")
if _missing_tts_deps:
    pytest.skip(
        f"TTS extra is not installed; install with {install_instructions('tts')}. "
        f"Missing: {', '.join(_missing_tts_deps)}",
        allow_module_level=True,
    )


@pytest.mark.asyncio
async def test_tts_uses_request_scoped_temp_paths(async_client, monkeypatch):
    written_paths: list[Path] = []

    def fake_generate_audio(self, request, output_path):  # noqa: ANN001
        output_path = Path(output_path)
        written_paths.append(output_path)
        output_path.write_bytes(request.input.encode("utf-8"))
        return True

    monkeypatch.setattr(
        "mlx_omni_server.tts.tts_service.MlxAudioModel.generate_audio",
        fake_generate_audio,
    )

    req1 = {
        "model": "mlx-community/Kokoro-82M-4bit",
        "input": "one",
        "voice": "af_sky",
        "response_format": "wav",
    }
    req2 = {
        "model": "mlx-community/Kokoro-82M-4bit",
        "input": "two",
        "voice": "af_sky",
        "response_format": "wav",
    }

    resp1, resp2 = await asyncio.gather(
        async_client.post("/v1/audio/speech", json=req1),
        async_client.post("/v1/audio/speech", json=req2),
    )

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp1.content == b"one"
    assert resp2.content == b"two"
    assert len(written_paths) == 2
    assert written_paths[0] != written_paths[1]


@pytest.mark.asyncio
async def test_mlx_gate_serializes_across_endpoints(async_client, monkeypatch):
    timings: dict[str, float] = {}

    def fake_generate_audio(self, request, output_path):  # noqa: ANN001
        timings["tts_start"] = time.monotonic()
        time.sleep(0.2)
        Path(output_path).write_bytes(b"x")
        timings["tts_end"] = time.monotonic()
        return True

    monkeypatch.setattr(
        "mlx_omni_server.tts.tts_service.MlxAudioModel.generate_audio",
        fake_generate_audio,
    )

    from mlx_omni_server.embeddings.router import embeddings_service

    def fake_generate_embeddings(request):  # noqa: ANN001
        timings["emb_start"] = time.monotonic()
        time.sleep(0.2)
        response = EmbeddingResponse(
            data=[EmbeddingData(embedding=[0.0, 0.0, 0.0], index=0)],
            model=request.model,
            usage=EmbeddingUsage(prompt_tokens=1, total_tokens=1),
        )
        timings["emb_end"] = time.monotonic()
        return response

    monkeypatch.setattr(embeddings_service, "generate_embeddings", fake_generate_embeddings)

    tts_req = {
        "model": "mlx-community/Kokoro-82M-4bit",
        "input": "hello",
        "voice": "af_sky",
        "response_format": "wav",
    }
    emb_req = {"model": "test-embedding-model", "input": "hi"}

    resp_tts, resp_emb = await asyncio.gather(
        async_client.post("/v1/audio/speech", json=tts_req),
        async_client.post("/v1/embeddings", json=emb_req),
    )

    assert resp_tts.status_code == 200
    assert resp_emb.status_code == 200

    assert {"tts_start", "tts_end", "emb_start", "emb_end"} <= timings.keys()
    tts_start = timings["tts_start"]
    tts_end = timings["tts_end"]
    emb_start = timings["emb_start"]
    emb_end = timings["emb_end"]

    assert not (tts_start < emb_end and emb_start < tts_end), (
        f"MLX gate did not serialize execution: tts=({tts_start}, {tts_end}) "
        f"emb=({emb_start}, {emb_end})"
    )
