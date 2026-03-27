from __future__ import annotations

import gzip
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mlx_omni_server.middleware.logging import (
    MAX_LOG_BODY_BYTES,
    RequestResponseLoggingMiddleware,
)
from mlx_omni_server.utils.logger import configured_run_id


def _make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestResponseLoggingMiddleware)

    @app.post("/echo")  # type: ignore[misc]
    async def echo(payload: dict) -> dict:
        return {"ok": True, "length": len(str(payload.get("data") or ""))}

    return app


def test_request_body_truncation_includes_head_and_tail(caplog) -> None:
    app = _make_app()

    payload = "HEAD_MARKER" + ("x" * (MAX_LOG_BODY_BYTES * 2)) + "TAIL_MARKER"

    caplog.set_level(logging.INFO, logger="mlx_omni_server")
    with TestClient(app) as client:
        response = client.post("/echo", json={"data": payload})
        assert response.status_code == 200

    request_logs = [
        record.message for record in caplog.records if record.message.startswith("Request [")
    ]
    assert request_logs

    message = request_logs[-1]
    assert "<...snip...>" in message
    assert "HEAD_MARKER" in message
    assert "TAIL_MARKER" in message


def test_http_body_artifacts_are_written(monkeypatch, tmp_path: Path, caplog) -> None:
    monkeypatch.setenv("MLX_OMNI_SERVER_LOG_HTTP_BODY_ARTIFACTS", "1")
    monkeypatch.setenv("MLX_OMNI_SERVER_LOG_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.delenv("MLX_OMNI_SERVER_LOG_ARTIFACTS_GZIP", raising=False)

    app = _make_app()
    caplog.set_level(logging.INFO, logger="mlx_omni_server")

    payload = "HEAD_MARKER" + ("x" * (MAX_LOG_BODY_BYTES * 2)) + "TAIL_MARKER"
    with TestClient(app) as client:
        response = client.post("/echo", json={"data": payload})
        assert response.status_code == 200

    run_dir = tmp_path / configured_run_id()
    request_files = sorted(run_dir.glob("*-http-request.json"))
    response_files = sorted(run_dir.glob("*-http-response.json"))
    assert request_files
    assert response_files

    request_body = request_files[0].read_text(encoding="utf-8")
    assert "HEAD_MARKER" in request_body
    assert "TAIL_MARKER" in request_body

    response_body = response_files[0].read_text(encoding="utf-8")
    assert '"ok"' in response_body


def test_http_body_artifacts_support_gzip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MLX_OMNI_SERVER_LOG_HTTP_BODY_ARTIFACTS", "1")
    monkeypatch.setenv("MLX_OMNI_SERVER_LOG_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("MLX_OMNI_SERVER_LOG_ARTIFACTS_GZIP", "1")

    app = _make_app()

    payload = "HEAD_MARKER" + ("x" * (MAX_LOG_BODY_BYTES * 2)) + "TAIL_MARKER"
    with TestClient(app) as client:
        response = client.post("/echo", json={"data": payload})
        assert response.status_code == 200

    run_dir = tmp_path / configured_run_id()
    request_files = sorted(run_dir.glob("*-http-request.json.gz"))
    response_files = sorted(run_dir.glob("*-http-response.json.gz"))
    assert request_files
    assert response_files

    with gzip.open(request_files[0], "rt", encoding="utf-8") as handle:
        text = handle.read()
    assert "HEAD_MARKER" in text
    assert "TAIL_MARKER" in text
