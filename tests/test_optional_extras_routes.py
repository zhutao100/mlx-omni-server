from mlx_omni_server.optional_features import missing_packages


def test_images_route_returns_501_when_missing(client):
    if missing_packages("images"):
        response = client.post("/v1/images/generations", json={"prompt": "test"})
        assert response.status_code == 501
        assert "mlx-omni-server[images]" in response.json()["detail"]
        return

    response = client.post("/v1/images/generations", json={})
    assert response.status_code == 422


def test_tts_route_returns_501_when_missing(client):
    if missing_packages("tts"):
        response = client.post(
            "/v1/audio/speech",
            json={"model": "mlx-community/Kokoro-82M-4bit", "input": "test"},
        )
        assert response.status_code == 501
        assert "mlx-omni-server[tts]" in response.json()["detail"]
        return

    response = client.post("/v1/audio/speech", json={})
    assert response.status_code == 422


def test_stt_route_returns_501_when_missing(client):
    response = client.post("/v1/audio/transcriptions")

    if missing_packages("stt"):
        assert response.status_code == 501
        assert "mlx-omni-server[stt]" in response.json()["detail"]
        return

    assert response.status_code != 404
