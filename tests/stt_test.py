from fastapi.testclient import TestClient

from src.mlx_omni_server.main import app

client = TestClient(app)


def test_tts():
    response = client.post(
        "/audio/speech/",
        json={
            "model": "lucasnewman/f5-tts-mlx",
            "input": "The quick brown fox jumped over the lazy dog.",
            "voice": "alloy",
        },
    )
    print(f"response: {response}")

    assert response.status_code == 200
    # assert response.json() == {"username": "testuser", "email": "test@example.com"}
