def test_chat_completions_rejects_web_search_tool(client) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "web_search"}],
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["code"] == "invalid_request"
    assert "web_search" in payload["error"]["message"]
