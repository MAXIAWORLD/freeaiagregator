import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from api import app
    return TestClient(app)


def test_models_endpoint(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) >= 1
    assert body["data"][0]["id"] == "freeai-gateway"


def test_chat_completions_success(client):
    with patch("api.gateway") as mock_gw:
        mock_gw.ask_with_messages.return_value = "Bonjour !"
        payload = {
            "model": "freeai-gateway",
            "messages": [{"role": "user", "content": "Dis bonjour"}],
        }
        r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Bonjour !"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_all_providers_fail(client):
    with patch("api.gateway") as mock_gw:
        mock_gw.ask_with_messages.side_effect = RuntimeError("Tous les providers ont échoué.")
        payload = {
            "model": "freeai-gateway",
            "messages": [{"role": "user", "content": "test"}],
        }
        r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 503
    assert "providers" in r.json()["detail"].lower() or "échoué" in r.json()["detail"]


def test_chat_completions_empty_messages(client):
    payload = {"model": "freeai-gateway", "messages": []}
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 422


def test_ask_with_messages_routes_providers(tmp_path):
    from freeai_gateway import FreeIAGateway
    gw = FreeIAGateway(session_id="test_api", db_path=str(tmp_path / "test.db"))

    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "OK"
    mock_resp.usage.total_tokens = 10

    with patch("freeai_gateway.litellm.completion", return_value=mock_resp):
        result = gw.ask_with_messages([{"role": "user", "content": "test"}])

    assert result == "OK"


def test_ask_with_messages_fallback(tmp_path):
    import litellm as _litellm
    from freeai_gateway import FreeIAGateway
    gw = FreeIAGateway(session_id="test_fallback", db_path=str(tmp_path / "test.db"))

    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "fallback"
    mock_resp.usage.total_tokens = 5

    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _litellm.RateLimitError("429", llm_provider="test", model="test")
        return mock_resp

    with patch("freeai_gateway.litellm.completion", side_effect=side_effect):
        result = gw.ask_with_messages([{"role": "user", "content": "test"}])

    assert result == "fallback"
    assert call_count == 2
