"""Contract and error-path tests for pipeline/server.py HTTP endpoints.

Uses FastAPI TestClient (no lifespan triggered — no live HA/Ollama needed).
"""
import ast, os, pathlib, pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Set env vars before importing server (module-level init reads them)
os.environ.setdefault("HA_URL",     "http://test-ha:8123")
os.environ.setdefault("OLLAMA_URL", "http://test-ollama:11434")
os.environ.setdefault("HA_TOKEN",   "test-token")
os.environ.setdefault("MODEL",      "test-model")

from fastapi.testclient import TestClient
from pipeline.server import app, _resolve_ollama_url


# ── Helpers ───────────────────────────────────────────────────────────────────

def _client() -> TestClient:
    """Fresh TestClient without triggering lifespan."""
    return TestClient(app, raise_server_exceptions=True)


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_ok():
    r = _client().get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ── /health/deep ──────────────────────────────────────────────────────────────

def test_health_deep_returns_expected_shape():
    """GET /health/deep must return status and per-component health."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock(side_effect=Exception("unreachable"))

    mock_instance = AsyncMock()
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock(return_value=False)
    mock_instance.get = AsyncMock(side_effect=Exception("unreachable"))

    with patch("pipeline.server._httpx" if False else "httpx.AsyncClient",
               return_value=mock_instance):
        r = _client().get("/health/deep")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded", "down")
    assert "components" in body
    for key in ("ha", "ollama"):
        assert key in body["components"]
        assert body["components"][key]["status"] in ("ok", "error")


# ── /v1/models ────────────────────────────────────────────────────────────────

def test_models_endpoint_shape():
    r = _client().get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body.get("object") == "list"
    assert isinstance(body.get("data"), list) and len(body["data"]) >= 1
    model = body["data"][0]
    assert model["object"] == "model"
    assert "id" in model


# ── /status ───────────────────────────────────────────────────────────────────

def test_status_endpoint_shape():
    r = _client().get("/status")
    assert r.status_code == 200
    body = r.json()
    assert "model" in body
    assert isinstance(body["satellite_map"], dict)
    assert isinstance(body["spotify_sync_enabled"], bool)
    assert "vram_manager_url" in body


# ── /reload ───────────────────────────────────────────────────────────────────

def test_reload_triggers_discovery_and_returns_map():
    with patch("pipeline.server._ma") as mock_ma:
        mock_ma.discover = AsyncMock()
        mock_ma._satellite_map = {"respeaker_lite": "media_player.respeaker_lite_media_player_2"}
        r = _client().post("/reload")
    assert r.status_code == 200
    assert "satellite_map" in r.json()
    mock_ma.discover.assert_called_once()


# ── /v1/chat/completions — response shape ────────────────────────────────────

def test_chat_completions_response_shape():
    with patch("pipeline.server.run_pipeline", new_callable=AsyncMock, return_value="The light is on."):
        r = _client().post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "is the light on"}]
        })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "The light is on."
    assert choice["finish_reason"] == "stop"
    assert "usage" in body


def test_chat_completions_streaming_shape():
    with patch("pipeline.server.run_pipeline", new_callable=AsyncMock, return_value="Done."):
        r = _client().post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "turn on the light"}],
            "stream": True,
        })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")
    assert "data:" in r.text
    assert "[DONE]" in r.text


# ── /v1/chat/completions — error paths ────────────────────────────────────────

def test_no_user_message_returns_400():
    r = _client().post("/v1/chat/completions", json={
        "messages": [{"role": "system", "content": "hello"}]
    })
    assert r.status_code == 400
    assert "error" in r.json()


def test_empty_messages_returns_400():
    r = _client().post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 400


def test_transcript_too_long_returns_400():
    r = _client().post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "x" * 501}]
    })
    assert r.status_code == 400
    assert "too long" in r.json().get("error", "").lower()


def test_transcript_at_limit_succeeds():
    with patch("pipeline.server.run_pipeline", new_callable=AsyncMock, return_value="ok"):
        r = _client().post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "x" * 500}]
        })
    assert r.status_code == 200


# ── API key auth ──────────────────────────────────────────────────────────────

def test_api_key_auth_rejects_missing_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-key-123")
    r = _client().post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "test"}]
    })
    assert r.status_code == 401


def test_api_key_auth_rejects_wrong_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-key-123")
    r = _client().post("/v1/chat/completions",
                       headers={"X-API-Key": "wrong"},
                       json={"messages": [{"role": "user", "content": "test"}]})
    assert r.status_code == 401


def test_api_key_auth_accepts_correct_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-key-123")
    with patch("pipeline.server.run_pipeline", new_callable=AsyncMock, return_value="done"):
        r = _client().post("/v1/chat/completions",
                           headers={"X-API-Key": "secret-key-123"},
                           json={"messages": [{"role": "user", "content": "test"}]})
    assert r.status_code == 200


def test_api_key_auth_bypassed_when_not_set(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with patch("pipeline.server.run_pipeline", new_callable=AsyncMock, return_value="done"):
        r = _client().post("/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": "test"}]})
    assert r.status_code == 200


def test_health_never_requires_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-key-123")
    r = _client().get("/health")
    assert r.status_code == 200


# ── VRAM proxy URL routing ────────────────────────────────────────────────────

def test_resolve_ollama_url_via_vram_proxy():
    assert _resolve_ollama_url("http://vram:8890", "http://ollama:11434") == "http://vram:8890/ollama"


def test_resolve_ollama_url_direct():
    assert _resolve_ollama_url("", "http://ollama:11434") == "http://ollama:11434"


def test_resolve_ollama_url_vram_with_trailing_slash():
    assert _resolve_ollama_url("http://vram:8890/", "http://ollama:11434") == "http://vram:8890/ollama"


# ── Structural checks ─────────────────────────────────────────────────────────

def test_dockerfile_uses_curl_healthcheck():
    dockerfile = pathlib.Path("Dockerfile").read_text()
    assert "curl" in dockerfile, "Dockerfile healthcheck must use curl"
    assert 'python -c "import httpx' not in dockerfile, \
        "Dockerfile must not use python -c import httpx for healthcheck"


def test_test_pipeline_has_no_local_fixtures():
    """test_pipeline.py must not redefine ha() or ollama() — conftest.py owns them."""
    src = pathlib.Path("tests/test_pipeline.py").read_text()
    # Check for @pytest.fixture decorator on ha() or ollama() functions (any form)
    assert "def ha(" not in src or "@pytest.fixture" not in src.split("def ha(")[0].split("\n")[-2], \
        "ha() fixture redefined in test_pipeline.py — remove it, conftest.py owns it"
    assert "def ollama(" not in src or "@pytest.fixture" not in src.split("def ollama(")[0].split("\n")[-2], \
        "ollama() fixture redefined in test_pipeline.py — remove it, conftest.py owns it"


def test_test_agents_has_no_hardcoded_entity_ids():
    src = pathlib.Path("tests/test_agents.py").read_text()
    assert "light.kitchen_ceiling" not in src, \
        "Hardcoded entity_id 'light.kitchen_ceiling' found in test_agents.py"
