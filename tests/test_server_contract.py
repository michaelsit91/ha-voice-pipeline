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

def _patch_probes(*, ha_ok: bool, ollama_ok: bool):
    """Patch httpx.AsyncClient so /health/deep's two probes succeed or fail per-URL."""
    async def _get(url, headers=None):
        ok = ha_ok if "/api/version" not in url else ollama_ok
        if not ok:
            raise Exception("unreachable")
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        return resp

    mock_instance = AsyncMock()
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock(return_value=False)
    mock_instance.get = AsyncMock(side_effect=_get)
    return patch("httpx.AsyncClient", return_value=mock_instance)


def test_health_deep_returns_expected_shape():
    """GET /health/deep must return status and per-component health."""
    with _patch_probes(ha_ok=False, ollama_ok=False):
        r = _client().get("/health/deep")

    body = r.json()
    assert body["status"] in ("ok", "degraded", "down")
    assert "components" in body
    for key in ("ha", "ollama"):
        assert key in body["components"]
        assert body["components"][key]["status"] in ("ok", "error")


def test_health_deep_returns_200_when_all_ok():
    with _patch_probes(ha_ok=True, ollama_ok=True):
        r = _client().get("/health/deep")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_deep_returns_503_when_down():
    """A 20-day outage stayed green because the verdict was body-only — curl -sf must fail."""
    with _patch_probes(ha_ok=False, ollama_ok=False):
        r = _client().get("/health/deep")
    assert r.status_code == 503
    assert r.json()["status"] == "down"


def test_health_deep_returns_503_when_degraded():
    with _patch_probes(ha_ok=True, ollama_ok=False):
        r = _client().get("/health/deep")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "degraded"
    assert body["components"]["ha"]["status"] == "ok"
    assert body["components"]["ollama"]["status"] == "error"


def test_health_stays_unconditional_liveness():
    """/health must not gain dependency probing — it is the liveness probe."""
    with _patch_probes(ha_ok=False, ollama_ok=False):
        r = _client().get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


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
    assert isinstance(body["spotify_search_enabled"], bool)
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


def test_api_key_auth_accepts_bearer_header(monkeypatch):
    """Home Assistant's OpenAI-compatible client sends the key as Bearer, never X-API-Key."""
    monkeypatch.setenv("API_KEY", "secret-key-123")
    with patch("pipeline.server.run_pipeline", new_callable=AsyncMock, return_value="done"):
        r = _client().post("/v1/chat/completions",
                           headers={"Authorization": "Bearer secret-key-123"},
                           json={"messages": [{"role": "user", "content": "test"}]})
    assert r.status_code == 200


def test_api_key_auth_rejects_wrong_bearer_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-key-123")
    r = _client().post("/v1/chat/completions",
                       headers={"Authorization": "Bearer wrong"},
                       json={"messages": [{"role": "user", "content": "test"}]})
    assert r.status_code == 401


def test_api_key_auth_rejects_bare_authorization_value(monkeypatch):
    """The raw key without the Bearer scheme is not a valid credential."""
    monkeypatch.setenv("API_KEY", "secret-key-123")
    r = _client().post("/v1/chat/completions",
                       headers={"Authorization": "secret-key-123"},
                       json={"messages": [{"role": "user", "content": "test"}]})
    assert r.status_code == 401


def test_reload_accepts_bearer_header(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-key-123")
    with patch("pipeline.server._ma") as mock_ma:
        mock_ma.discover = AsyncMock()
        mock_ma._satellite_map = {}
        r = _client().post("/reload", headers={"Authorization": "Bearer secret-key-123"})
    assert r.status_code == 200


def test_health_never_requires_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-key-123")
    r = _client().get("/health")
    assert r.status_code == 200


# ── Startup warmup ────────────────────────────────────────────────────────────

def test_warmup_uses_configured_num_ctx(monkeypatch):
    """Ollama keys a resident model on num_ctx, so a hardcoded warmup context
    loads a second instance and the first real command pays the load cost."""
    from pipeline.server import _ollama
    monkeypatch.setattr(_ollama, "_num_ctx", 65536, raising=False)

    sent: dict = {}

    async def _post(url, json=None):
        sent["url"] = url
        sent["json"] = json
        return MagicMock()

    mock_instance = AsyncMock()
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock(return_value=False)
    mock_instance.post = AsyncMock(side_effect=_post)

    with patch("httpx.AsyncClient", return_value=mock_instance), \
         patch("pipeline.server._ma") as mock_ma:
        mock_ma.discover = AsyncMock()
        mock_ma.close = AsyncMock()
        mock_ma._satellite_map = {}
        with TestClient(app):
            pass

    assert sent["json"]["options"]["num_ctx"] == 65536
    assert sent["json"]["keep_alive"] == -1


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


def test_chat_completions_malformed_body_returns_400():
    """Non-JSON request body must return 400, not 500."""
    r = _client().post(
        "/v1/chat/completions",
        content=b"this is not json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400
    assert "error" in r.json()
