"""Error-path UX tests: user always hears a spoken response, never a raw 500.

Uses FastAPI TestClient (no lifespan triggered — no live HA/Ollama needed).
"""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Set env vars before importing server (module-level init reads them)
os.environ.setdefault("HA_URL",     "http://test-ha:8123")
os.environ.setdefault("OLLAMA_URL", "http://test-ollama:11434")
os.environ.setdefault("HA_TOKEN",   "test-token")
os.environ.setdefault("MODEL",      "test-model")

from fastapi.testclient import TestClient
from pipeline.server import app
from pipeline.agents import executor
from pipeline.agents.executor import execute


def _client() -> TestClient:
    """Fresh TestClient without triggering lifespan."""
    return TestClient(app, raise_server_exceptions=True)


# ── /v1/chat/completions — pipeline exception → spoken apology ───────────────

def test_pipeline_exception_returns_spoken_apology():
    with patch("pipeline.server.run_pipeline", new_callable=AsyncMock,
               side_effect=Exception("HA is down")):
        r = _client().post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "turn on the light"}]
        })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    content = body["choices"][0]["message"]["content"]
    assert "sorry" in content.lower()
    assert body["choices"][0]["finish_reason"] == "stop"


def test_pipeline_exception_streaming_returns_spoken_apology():
    with patch("pipeline.server.run_pipeline", new_callable=AsyncMock,
               side_effect=Exception("HA is down")):
        r = _client().post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "turn on the light"}],
            "stream": True,
        })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")
    assert "sorry" in r.text.lower()
    assert "[DONE]" in r.text


# ── executor.execute — partial-failure LLM fallback ──────────────────────────

@pytest.mark.asyncio
async def test_partial_failure_llm_error_falls_back_to_static_sentence():
    steps = [
        {"domain": "light", "service": "turn_on", "entity_id": "light.a"},
        {"domain": "light", "service": "turn_on", "entity_id": "light.b"},
    ]
    ollama = MagicMock()
    ollama.chat = AsyncMock(side_effect=Exception("Ollama down"))
    results = [
        {"entity_id": "light.a", "outcome": "success"},
        {"entity_id": "light.b", "outcome": "failed", "error": "timeout"},
    ]
    with patch.object(executor, "_run_step", new=AsyncMock(side_effect=results)):
        text = await execute(intent="action", steps=steps, ha=MagicMock(),
                             ollama=ollama, ok_response="Light is on.")
    assert text == "Done, but 1 device(s) didn't respond."


@pytest.mark.asyncio
async def test_partial_failure_llm_ok_still_uses_llm_sentence():
    steps = [
        {"domain": "light", "service": "turn_on", "entity_id": "light.a"},
        {"domain": "light", "service": "turn_on", "entity_id": "light.b"},
    ]
    ollama = MagicMock()
    ollama.chat = AsyncMock(return_value="Light A is on but light B didn't respond.")
    results = [
        {"entity_id": "light.a", "outcome": "success"},
        {"entity_id": "light.b", "outcome": "failed", "error": "timeout"},
    ]
    with patch.object(executor, "_run_step", new=AsyncMock(side_effect=results)):
        text = await execute(intent="action", steps=steps, ha=MagicMock(),
                             ollama=ollama, ok_response="Light is on.")
    assert text == "Light A is on but light B didn't respond."
