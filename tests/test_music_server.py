import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


# Import the real production server.  Module-level singletons (_ha, _ollama, _ma)
# are already initialised from env vars; we patch run_pipeline at the server's
# own import site so the mock is picked up by the request handler.
import pipeline.server as _srv


def test_satellite_param_passed_to_run_pipeline():
    with patch.object(_srv, "run_pipeline", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "Playing Blinding Lights by The Weeknd."
        client = TestClient(_srv.app)

        client.post(
            "/v1/chat/completions?satellite=respeaker_lite",
            json={"messages": [{"role": "user", "content": "play Blinding Lights"}]},
        )

    assert mock_run.called
    _, kwargs = mock_run.call_args
    assert kwargs.get("satellite") == "respeaker_lite"


def test_no_satellite_param_passes_none():
    with patch.object(_srv, "run_pipeline", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "The office light is on."
        client = TestClient(_srv.app)

        client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "turn on the light"}]},
        )

    assert mock_run.called
    _, kwargs = mock_run.call_args
    assert kwargs.get("satellite") is None


def test_response_contains_pipeline_output():
    with patch.object(_srv, "run_pipeline", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "Playing Blinding Lights by The Weeknd."
        client = TestClient(_srv.app)

        r = client.post(
            "/v1/chat/completions?satellite=respeaker_lite",
            json={"messages": [{"role": "user", "content": "play Blinding Lights"}]},
        )

    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Playing Blinding Lights by The Weeknd."
