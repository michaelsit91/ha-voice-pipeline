import pytest
from unittest.mock import AsyncMock, patch


def test_satellite_param_passed_to_run_pipeline():
    # First load the module normally to see what we're dealing with
    import sys

    with patch.dict(sys.modules):
        with patch("pipeline.runner.run_pipeline", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Playing Blinding Lights by The Weeknd."

            # Now import and create test client
            import importlib
            if "pipeline.agents.server" in sys.modules:
                del sys.modules["pipeline.agents.server"]

            from pipeline.agents.server import app
            from fastapi.testclient import TestClient
            client = TestClient(app)

            response = client.post(
                "/v1/chat/completions?satellite=respeaker_lite",
                json={"messages": [{"role": "user", "content": "play Blinding Lights"}]},
            )

    assert mock_run.called
    _, kwargs = mock_run.call_args
    assert kwargs.get("satellite") == "respeaker_lite"


def test_no_satellite_param_passes_none():
    import sys

    with patch.dict(sys.modules):
        with patch("pipeline.runner.run_pipeline", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "The office light is on."

            if "pipeline.agents.server" in sys.modules:
                del sys.modules["pipeline.agents.server"]

            from pipeline.agents.server import app
            from fastapi.testclient import TestClient
            client = TestClient(app)

            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "turn on the light"}]},
            )

    assert mock_run.called
    _, kwargs = mock_run.call_args
    assert kwargs.get("satellite") is None


def test_response_contains_pipeline_output():
    import sys

    with patch.dict(sys.modules):
        with patch("pipeline.runner.run_pipeline", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Playing Blinding Lights by The Weeknd."

            if "pipeline.agents.server" in sys.modules:
                del sys.modules["pipeline.agents.server"]

            from pipeline.agents.server import app
            from fastapi.testclient import TestClient
            client = TestClient(app)

            r = client.post(
                "/v1/chat/completions?satellite=respeaker_lite",
                json={"messages": [{"role": "user", "content": "play Blinding Lights"}]},
            )

    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Playing Blinding Lights by The Weeknd."
