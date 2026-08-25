"""TV-noise rejection: planner classifies non-directed speech as intent=ignore,
runner responds with silence instead of a spoken error."""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipeline.agents.planner import plan, _RESPONSE_SCHEMA
from pipeline.runner import run_pipeline


def _ignore_json():
    return json.dumps({
        "corrected": "", "intent": "ignore", "steps": [],
        "ok_response": "", "already_response": "", "fail_response": "",
    })


def _make_ha():
    ha = MagicMock()
    ha.get_entities = AsyncMock(return_value=[
        {"entity_id": "light.office_light", "name": "Office Light", "state": "off"},
    ])
    ha.get_areas = AsyncMock(return_value=[{"area_id": "office", "name": "Office"}])
    return ha


class TestPlannerIgnoreIntent:
    def test_schema_allows_ignore(self):
        assert "ignore" in _RESPONSE_SCHEMA["properties"]["intent"]["enum"]

    @pytest.mark.asyncio
    async def test_ignore_returns_without_thinking_retry(self):
        """intent=ignore with no steps must NOT trigger the attempt-2 retry."""
        ollama = MagicMock()
        ollama.chat = AsyncMock(return_value=_ignore_json())
        result = await plan("flable suit of jade", [], [], ollama)
        assert result["intent"] == "ignore"
        assert ollama.chat.await_count == 1

    @pytest.mark.asyncio
    async def test_empty_action_still_retries_with_thinking(self):
        """Existing behavior: action intent with no steps retries once."""
        empty_action = json.dumps({
            "corrected": "x", "intent": "action", "steps": [],
            "ok_response": "", "already_response": "", "fail_response": "",
        })
        ollama = MagicMock()
        ollama.chat = AsyncMock(return_value=empty_action)
        await plan("do the thing", [], [], ollama)
        assert ollama.chat.await_count == 2


class TestRunnerIgnoreResponse:
    @pytest.mark.asyncio
    async def test_tv_noise_gets_silent_response(self):
        """Ignored transcript → empty response (no spoken 'Sorry...')."""
        ha = _make_ha()
        ollama = MagicMock()
        ollama.chat = AsyncMock(return_value=_ignore_json())
        resp = await run_pipeline("flable suit of jade", ha, ollama)
        assert resp == ""

    @pytest.mark.asyncio
    async def test_unparseable_command_still_speaks_error(self):
        """action intent with no steps keeps the audible 'Sorry' (real garble
        from a present user should not fail silently)."""
        empty_action = json.dumps({
            "corrected": "turn on the flux", "intent": "action", "steps": [],
            "ok_response": "", "already_response": "", "fail_response": "",
        })
        ha = _make_ha()
        ollama = MagicMock()
        ollama.chat = AsyncMock(return_value=empty_action)
        resp = await run_pipeline("turn on the flux", ha, ollama)
        assert resp == "Sorry, I didn't understand that command."
