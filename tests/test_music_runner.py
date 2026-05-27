import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pipeline.runner import run_pipeline
from pipeline.music_assistant_client import MusicAssistantClient


def _make_ha(entities=None, areas=None):
    ha = MagicMock()
    ha.get_entities = AsyncMock(return_value=entities or [
        {"entity_id": "media_player.respeaker_lite_media_player_2",
         "name": "Spotify", "state": "playing"},
    ])
    ha.get_areas = AsyncMock(return_value=areas or [])
    return ha


def _make_ollama(plan_result):
    import json
    ollama = MagicMock()
    ollama.chat = AsyncMock(return_value=json.dumps(plan_result))
    return ollama


def _make_ma(player="media_player.respeaker_lite_media_player_2"):
    ma = MagicMock(spec=MusicAssistantClient)
    ma.resolve_player = MagicMock(return_value=player)
    ma.search = AsyncMock(return_value=[
        {"uri": "spotify://track/abc", "name": "Blinding Lights", "artist": "The Weeknd"}
    ])
    return ma


@pytest.mark.asyncio
async def test_runner_injects_satellite_player_into_music_step():
    """Runner overrides entity_id in music steps with the resolved satellite player."""
    plan_result = {
        "corrected": "play Blinding Lights",
        "intent": "action",
        "steps": [{
            "domain": "music_assistant",
            "service": "play_media",
            "entity_id": "media_player.wrong_player",
            "query": "Blinding Lights",
            "media_type": "track",
        }],
        "ok_response": "Playing Blinding Lights.",
        "already_response": "",
        "fail_response": "Sorry.",
    }
    ha = _make_ha()
    ollama = _make_ollama(plan_result)
    ma = _make_ma("media_player.respeaker_lite_media_player_2")

    with patch("pipeline.runner.execute") as mock_exec:
        mock_exec.return_value = "Playing Blinding Lights by The Weeknd."
        await run_pipeline("play Blinding Lights", ha, ollama, ma=ma, satellite="respeaker_lite")

    called_steps = mock_exec.call_args.kwargs["steps"]
    assert called_steps[0]["entity_id"] == "media_player.respeaker_lite_media_player_2"
    ma.resolve_player.assert_called_once_with("respeaker_lite")


@pytest.mark.asyncio
async def test_runner_passes_ma_to_execute():
    plan_result = {
        "corrected": "play Blinding Lights",
        "intent": "action",
        "steps": [{
            "domain": "music_assistant",
            "service": "play_media",
            "entity_id": "media_player.respeaker_lite_media_player_2",
            "query": "Blinding Lights",
            "media_type": "track",
        }],
        "ok_response": "Playing.", "already_response": "", "fail_response": "Sorry.",
    }
    ha = _make_ha()
    ollama = _make_ollama(plan_result)
    ma = _make_ma()

    with patch("pipeline.runner.execute") as mock_exec:
        mock_exec.return_value = "Playing Blinding Lights by The Weeknd."
        await run_pipeline("play Blinding Lights", ha, ollama, ma=ma, satellite="respeaker_lite")

    assert mock_exec.call_args.kwargs.get("ma") is ma


@pytest.mark.asyncio
async def test_runner_works_without_ma_for_non_music_commands():
    """Non-music commands still work when ma=None."""
    plan_result = {
        "corrected": "turn on the office light",
        "intent": "action",
        "steps": [{"domain": "light", "service": "turn_on",
                   "entity_id": "light.office_light"}],
        "ok_response": "The office light is on.",
        "already_response": "Already on.", "fail_response": "Failed.",
    }
    ha = _make_ha(entities=[{"entity_id": "light.office_light", "name": "Office Light", "state": "off"}])
    ha.call_service = AsyncMock(return_value={})
    ha.get_state = AsyncMock(return_value={"entity_id": "light.office_light", "state": "on", "attributes": {}})
    ollama = _make_ollama(plan_result)

    result = await run_pipeline("turn on the office light", ha, ollama, ma=None, satellite=None)

    assert isinstance(result, str) and len(result) > 0
