"""Tests for music stop/pause/volume voice commands."""
import pytest
import re
from pipeline.agents.planner import _HESITATION_PATTERNS


# ── Task 1: hesitation pattern ────────────────────────────────────────────────

def test_stop_the_music_not_hesitation():
    assert not _HESITATION_PATTERNS.search("stop the music")

def test_stop_playing_not_hesitation():
    assert not _HESITATION_PATTERNS.search("stop playing")

def test_stop_that_song_not_hesitation():
    assert not _HESITATION_PATTERNS.search("stop that song")

def test_stop_the_track_not_hesitation():
    assert not _HESITATION_PATTERNS.search("stop the track")

def test_bare_stop_is_still_hesitation():
    assert _HESITATION_PATTERNS.search("stop")

def test_stop_with_intermediate_words_not_hesitation():
    # "(?:\w+\s+)*" allows any words between stop and the music keyword
    assert not _HESITATION_PATTERNS.search("stop all the music")

def test_stop_alone_in_context_is_still_hesitation():
    assert _HESITATION_PATTERNS.search("please stop")


# ── Task 2: planner music control rules ───────────────────────────────────────

import json
from unittest.mock import AsyncMock, MagicMock


def _make_ollama_returning(json_obj):
    """Return a mock OllamaClient whose .chat() resolves to the given JSON string."""
    ollama = MagicMock()
    ollama.chat = AsyncMock(return_value=json.dumps(json_obj))
    return ollama


_MUSIC_ENTITY = {
    "entity_id": "media_player.respeaker_lite_media_player_2",
    "name": "Spotify",
    "state": "playing",
}

_MEDIA_STOP_PLAN = {
    "corrected": "stop the music",
    "intent": "action",
    "steps": [{
        "domain": "media_player",
        "service": "media_stop",
        "entity_id": "media_player.respeaker_lite_media_player_2",
    }],
    "ok_response": "Music stopped.",
    "already_response": "",
    "fail_response": "Sorry, I couldn't stop the music.",
}

_MEDIA_PAUSE_PLAN = {
    "corrected": "pause",
    "intent": "action",
    "steps": [{
        "domain": "media_player",
        "service": "media_pause",
        "entity_id": "media_player.respeaker_lite_media_player_2",
    }],
    "ok_response": "Paused.",
    "already_response": "",
    "fail_response": "Sorry, I couldn't pause the music.",
}


@pytest.mark.asyncio
async def test_planner_stop_music_emits_media_stop():
    from pipeline.agents.planner import plan
    ollama = _make_ollama_returning(_MEDIA_STOP_PLAN)
    result = await plan("stop the music", [_MUSIC_ENTITY], [], ollama)
    steps = result["steps"]
    assert len(steps) == 1
    assert steps[0]["domain"] == "media_player"
    assert steps[0]["service"] == "media_stop"
    assert result["ok_response"] == "Music stopped."


@pytest.mark.asyncio
async def test_planner_pause_emits_media_pause():
    from pipeline.agents.planner import plan
    ollama = _make_ollama_returning(_MEDIA_PAUSE_PLAN)
    result = await plan("pause", [_MUSIC_ENTITY], [], ollama)
    steps = result["steps"]
    assert len(steps) == 1
    assert steps[0]["domain"] == "media_player"
    assert steps[0]["service"] == "media_pause"
    assert result["ok_response"] == "Paused."


# ── Task 3: runner injects MA player into media_stop / media_pause ────────────

from unittest.mock import patch
from pipeline.runner import run_pipeline
from pipeline.music_assistant_client import MusicAssistantClient


def _make_ha_runner(entities=None, areas=None):
    ha = MagicMock()
    ha.get_entities = AsyncMock(return_value=entities or [
        {"entity_id": "media_player.respeaker_lite_media_player_2",
         "name": "Spotify", "state": "playing"},
        {"entity_id": "media_player.wrong_player",
         "name": "Wrong", "state": "idle"},
    ])
    ha.get_areas = AsyncMock(return_value=areas or [])
    return ha


def _make_ma_runner(player="media_player.respeaker_lite_media_player_2"):
    ma = MagicMock(spec=MusicAssistantClient)
    ma.resolve_player = MagicMock(return_value=player)
    return ma


def _plan_with_service(service: str):
    return {
        "corrected": "stop the music",
        "intent": "action",
        "steps": [{
            "domain": "media_player",
            "service": service,
            "entity_id": "media_player.wrong_player",
        }],
        "ok_response": "Music stopped.", "already_response": "", "fail_response": "Sorry.",
    }


@pytest.mark.asyncio
async def test_runner_injects_ma_player_into_media_stop():
    ha = _make_ha_runner()
    ollama = _make_ollama_returning(_plan_with_service("media_stop"))
    ma = _make_ma_runner("media_player.respeaker_lite_media_player_2")

    with patch("pipeline.runner.execute") as mock_exec:
        mock_exec.return_value = "Music stopped."
        await run_pipeline("stop the music", ha, ollama, ma=ma, satellite="respeaker_lite")

    called_steps = mock_exec.call_args.kwargs["steps"]
    assert called_steps[0]["entity_id"] == "media_player.respeaker_lite_media_player_2"


@pytest.mark.asyncio
async def test_runner_injects_ma_player_into_media_pause():
    ha = _make_ha_runner()
    ollama = _make_ollama_returning(_plan_with_service("media_pause"))
    ma = _make_ma_runner("media_player.respeaker_lite_media_player_2")

    with patch("pipeline.runner.execute") as mock_exec:
        mock_exec.return_value = "Paused."
        await run_pipeline("pause", ha, ollama, ma=ma, satellite="respeaker_lite")

    called_steps = mock_exec.call_args.kwargs["steps"]
    assert called_steps[0]["entity_id"] == "media_player.respeaker_lite_media_player_2"


@pytest.mark.asyncio
async def test_runner_does_not_inject_unrelated_media_player_steps():
    """media_player.turn_on (e.g. TV) must NOT get the MA player injected."""
    ha = _make_ha_runner(entities=[
        {"entity_id": "media_player.tv", "name": "TV", "state": "off"},
        {"entity_id": "media_player.respeaker_lite_media_player_2", "name": "Spotify", "state": "idle"},
    ])
    plan_result = {
        "corrected": "turn on the TV",
        "intent": "action",
        "steps": [{"domain": "media_player", "service": "turn_on", "entity_id": "media_player.tv"}],
        "ok_response": "TV on.", "already_response": "", "fail_response": "Sorry.",
    }
    ollama = _make_ollama_returning(plan_result)
    ma = _make_ma_runner("media_player.respeaker_lite_media_player_2")

    with patch("pipeline.runner.execute") as mock_exec:
        mock_exec.return_value = "TV on."
        await run_pipeline("turn on the TV", ha, ollama, ma=ma, satellite="respeaker_lite")

    called_steps = mock_exec.call_args.kwargs["steps"]
    assert called_steps[0]["entity_id"] == "media_player.tv"


# ── Volume commands ────────────────────────────────────────────────────────────

_VOLUME_UP_PLAN = {
    "corrected": "louder",
    "intent": "action",
    "steps": [{
        "domain": "media_player",
        "service": "volume_up",
        "entity_id": "media_player.respeaker_lite_media_player_2",
    }],
    "ok_response": "Volume up.", "already_response": "", "fail_response": "Sorry.",
}

_VOLUME_DOWN_PLAN = {
    "corrected": "volume down",
    "intent": "action",
    "steps": [{
        "domain": "media_player",
        "service": "volume_down",
        "entity_id": "media_player.respeaker_lite_media_player_2",
    }],
    "ok_response": "Volume down.", "already_response": "", "fail_response": "Sorry.",
}

_VOLUME_SET_PLAN = {
    "corrected": "set volume to 40%",
    "intent": "action",
    "steps": [{
        "domain": "media_player",
        "service": "volume_set",
        "entity_id": "media_player.respeaker_lite_media_player_2",
        "volume_level": 0.4,
    }],
    "ok_response": "Volume set to 40%.", "already_response": "", "fail_response": "Sorry.",
}


@pytest.mark.asyncio
async def test_planner_volume_up_emits_correct_step():
    from pipeline.agents.planner import plan
    ollama = _make_ollama_returning(_VOLUME_UP_PLAN)
    result = await plan("louder", [_MUSIC_ENTITY], [], ollama)
    step = result["steps"][0]
    assert step["domain"] == "media_player"
    assert step["service"] == "volume_up"
    assert result["ok_response"] == "Volume up."


@pytest.mark.asyncio
async def test_planner_volume_down_emits_correct_step():
    from pipeline.agents.planner import plan
    ollama = _make_ollama_returning(_VOLUME_DOWN_PLAN)
    result = await plan("volume down", [_MUSIC_ENTITY], [], ollama)
    step = result["steps"][0]
    assert step["domain"] == "media_player"
    assert step["service"] == "volume_down"


@pytest.mark.asyncio
async def test_planner_volume_set_carries_volume_level():
    from pipeline.agents.planner import plan
    ollama = _make_ollama_returning(_VOLUME_SET_PLAN)
    result = await plan("set volume to 40 percent", [_MUSIC_ENTITY], [], ollama)
    step = result["steps"][0]
    assert step["service"] == "volume_set"
    assert abs(step["volume_level"] - 0.4) < 0.001


@pytest.mark.asyncio
async def test_runner_injects_ma_player_into_volume_up():
    """volume_up step gets the satellite MA player injected just like stop/pause."""
    ha = _make_ha_runner()
    ollama = _make_ollama_returning(_plan_with_service("volume_up"))
    ma = _make_ma_runner("media_player.respeaker_lite_media_player_2")

    with patch("pipeline.runner.execute") as mock_exec:
        mock_exec.return_value = "Volume up."
        await run_pipeline("louder", ha, ollama, ma=ma, satellite="respeaker_lite")

    called_steps = mock_exec.call_args.kwargs["steps"]
    assert called_steps[0]["entity_id"] == "media_player.respeaker_lite_media_player_2"


@pytest.mark.asyncio
async def test_runner_injects_ma_player_into_volume_down():
    ha = _make_ha_runner()
    ollama = _make_ollama_returning(_plan_with_service("volume_down"))
    ma = _make_ma_runner("media_player.respeaker_lite_media_player_2")

    with patch("pipeline.runner.execute") as mock_exec:
        mock_exec.return_value = "Volume down."
        await run_pipeline("quieter", ha, ollama, ma=ma, satellite="respeaker_lite")

    called_steps = mock_exec.call_args.kwargs["steps"]
    assert called_steps[0]["entity_id"] == "media_player.respeaker_lite_media_player_2"

