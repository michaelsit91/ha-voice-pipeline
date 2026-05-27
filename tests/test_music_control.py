"""Tests for music stop/pause voice commands."""
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

import asyncio
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


@pytest.mark.asyncio
async def test_planner_pause_emits_media_pause():
    from pipeline.agents.planner import plan
    ollama = _make_ollama_returning(_MEDIA_PAUSE_PLAN)
    result = await plan("pause", [_MUSIC_ENTITY], [], ollama)
    steps = result["steps"]
    assert len(steps) == 1
    assert steps[0]["domain"] == "media_player"
    assert steps[0]["service"] == "media_pause"


@pytest.mark.asyncio
async def test_planner_stop_music_ok_response():
    from pipeline.agents.planner import plan
    ollama = _make_ollama_returning(_MEDIA_STOP_PLAN)
    result = await plan("stop the music", [_MUSIC_ENTITY], [], ollama)
    assert result["ok_response"] == "Music stopped."


@pytest.mark.asyncio
async def test_planner_pause_ok_response():
    from pipeline.agents.planner import plan
    ollama = _make_ollama_returning(_MEDIA_PAUSE_PLAN)
    result = await plan("pause", [_MUSIC_ENTITY], [], ollama)
    assert result["ok_response"] == "Paused."
