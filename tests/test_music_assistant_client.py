import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pipeline.music_assistant_client import (
    MusicAssistantClient,
    _satellite_slug,
    _physical_player_slug,
    _discover_satellite_players,
)

# ── slug helpers ──────────────────────────────────────────────────────────────

def test_satellite_slug_respeaker():
    assert _satellite_slug("assist_satellite.respeaker_lite_assist_satellite") == "respeaker_lite"

def test_satellite_slug_ha_voice():
    assert _satellite_slug(
        "assist_satellite.home_assistant_voice_09d0e0_assist_satellite"
    ) == "home_assistant_voice_09d0e0"

def test_physical_player_slug_plain():
    assert _physical_player_slug("media_player.respeaker_lite_media_player") == "respeaker_lite"

def test_physical_player_slug_numbered():
    assert _physical_player_slug("media_player.respeaker_lite_media_player_2") == "respeaker_lite"

def test_physical_player_slug_ha_voice():
    assert _physical_player_slug(
        "media_player.home_assistant_voice_09d0e0_media_player"
    ) == "home_assistant_voice_09d0e0"

# ── discovery ─────────────────────────────────────────────────────────────────

_MOCK_STATES = [
    {"entity_id": "assist_satellite.respeaker_lite_assist_satellite",
     "state": "idle", "attributes": {}},
    {"entity_id": "assist_satellite.home_assistant_voice_09d0e0_assist_satellite",
     "state": "idle", "attributes": {}},
    {"entity_id": "media_player.respeaker_lite_media_player",
     "state": "idle", "attributes": {}},
    {"entity_id": "media_player.home_assistant_voice_09d0e0_media_player",
     "state": "idle", "attributes": {}},
    {"entity_id": "media_player.respeaker_lite_media_player_2",
     "state": "playing",
     "attributes": {"mass_player_type": "player",
                    "active_queue": "media_player.respeaker_lite_media_player"}},
    {"entity_id": "media_player.home_assistant_voice_media_player",
     "state": "idle",
     "attributes": {"mass_player_type": "player",
                    "active_queue": "media_player.home_assistant_voice_09d0e0_media_player"}},
]

def _mock_http(states):
    mock_resp = MagicMock()
    mock_resp.json.return_value = states
    mock_resp.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client

@pytest.mark.asyncio
async def test_discover_maps_both_satellites():
    with patch("pipeline.music_assistant_client.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_http(_MOCK_STATES)
        result = await _discover_satellite_players(
            "http://ha:8123", {"Authorization": "Bearer x"}
        )
    assert result["respeaker_lite"] == "media_player.respeaker_lite_media_player_2"
    assert result["home_assistant_voice_09d0e0"] == "media_player.home_assistant_voice_media_player"

@pytest.mark.asyncio
async def test_discover_fallback_when_active_queue_null():
    states = []
    for s in _MOCK_STATES:
        s2 = {"entity_id": s["entity_id"], "state": s["state"],
              "attributes": dict(s["attributes"])}
        if s2["attributes"].get("mass_player_type") == "player":
            s2["attributes"]["active_queue"] = None
        states.append(s2)

    with patch("pipeline.music_assistant_client.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_http(states)
        result = await _discover_satellite_players(
            "http://ha:8123", {"Authorization": "Bearer x"}
        )
    assert result.get("respeaker_lite") == "media_player.respeaker_lite_media_player_2"

# ── resolve_player ────────────────────────────────────────────────────────────

def test_resolve_player_known_slug():
    ma = MusicAssistantClient("http://ha:8123", "token", "entry-id")
    ma._satellite_map = {
        "respeaker_lite": "media_player.respeaker_lite_media_player_2",
        "ha_voice": "media_player.home_assistant_voice_media_player",
    }
    assert ma.resolve_player("respeaker_lite") == "media_player.respeaker_lite_media_player_2"

def test_resolve_player_unknown_slug_falls_back_to_first():
    ma = MusicAssistantClient("http://ha:8123", "token", "entry-id")
    ma._satellite_map = {"respeaker_lite": "media_player.respeaker_lite_media_player_2"}
    assert ma.resolve_player("unknown") == "media_player.respeaker_lite_media_player_2"

def test_resolve_player_none_falls_back_to_first():
    ma = MusicAssistantClient("http://ha:8123", "token", "entry-id")
    ma._satellite_map = {"respeaker_lite": "media_player.respeaker_lite_media_player_2"}
    assert ma.resolve_player(None) == "media_player.respeaker_lite_media_player_2"

def test_resolve_player_empty_map_returns_none():
    ma = MusicAssistantClient("http://ha:8123", "token", "entry-id")
    assert ma.resolve_player("respeaker_lite") is None
