import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock
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

def test_ma_client_reuses_http_client():
    """MusicAssistantClient._get_client() must return the same instance on repeated calls."""
    ma = MusicAssistantClient("http://test", "token", "entry")
    assert ma._client is None
    c1 = ma._get_client()
    c2 = ma._get_client()
    assert c1 is c2
    assert isinstance(c1, httpx.AsyncClient)

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

def _mock_http_get(states) -> AsyncMock:
    mock_resp = MagicMock()
    mock_resp.json.return_value = states
    mock_resp.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.is_closed = False
    mock_client.get = AsyncMock(return_value=mock_resp)
    return mock_client

def _mock_http_post(response_body) -> AsyncMock:
    mock_resp = MagicMock()
    mock_resp.json.return_value = response_body
    mock_resp.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.is_closed = False
    mock_client.post = AsyncMock(return_value=mock_resp)
    return mock_client

@pytest.mark.asyncio
async def test_discover_maps_both_satellites():
    mock_client = _mock_http_get(_MOCK_STATES)
    result = await _discover_satellite_players(
        "http://ha:8123", {"Authorization": "Bearer x"}, mock_client
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
    mock_client = _mock_http_get(states)
    result = await _discover_satellite_players(
        "http://ha:8123", {"Authorization": "Bearer x"}, mock_client
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
    ma = MusicAssistantClient("http://ha:8123", "token", "entry-123")
    ma._satellite_map = {"respeaker_lite": "media_player.respeaker_lite_media_player_2"}
    assert ma.resolve_player("unknown") == "media_player.respeaker_lite_media_player_2"

def test_resolve_player_none_falls_back_to_first():
    ma = MusicAssistantClient("http://ha:8123", "token", "entry-123")
    ma._satellite_map = {"respeaker_lite": "media_player.respeaker_lite_media_player_2"}
    assert ma.resolve_player(None) == "media_player.respeaker_lite_media_player_2"

def test_resolve_player_empty_map_returns_none():
    ma = MusicAssistantClient("http://ha:8123", "token", "entry-id")
    assert ma.resolve_player("respeaker_lite") is None

# ── search ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_track_returns_name_and_artist():
    ma = MusicAssistantClient("http://ha:8123", "token", "entry-123")
    body = {
        "service_response": {
            "tracks": [{
                "media_type": "track",
                "uri": "spotify://track/abc123",
                "name": "Blinding Lights",
                "artists": [{"name": "The Weeknd", "media_type": "artist"}],
            }],
            "artists": [], "albums": [], "playlists": [],
        }
    }
    ma._client = _mock_http_post(body)
    results = await ma.search("Blinding Lights", media_type="track")
    assert len(results) == 1
    assert results[0] == {
        "uri": "spotify://track/abc123",
        "name": "Blinding Lights",
        "artist": "The Weeknd",
    }

@pytest.mark.asyncio
async def test_search_artist_uses_name_as_artist():
    ma = MusicAssistantClient("http://ha:8123", "token", "entry-123")
    body = {
        "service_response": {
            "artists": [{
                "media_type": "artist",
                "uri": "spotify://artist/xyz",
                "name": "The Weeknd",
            }],
            "tracks": [], "albums": [], "playlists": [],
        }
    }
    ma._client = _mock_http_post(body)
    results = await ma.search("The Weeknd", media_type="artist")
    assert results[0]["name"] == "The Weeknd"
    assert results[0]["artist"] == "The Weeknd"

@pytest.mark.asyncio
async def test_search_empty_returns_empty_list():
    ma = MusicAssistantClient("http://ha:8123", "token", "entry-123")
    body = {
        "service_response": {
            "tracks": [], "artists": [], "albums": [], "playlists": [],
        }
    }
    ma._client = _mock_http_post(body)
    results = await ma.search("xyzzy404notfound", media_type="track")
    assert results == []
