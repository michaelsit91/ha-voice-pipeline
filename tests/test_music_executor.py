import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pipeline.agents.executor import _run_music_step, execute
from pipeline.music_assistant_client import MusicAssistantClient


def _make_ma(search_results):
    ma = MagicMock(spec=MusicAssistantClient)
    ma.search = AsyncMock(return_value=search_results)
    return ma


def _make_ha(call_ok=True):
    ha = MagicMock()
    if call_ok:
        ha.call_service = AsyncMock(return_value={})
    else:
        ha.call_service = AsyncMock(side_effect=Exception("HA unreachable"))
    return ha


# ── _run_music_step ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_music_step_plays_top_result():
    step = {
        "domain": "music_assistant",
        "service": "play_media",
        "entity_id": "media_player.respeaker_lite_media_player_2",
        "query": "Blinding Lights",
        "media_type": "track",
    }
    ma = _make_ma([{"uri": "spotify://track/abc", "name": "Blinding Lights", "artist": "The Weeknd"}])
    ha = _make_ha()

    result = await _run_music_step(step, ha, ma)

    ma.search.assert_awaited_once_with("Blinding Lights", media_type="track", artist=None)
    ha.call_service.assert_awaited_once_with(
        "music_assistant", "play_media",
        entity_id="media_player.respeaker_lite_media_player_2",
        media_id="spotify://track/abc",
        media_type="track",
    )
    assert result == "Playing Blinding Lights by The Weeknd."


@pytest.mark.asyncio
async def test_music_step_no_results_returns_not_found():
    step = {
        "domain": "music_assistant",
        "service": "play_media",
        "entity_id": "media_player.respeaker_lite_media_player_2",
        "query": "xyzzy404",
        "media_type": "track",
    }
    ma = _make_ma([])
    ha = _make_ha()

    result = await _run_music_step(step, ha, ma)

    ha.call_service.assert_not_awaited()
    assert "couldn't find" in result.lower()
    assert "xyzzy404" in result


@pytest.mark.asyncio
async def test_music_step_play_failure_returns_error():
    step = {
        "domain": "music_assistant",
        "service": "play_media",
        "entity_id": "media_player.respeaker_lite_media_player_2",
        "query": "Blinding Lights",
        "media_type": "track",
    }
    ma = _make_ma([{"uri": "spotify://track/abc", "name": "Blinding Lights", "artist": "The Weeknd"}])
    ha = _make_ha(call_ok=False)

    result = await _run_music_step(step, ha, ma)

    assert "couldn't play" in result.lower()


@pytest.mark.asyncio
async def test_music_step_ma_exception_returns_error():
    step = {
        "domain": "music_assistant",
        "service": "play_media",
        "entity_id": "media_player.respeaker_lite_media_player_2",
        "query": "Blinding Lights",
        "media_type": "track",
    }
    ma = MagicMock(spec=MusicAssistantClient)
    ma.search = AsyncMock(side_effect=Exception("MA down"))
    ha = _make_ha()

    result = await _run_music_step(step, ha, ma)

    ha.call_service.assert_not_awaited()
    assert "responding" in result.lower() or "couldn't" in result.lower()


@pytest.mark.asyncio
async def test_music_step_passes_artist_to_search():
    step = {
        "domain": "music_assistant",
        "service": "play_media",
        "entity_id": "media_player.respeaker_lite_media_player_2",
        "query": "Hotel California",
        "artist": "Eagles",
        "media_type": "track",
    }
    ma = _make_ma([{"uri": "spotify://track/hc", "name": "Hotel California", "artist": "Eagles"}])
    ha = _make_ha()

    await _run_music_step(step, ha, ma)

    ma.search.assert_awaited_once_with("Hotel California", media_type="track", artist="Eagles")


@pytest.mark.asyncio
async def test_music_step_no_artist_in_result_omits_by():
    step = {
        "domain": "music_assistant",
        "service": "play_media",
        "entity_id": "media_player.respeaker_lite_media_player_2",
        "query": "chill vibes",
        "media_type": "playlist",
    }
    ma = _make_ma([{"uri": "spotify://playlist/xyz", "name": "Chill Vibes", "artist": ""}])
    ha = _make_ha()

    result = await _run_music_step(step, ha, ma)

    assert result == "Playing Chill Vibes."
    assert " by " not in result


# ── execute() branches music steps ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_routes_music_step_through_ma():
    steps = [{
        "domain": "music_assistant",
        "service": "play_media",
        "entity_id": "media_player.respeaker_lite_media_player_2",
        "query": "Blinding Lights",
        "media_type": "track",
    }]
    ma = _make_ma([{"uri": "spotify://track/abc", "name": "Blinding Lights", "artist": "The Weeknd"}])
    ha = _make_ha()
    ollama = MagicMock()

    result = await execute(
        intent="action", steps=steps, ha=ha, ollama=ollama,
        ok_response="Playing.", ma=ma,
    )

    assert "Blinding Lights" in result
    ma.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_music_step_without_ma_returns_error():
    steps = [{
        "domain": "music_assistant",
        "service": "play_media",
        "entity_id": "media_player.respeaker_lite_media_player_2",
        "query": "Blinding Lights",
        "media_type": "track",
    }]
    ha = _make_ha()
    ollama = MagicMock()

    result = await execute(
        intent="action", steps=steps, ha=ha, ollama=ollama,
        ok_response="Playing.", ma=None,
    )

    assert "not configured" in result.lower() or "sorry" in result.lower()
