import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from pipeline.agents.executor import _run_music_step, _run_volume_step, execute, _VOLUME_STEP
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


# ── _run_volume_step: 10% steps via volume_set ───────────────────────────────

def _make_ha_with_volume(current_volume: float) -> MagicMock:
    ha = MagicMock()
    ha.get_state = AsyncMock(return_value={
        "entity_id": "media_player.respeaker",
        "state": "playing",
        "attributes": {"volume_level": current_volume},
    })
    ha.call_service = AsyncMock(return_value={})
    return ha


@pytest.mark.asyncio
async def test_volume_up_adds_10_percent():
    ha = _make_ha_with_volume(0.49)
    result = await _run_volume_step(ha, "media_player.respeaker", "volume_up")

    assert result["outcome"] == "success"
    ha.call_service.assert_awaited_once_with(
        "media_player", "volume_set",
        entity_id="media_player.respeaker",
        volume_level=0.59,
    )


@pytest.mark.asyncio
async def test_volume_down_subtracts_10_percent():
    ha = _make_ha_with_volume(0.49)
    result = await _run_volume_step(ha, "media_player.respeaker", "volume_down")

    assert result["outcome"] == "success"
    ha.call_service.assert_awaited_once_with(
        "media_player", "volume_set",
        entity_id="media_player.respeaker",
        volume_level=0.39,
    )


@pytest.mark.asyncio
async def test_volume_up_clamps_at_100_percent():
    ha = _make_ha_with_volume(0.95)
    await _run_volume_step(ha, "media_player.respeaker", "volume_up")

    _, kwargs = ha.call_service.call_args
    assert kwargs["volume_level"] == 1.0


@pytest.mark.asyncio
async def test_volume_down_clamps_at_0_percent():
    ha = _make_ha_with_volume(0.05)
    await _run_volume_step(ha, "media_player.respeaker", "volume_down")

    _, kwargs = ha.call_service.call_args
    assert kwargs["volume_level"] == 0.0


@pytest.mark.asyncio
async def test_volume_step_returns_failed_on_ha_error():
    ha = MagicMock()
    ha.get_state = AsyncMock(side_effect=Exception("HA down"))
    result = await _run_volume_step(ha, "media_player.respeaker", "volume_up")
    assert result["outcome"] == "failed"


@pytest.mark.asyncio
async def test_run_step_routes_volume_up_through_volume_step():
    """_run_step dispatches volume_up to _run_volume_step, not raw HA call."""
    from pipeline.agents.executor import _run_step
    ha = _make_ha_with_volume(0.5)

    result = await _run_step(
        {"domain": "media_player", "service": "volume_up",
         "entity_id": "media_player.respeaker"},
        ha,
    )

    assert result["outcome"] == "success"
    # Must call volume_set, not volume_up
    args, kwargs = ha.call_service.call_args
    assert args[1] == "volume_set"
    assert kwargs["volume_level"] == pytest.approx(0.6)


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
