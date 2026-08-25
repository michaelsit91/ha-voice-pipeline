import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pipeline.agents import executor
from pipeline.agents.executor import _run_step


@pytest.mark.asyncio
async def test_disallowed_service_refused_without_ha_call():
    ha = MagicMock()
    ha.call_service = AsyncMock()
    ha.get_state = AsyncMock()
    step = {"domain": "hassio", "service": "host_reboot", "entity_id": "x"}
    result = await _run_step(step, ha)
    assert result["outcome"] == "failed"
    ha.call_service.assert_not_awaited()


@pytest.mark.asyncio
async def test_allowed_service_executes():
    ha = MagicMock()
    ha.call_service = AsyncMock(return_value={})
    ha.get_state = AsyncMock()
    step = {"domain": "light", "service": "turn_on", "entity_id": "light.a"}
    with patch.object(executor, "_ZIGBEE_SETTLE_S", 0):
        result = await _run_step(step, ha)
    assert result["outcome"] == "success"
    ha.call_service.assert_awaited_once()


@pytest.mark.asyncio
async def test_service_data_not_permitted_for_the_service_is_dropped():
    """The allowlist gates which service runs; unpermitted parameters must not
    ride along with it into HA."""
    ha = MagicMock()
    ha.call_service = AsyncMock(return_value={})
    ha.get_state = AsyncMock()
    step = {"domain": "light", "service": "turn_on", "entity_id": "light.a",
            "brightness_pct": 100, "effect": "colorloop"}
    with patch.object(executor, "_ZIGBEE_SETTLE_S", 0):
        await _run_step(step, ha)
    _, kwargs = ha.call_service.call_args
    assert "brightness_pct" not in kwargs and "effect" not in kwargs


@pytest.mark.asyncio
async def test_service_data_permitted_for_the_service_is_forwarded():
    ha = MagicMock()
    ha.call_service = AsyncMock(return_value={})
    ha.get_state = AsyncMock()
    step = {"domain": "media_player", "service": "volume_set",
            "entity_id": "media_player.a", "volume_level": 0.4}
    with patch.object(executor, "_ZIGBEE_SETTLE_S", 0):
        await _run_step(step, ha)
    _, kwargs = ha.call_service.call_args
    assert kwargs["volume_level"] == 0.4


@pytest.mark.asyncio
async def test_extra_music_steps_are_reported_not_silently_dropped():
    ha = MagicMock()
    ha.call_service = AsyncMock(return_value={})
    ma = MagicMock()
    ma.search = AsyncMock(return_value=[{"uri": "u", "name": "A Song", "artist": "X"}])
    steps = [{"domain": "music_assistant", "service": "play_media",
              "query": "first", "entity_id": "media_player.a"},
             {"domain": "music_assistant", "service": "play_media",
              "query": "second", "entity_id": "media_player.a"}]
    with patch.object(executor.log, "warning") as warn:
        await executor.execute(intent="action", steps=steps, ha=ha,
                               ollama=MagicMock(), ma=ma)
    assert any("second" in str(call.args) for call in warn.call_args_list)
