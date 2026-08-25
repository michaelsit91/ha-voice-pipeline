import pytest
from unittest.mock import AsyncMock, MagicMock
from tests._cleanup import snapshot_states, restore_states


@pytest.mark.asyncio
async def test_snapshot_excludes_atx_and_non_onoff_domains():
    ha = MagicMock()
    ha.get_entities = AsyncMock(return_value=[
        {"entity_id": "light.a", "name": "A", "state": "on"},
        {"entity_id": "switch.b", "name": "B", "state": "off"},
        {"entity_id": "switch.atx_control_atx_power", "name": "ATX", "state": "on"},
        {"entity_id": "media_player.x", "name": "X", "state": "playing"},
    ])
    snap = await snapshot_states(ha)
    assert snap == {"light.a": "on", "switch.b": "off"}  # atx + media_player excluded


@pytest.mark.asyncio
async def test_restore_touches_only_drifted_devices():
    snap = {"light.a": "on", "switch.b": "off"}
    ha = MagicMock()
    # light.a drifted on→off; switch.b unchanged
    ha.get_entities = AsyncMock(return_value=[
        {"entity_id": "light.a", "name": "A", "state": "off"},
        {"entity_id": "switch.b", "name": "B", "state": "off"},
    ])
    ha.clear_cache = MagicMock()
    ha.call_service = AsyncMock()
    await restore_states(ha, snap, settle_s=0)
    ha.call_service.assert_awaited_once()
    assert ha.call_service.await_args.args[:2] == ("light", "turn_on")
    assert ha.call_service.await_args.kwargs["entity_id"] == "light.a"


@pytest.mark.asyncio
async def test_restore_never_touches_excluded_even_if_drifted():
    snap = {"switch.atx_control_atx_power": "on"}  # shouldn't even be here, but guard anyway
    ha = MagicMock()
    ha.get_entities = AsyncMock(return_value=[
        {"entity_id": "switch.atx_control_atx_power", "name": "ATX", "state": "off"},
    ])
    ha.clear_cache = MagicMock()
    ha.call_service = AsyncMock()
    await restore_states(ha, snap, settle_s=0)
    ha.call_service.assert_not_awaited()


@pytest.mark.asyncio
async def test_restore_empty_snapshot_is_safe():
    ha = MagicMock()
    ha.call_service = AsyncMock()
    await restore_states(ha, {}, settle_s=0)
    ha.call_service.assert_not_awaited()
