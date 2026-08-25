import pytest
from unittest.mock import AsyncMock, MagicMock
from tests._cleanup import turn_off_all_switches


@pytest.mark.asyncio
async def test_turns_off_only_switch_domain():
    ha = MagicMock()
    ha.get_entities = AsyncMock(return_value=[
        {"entity_id": "switch.a", "name": "A", "state": "on"},
        {"entity_id": "switch.b", "name": "B", "state": "off"},
        {"entity_id": "light.c", "name": "C", "state": "on"},
    ])
    ha.call_service = AsyncMock()
    await turn_off_all_switches(ha)

    eids = [c.kwargs["entity_id"] for c in ha.call_service.await_args_list]
    assert set(eids) == {"switch.a", "switch.b"}  # both switches; light untouched
    for c in ha.call_service.await_args_list:
        assert c.args == ("switch", "turn_off")


@pytest.mark.asyncio
async def test_best_effort_swallows_errors():
    ha = MagicMock()
    ha.get_entities = AsyncMock(return_value=[{"entity_id": "switch.a", "name": "A", "state": "on"}])
    ha.call_service = AsyncMock(side_effect=Exception("HA down"))
    # must not raise
    await turn_off_all_switches(ha)
