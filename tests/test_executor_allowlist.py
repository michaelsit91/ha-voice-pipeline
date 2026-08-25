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
