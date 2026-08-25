import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pipeline.agents import executor
from pipeline.agents.executor import _run_step

_STEP = {"domain": "light", "service": "turn_on", "entity_id": "light.a"}


def _make_ha(states):
    """states: list of return values for successive get_state calls."""
    ha = MagicMock()
    ha.call_service = AsyncMock(return_value={})
    ha.get_state = AsyncMock(side_effect=[
        {"entity_id": "light.a", "state": s, "attributes": {}} for s in states
    ])
    return ha


@pytest.mark.asyncio
async def test_settle_zero_skips_readback():
    ha = _make_ha([])  # get_state must not be called
    with patch.object(executor, "_ZIGBEE_SETTLE_S", 0), \
         patch("asyncio.sleep", new=AsyncMock()) as sleep:
        result = await _run_step(_STEP, ha)
    assert result["outcome"] == "success"
    ha.get_state.assert_not_awaited()
    sleep.assert_not_awaited()
    ha.call_service.assert_awaited_once()


@pytest.mark.asyncio
async def test_settle_positive_detects_already():
    ha = _make_ha(["on", "on"])  # before == after → already
    with patch.object(executor, "_ZIGBEE_SETTLE_S", 0.4), \
         patch("asyncio.sleep", new=AsyncMock()):
        result = await _run_step(_STEP, ha)
    assert result["outcome"] == "already"


@pytest.mark.asyncio
async def test_settle_positive_detects_success():
    ha = _make_ha(["off", "on"])  # before != after → success
    with patch.object(executor, "_ZIGBEE_SETTLE_S", 0.4), \
         patch("asyncio.sleep", new=AsyncMock()):
        result = await _run_step(_STEP, ha)
    assert result["outcome"] == "success"


@pytest.mark.asyncio
async def test_failure_caught_in_fast_mode():
    ha = MagicMock()
    ha.call_service = AsyncMock(side_effect=Exception("HA down"))
    ha.get_state = AsyncMock()
    with patch.object(executor, "_ZIGBEE_SETTLE_S", 0):
        result = await _run_step(_STEP, ha)
    assert result["outcome"] == "failed"
