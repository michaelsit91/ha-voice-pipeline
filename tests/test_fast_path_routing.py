import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pipeline import runner
from pipeline.agents import executor

_ENTS = [{"entity_id": "light.kitchen", "name": "Kitchen Light", "state": "off"}]
_PLAN = ('{"corrected":"x","intent":"action","steps":'
         '[{"domain":"light","service":"turn_on","entity_id":"light.kitchen"}],'
         '"ok_response":"ok","already_response":"","fail_response":""}')


def _ha():
    ha = MagicMock()
    ha.get_entities = AsyncMock(return_value=_ENTS)
    ha.get_areas = AsyncMock(return_value=[])
    ha.call_service = AsyncMock(return_value={})
    ha.get_state = AsyncMock(return_value={"entity_id": "light.kitchen", "state": "on", "attributes": {}})
    return ha


@pytest.mark.asyncio
async def test_fast_path_hit_executes_without_llm():
    ha = _ha()
    ollama = MagicMock()
    ollama.chat = AsyncMock()
    with patch.object(runner, "_FAST_PATH_ENABLED", True):
        resp = await runner.run_pipeline("turn on the kitchen light", ha, ollama)
    # Empty response = silent success ack (satellite chime replaces spoken "OK").
    assert isinstance(resp, str)
    ha.call_service.assert_awaited_once()
    ollama.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_fast_path_miss_falls_back_to_llm():
    ha = _ha()
    ollama = MagicMock()
    ollama.chat = AsyncMock(return_value=_PLAN)
    with patch.object(runner, "_FAST_PATH_ENABLED", True), \
         patch.object(executor, "_ZIGBEE_SETTLE_S", 0):
        await runner.run_pipeline("please illuminate the cooking area", ha, ollama)
    ollama.chat.assert_awaited()


@pytest.mark.asyncio
async def test_fast_path_room_command_targets_area_id():
    ha = MagicMock()
    ha.get_entities = AsyncMock(return_value=[{"entity_id": "light.kitchen_a", "name": "Kitchen A", "state": "on"}])
    ha.get_areas = AsyncMock(return_value=[{"area_id": "kitchen", "name": "Kitchen"}])
    ha.call_service = AsyncMock(return_value={})
    ollama = MagicMock()
    ollama.chat = AsyncMock()
    with patch.object(runner, "_FAST_PATH_ENABLED", True):
        await runner.run_pipeline("turn off the kitchen light", ha, ollama)
    ollama.chat.assert_not_awaited()
    assert ha.call_service.await_args.kwargs["area_id"] == "kitchen"
    assert ha.call_service.await_args.kwargs.get("entity_id") is None


@pytest.mark.asyncio
async def test_fast_path_media_stop_targets_ma_player():
    ha = _ha()
    ollama = MagicMock()
    ollama.chat = AsyncMock()
    ma = MagicMock()
    ma.resolve_player = MagicMock(return_value="media_player.respeaker")
    with patch.object(runner, "_FAST_PATH_ENABLED", True):
        await runner.run_pipeline("stop the music", ha, ollama, ma=ma, satellite="respeaker")
    ollama.chat.assert_not_awaited()
    assert ha.call_service.await_args.kwargs["entity_id"] == "media_player.respeaker"
    assert ha.call_service.await_args.args[:2] == ("media_player", "media_stop")


@pytest.mark.asyncio
async def test_fast_path_media_without_player_falls_back_to_llm():
    ha = _ha()
    ollama = MagicMock()
    ollama.chat = AsyncMock(return_value=_PLAN)
    ma = MagicMock()
    ma.resolve_player = MagicMock(return_value=None)  # no satellite player
    with patch.object(runner, "_FAST_PATH_ENABLED", True), \
         patch.object(executor, "_ZIGBEE_SETTLE_S", 0):
        await runner.run_pipeline("stop the music", ha, ollama, ma=ma, satellite=None)
    ollama.chat.assert_awaited()  # fell back


@pytest.mark.asyncio
async def test_flag_disabled_uses_llm():
    ha = _ha()
    ollama = MagicMock()
    ollama.chat = AsyncMock(return_value=_PLAN)
    with patch.object(runner, "_FAST_PATH_ENABLED", False), \
         patch.object(executor, "_ZIGBEE_SETTLE_S", 0):
        await runner.run_pipeline("turn on the kitchen light", ha, ollama)
    ollama.chat.assert_awaited()


@pytest.mark.asyncio
async def test_area_query_resolves_by_membership_not_entity_name():
    """An area query must read the devices the area actually contains. Entity ids
    and names are not evidence of membership here: the kitchen's light carries a
    living_room_* id, and a kitchen_* id is the washing machine light."""
    entities = [
        {"entity_id": "light.living_room_3_gang_1_center", "name": "Kitchen Light 1", "state": "on"},
        {"entity_id": "light.kitchen_2_gang_left", "name": "Washing Machine Light", "state": "off"},
    ]
    areas = [{"area_id": "kitchen", "name": "Kitchen",
              "entities": ["light.living_room_3_gang_1_center"]}]
    ha = MagicMock()
    ha.get_state = AsyncMock(return_value={
        "entity_id": "light.living_room_3_gang_1_center", "state": "on", "attributes": {}})

    resp = await runner._query_fast_path(
        [{"domain": "light", "service": "get_state", "area_id": "kitchen"}],
        entities, ha, areas)

    assert resp == "The Kitchen Light 1 is on."
    read = [c.args[0] for c in ha.get_state.await_args_list]
    assert read == ["light.living_room_3_gang_1_center"]
