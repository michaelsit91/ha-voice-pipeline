import pytest
from pipeline.agents.planner import plan
from pipeline.agents.executor import execute

@pytest.mark.asyncio(loop_scope="session")
async def test_plan_single_action(ollama, ha_context):
    result = await plan("turn on the kitchen light", ha_context["entities"], ha_context["areas"], ollama)
    assert result["intent"] in ("action", "query")
    assert isinstance(result["steps"], list) and len(result["steps"]) >= 1
    step = result["steps"][0]
    assert "domain" in step and "service" in step
    eid = step.get("entity_id", step.get("area_id", ""))
    if isinstance(eid, list):
        assert any("kitchen" in e for e in eid)
    else:
        assert "kitchen" in eid

@pytest.mark.asyncio(loop_scope="session")
async def test_plan_fixes_stt_error(ollama, ha_context):
    result = await plan("turn on the kitchin lait", ha_context["entities"], ha_context["areas"], ollama)
    assert len(result["steps"]) >= 1
    eid = result["steps"][0].get("entity_id", result["steps"][0].get("area_id", ""))
    if isinstance(eid, list):
        assert any("kitchen" in e for e in eid)
    else:
        assert "kitchen" in eid

@pytest.mark.asyncio(loop_scope="session")
async def test_plan_status_query(ollama, ha_context):
    result = await plan("is the living room light on", ha_context["entities"], ha_context["areas"], ollama)
    assert result["intent"] == "query"
    assert len(result["steps"]) >= 1

@pytest.mark.asyncio(loop_scope="session")
async def test_plan_multi_device(ollama, ha_context):
    result = await plan("turn off all living room lights", ha_context["entities"], ha_context["areas"], ollama)
    assert result["intent"] == "action"
    assert len(result["steps"]) == 1, f"Expected 1 consolidated step, got {len(result['steps'])}: {result['steps']}"
    step = result["steps"][0]
    assert step["domain"] == "light"
    assert step["service"] == "turn_off"
    has_area = "area_id" in step
    has_list = isinstance(step.get("entity_id"), list)
    assert has_area or has_list, f"Expected area_id or list entity_id: {step}"
    if has_list:
        known = {e["entity_id"] for e in ha_context["entities"]}
        for eid in step["entity_id"]:
            assert eid in known, f"Invented entity_id: {eid}"

@pytest.mark.asyncio(loop_scope="session")
async def test_plan_returns_valid_entity_ids(ollama, ha_context):
    result = await plan("turn on the fan", ha_context["entities"], ha_context["areas"], ollama)
    known = {e["entity_id"] for e in ha_context["entities"]}
    for step in result["steps"]:
        if isinstance(step.get("entity_id"), list):
            for eid in step["entity_id"]:
                assert eid in known, f"Invented entity_id: {eid}"
        elif "entity_id" in step:
            assert step["entity_id"] in known, f"Invented entity_id: {step['entity_id']}"
        # area_id steps have no entity_id to validate here

@pytest.mark.asyncio(loop_scope="session")
async def test_execute_query_returns_state(ha, ollama):
    steps = [{"domain": "homeassistant", "service": "get_state",
              "entity_id": "light.kitchen_ceiling"}]
    result = await execute(intent="query", steps=steps, ha=ha, ollama=ollama,
                           ok_response="The kitchen ceiling light state was checked.",
                           fail_response="Could not read state.")
    assert isinstance(result, str) and len(result) > 0

@pytest.mark.asyncio(loop_scope="session")
async def test_execute_action_turn_on(ha, ollama):
    steps = [{"domain": "light", "service": "turn_on",
              "entity_id": "light.kitchen_ceiling"}]
    result = await execute(intent="action", steps=steps, ha=ha, ollama=ollama,
                           ok_response="The kitchen ceiling light is now on.",
                           fail_response="Sorry, I couldn't reach the kitchen ceiling light.")
    assert isinstance(result, str) and len(result) > 0

@pytest.mark.asyncio(loop_scope="session")
async def test_execute_multi_step(ha, ollama, ha_context):
    lights = [e for e in ha_context["entities"] if e["entity_id"].startswith("light.")][:2]
    steps = [{"domain": "light", "service": "turn_off", "entity_id": l["entity_id"]}
             for l in lights]
    result = await execute(intent="action", steps=steps, ha=ha, ollama=ollama,
                           ok_response="The lights are off.",
                           fail_response="Couldn't turn off the lights.")
    assert isinstance(result, str) and len(result) > 0
