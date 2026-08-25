"""Spoken-response UX: informative failures, friendly names, no raw HA states."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pipeline import runner
from pipeline.agents import executor
from pipeline.agents.executor import execute


def _ha(entities=None, state="on"):
    ha = MagicMock()
    ha.get_entities = AsyncMock(return_value=entities or
        [{"entity_id": "light.kitchen", "name": "Kitchen Light", "state": "off"}])
    ha.get_areas = AsyncMock(return_value=[])
    ha.call_service = AsyncMock(return_value={})
    ha.get_state = AsyncMock(return_value={"entity_id": "light.kitchen", "state": state, "attributes": {}})
    return ha


@pytest.mark.asyncio
async def test_fast_path_failure_is_informative():
    ha = _ha()
    ha.call_service = AsyncMock(side_effect=Exception("boom"))
    ollama = MagicMock(); ollama.chat = AsyncMock()
    with patch.object(runner, "_FAST_PATH_ENABLED", True):
        resp = await runner.run_pipeline("turn on the kitchen light", ha, ollama)
    assert resp == "Sorry, that didn't work."
    ollama.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_unavailable_speaks_not_responding():
    ha = _ha(state="unavailable")
    ollama = MagicMock(); ollama.chat = AsyncMock()
    with patch.object(runner, "_FAST_PATH_ENABLED", True):
        resp = await runner.run_pipeline("is the kitchen light on", ha, ollama)
    assert "isn't responding" in resp
    assert "unavailable" not in resp


@pytest.mark.asyncio
async def test_partial_failure_prompt_uses_friendly_names():
    steps = [
        {"domain": "light", "service": "turn_on", "entity_id": "light.living_room_3_gang_1_left_3"},
        {"domain": "light", "service": "turn_on", "entity_id": "light.kitchen"},
    ]
    ollama = MagicMock()
    ollama.chat = AsyncMock(return_value="Partial done.")
    results = [
        {"entity_id": "light.living_room_3_gang_1_left_3", "outcome": "success"},
        {"entity_id": "light.kitchen", "outcome": "failed", "error": "timeout"},
    ]
    names = {"light.living_room_3_gang_1_left_3": "Living Room Middle Ceiling Light",
             "light.kitchen": "Kitchen Light"}
    with patch.object(executor, "_run_step", new=AsyncMock(side_effect=results)):
        await execute(intent="action", steps=steps, ha=MagicMock(), ollama=ollama,
                      ok_response="ok", entity_names=names)
    prompt = ollama.chat.await_args.kwargs["user"]
    assert "Living Room Middle Ceiling Light" in prompt
    assert "light.living_room_3_gang_1_left_3" not in prompt


# ── F2: one confirmation register — terse ack for simple LLM-path successes ──

def _step_results(results):
    return patch.object(executor, "_run_step", new=AsyncMock(side_effect=results))


@pytest.mark.asyncio
async def test_single_step_action_success_is_terse():
    """Simple single-step LLM-path success speaks the same terse register as the
    fast path ("Done.") instead of a varied full sentence."""
    steps = [{"domain": "light", "service": "turn_on", "entity_id": "light.a"}]
    with _step_results([{"entity_id": "light.a", "outcome": "success"}]):
        text = await execute(intent="action", steps=steps, ha=MagicMock(),
                             ollama=MagicMock(), ok_response="The office light is now on.")
    assert text == "Done."


@pytest.mark.asyncio
async def test_multi_step_success_keeps_informative_sentence():
    steps = [{"domain": "light", "service": "turn_off", "entity_id": "light.a"},
             {"domain": "fan", "service": "turn_off", "entity_id": "fan.b"}]
    with _step_results([{"entity_id": "light.a", "outcome": "success"},
                        {"entity_id": "fan.b", "outcome": "success"}]):
        text = await execute(intent="action", steps=steps, ha=MagicMock(),
                             ollama=MagicMock(), ok_response="Lights and fan are off.")
    assert "lights and fan are off" in text.lower()


@pytest.mark.asyncio
async def test_already_case_stays_informative():
    steps = [{"domain": "light", "service": "turn_on", "entity_id": "light.a"}]
    with _step_results([{"entity_id": "light.a", "outcome": "already"}]):
        text = await execute(intent="action", steps=steps, ha=MagicMock(), ollama=MagicMock(),
                             ok_response="The light is now on.",
                             already_response="The light is already on.")
    assert "already on" in text.lower()


@pytest.mark.asyncio
async def test_query_reaching_the_executor_speaks_state_not_done():
    """A question that falls past the query fast path must still be answered with
    the state. The action register's "Done." is not an answer to a question."""
    ha = _ha(state="on")
    resp = await execute(intent="query",
                         steps=[{"domain": "light", "service": "get_state",
                                 "entity_id": "light.kitchen"}],
                         ha=ha, ollama=MagicMock(),
                         entity_names={"light.kitchen": "Kitchen Light"})
    assert resp == "The Kitchen Light is on."


@pytest.mark.asyncio
async def test_query_intent_never_actuates():
    """A misclassified query carrying a write step must not toggle the device —
    asking whether a light is on cannot be allowed to turn it off."""
    ha = _ha()
    resp = await execute(intent="query",
                         steps=[{"domain": "light", "service": "toggle",
                                 "entity_id": "light.kitchen"}],
                         ha=ha, ollama=MagicMock())
    ha.call_service.assert_not_awaited()
    assert resp != "Done."
