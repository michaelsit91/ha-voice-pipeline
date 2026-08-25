import asyncio
import json

import pytest

from pipeline.runner import run_pipeline

# Zigbee needs a moment to report the new state back to HA before we read it.
ZIGBEE_SETTLE = 1.5


async def _area_lights(ha, area_id: str) -> list[str]:
    """Lights that actually belong to an area. Entity ids are unreliable for room
    membership — some light.living_room_* devices are wired into other rooms — and
    whole-room commands execute by area_id, so the area is what must be asserted on."""
    r = await ha._get_client().post(
        f"{ha._url}/api/template",
        headers=ha._hdrs,
        json={"template": "{{ area_entities('%s') | select('match', 'light\\.') | list | tojson }}" % area_id},
        timeout=10,
    )
    r.raise_for_status()
    return json.loads(r.text)

async def test_status_query_mentions_state(ha, ollama):
    r = await run_pipeline("is the living room light on", ha, ollama)
    assert any(w in r.lower() for w in ("on", "off", "living room", "light"))

async def test_single_action_responds(ha, ollama):
    # Capture kitchen lights state so we can restore after
    entities = await ha.get_entities()
    kitchen_lights = [e["entity_id"] for e in entities
                      if e["entity_id"].startswith("light.") and "kitchen" in e["entity_id"]]
    initial: dict[str, str] = {}
    for eid in kitchen_lights:
        try:
            initial[eid] = (await ha.get_state(eid))["state"]
        except Exception:
            pass
    try:
        r = await run_pipeline("turn on the kitchen light", ha, ollama)
        # Empty response = silent success ack (satellite chime); state is the proof.
        assert isinstance(r, str)
        await asyncio.sleep(ZIGBEE_SETTLE)
        states = [(await ha.get_state(eid))["state"] for eid in kitchen_lights]
        assert "on" in states
    finally:
        for eid, state in initial.items():
            svc = "turn_on" if state == "on" else "turn_off"
            try:
                await ha.call_service("light", svc, entity_id=eid)
            except Exception:
                pass

async def test_multi_device_responds(ha, ollama):
    # Capture living room lights state so we can restore after
    lr_lights = await _area_lights(ha, "living_room")
    initial: dict[str, str] = {}
    for eid in lr_lights:
        try:
            initial[eid] = (await ha.get_state(eid))["state"]
        except Exception:
            pass
    try:
        r = await run_pipeline("turn off all living room lights", ha, ollama)
        # Empty response = silent success ack (satellite chime); state is the proof.
        assert isinstance(r, str)
        await asyncio.sleep(ZIGBEE_SETTLE)
        states = [(await ha.get_state(eid))["state"] for eid in lr_lights]
        assert set(states) == {"off"}
    finally:
        for eid, state in initial.items():
            svc = "turn_on" if state == "on" else "turn_off"
            try:
                await ha.call_service("light", svc, entity_id=eid)
            except Exception:
                pass

async def test_nonsense_responds_gracefully(ha, ollama):
    """Gibberish must degrade, never raise. Since intent=ignore landed, the planner
    may legitimately answer a false wake with silence instead of a spoken error, and
    which of the two it picks is not deterministic across runs — so this asserts the
    graceful contract only. The exact silence-vs-speech split is pinned on mocked
    planner output in test_tv_noise_ignore.py."""
    r = await run_pipeline("xkqzfwm blarg", ha, ollama)
    assert isinstance(r, str)
