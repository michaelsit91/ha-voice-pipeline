import pytest
from pipeline.runner import run_pipeline

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
        assert isinstance(r, str) and len(r) > 0
    finally:
        for eid, state in initial.items():
            svc = "turn_on" if state == "on" else "turn_off"
            try:
                await ha.call_service("light", svc, entity_id=eid)
            except Exception:
                pass

async def test_multi_device_responds(ha, ollama):
    # Capture living room lights state so we can restore after
    entities = await ha.get_entities()
    lr_lights = [e["entity_id"] for e in entities
                 if e["entity_id"].startswith("light.") and "living" in e["entity_id"]]
    initial: dict[str, str] = {}
    for eid in lr_lights:
        try:
            initial[eid] = (await ha.get_state(eid))["state"]
        except Exception:
            pass
    try:
        r = await run_pipeline("turn off all living room lights", ha, ollama)
        assert isinstance(r, str) and len(r) > 0
    finally:
        for eid, state in initial.items():
            svc = "turn_on" if state == "on" else "turn_off"
            try:
                await ha.call_service("light", svc, entity_id=eid)
            except Exception:
                pass

async def test_nonsense_responds_gracefully(ha, ollama):
    r = await run_pipeline("xkqzfwm blarg", ha, ollama)
    assert isinstance(r, str) and len(r) > 0
