from pipeline.agents.planner import _validate_steps


def test_comma_joined_entity_id_split_and_validated():
    """A model that emits entity_id as one comma-joined string must not be dropped —
    split into a list so each id is validated and kept."""
    entities = [
        {"entity_id": "light.a", "name": "A", "state": "on"},
        {"entity_id": "light.b", "name": "B", "state": "on"},
    ]
    steps = [{"domain": "light", "service": "turn_off", "entity_id": "light.a, light.b"}]
    out = _validate_steps(steps, entities, [])
    assert len(out) == 1
    assert sorted(out[0]["entity_id"]) == ["light.a", "light.b"]


def test_single_valid_entity_id_unchanged():
    entities = [{"entity_id": "light.a", "name": "A", "state": "on"}]
    steps = [{"domain": "light", "service": "turn_on", "entity_id": "light.a"}]
    out = _validate_steps(steps, entities, [])
    assert out[0]["entity_id"] == "light.a"
