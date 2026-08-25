from pipeline.agents.planner import _build_context


def test_context_omits_state_column():
    entities = [{"entity_id": "light.a", "name": "A", "state": "on"}]
    areas = [{"area_id": "kitchen", "name": "Kitchen", "entities": ["light.a"]}]
    ctx = _build_context(entities, areas)
    assert "entity_id,name,area_id\n" in ctx
    assert "light.a,A" in ctx
    # the live state must NOT leak into the planner context
    assert "light.a,A,on" not in ctx
    assert ctx.count(",on") == 0


def test_context_binds_device_to_its_real_area():
    """Ids lie about rooms: the kitchen's light can carry a living_room_* id and a
    kitchen_* id can be the washing machine light. The area column is the truth."""
    entities = [
        {"entity_id": "light.living_room_3_gang_1_center", "name": "Kitchen Light 1", "state": "off"},
        {"entity_id": "light.kitchen_2_gang_left", "name": "Washing Machine Light", "state": "off"},
    ]
    areas = [
        {"area_id": "kitchen", "name": "Kitchen",
         "entities": ["light.living_room_3_gang_1_center", "light.kitchen_2_gang_left"]},
        {"area_id": "living_room", "name": "Living Room", "entities": []},
    ]
    ctx = _build_context(entities, areas)
    assert "light.living_room_3_gang_1_center,Kitchen Light 1,kitchen" in ctx
    assert "light.kitchen_2_gang_left,Washing Machine Light,kitchen" in ctx


def test_context_tolerates_areas_without_membership():
    """Older cached area payloads carry no entities key — the column stays empty
    rather than raising."""
    entities = [{"entity_id": "light.a", "name": "A", "state": "on"}]
    areas = [{"area_id": "kitchen", "name": "Kitchen"}]
    ctx = _build_context(entities, areas)
    assert "light.a,A," in ctx
