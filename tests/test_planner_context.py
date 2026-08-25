from pipeline.agents.planner import _build_context


def test_context_omits_state_column():
    entities = [{"entity_id": "light.a", "name": "A", "state": "on"}]
    areas = [{"area_id": "kitchen", "name": "Kitchen"}]
    ctx = _build_context(entities, areas)
    assert "entity_id,name\n" in ctx
    assert "light.a,A" in ctx
    # the live state must NOT leak into the planner context
    assert "light.a,A,on" not in ctx
    assert ctx.count(",on") == 0
