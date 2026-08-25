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


def test_hallucinated_id_cannot_cross_domains():
    """A switch.* id the model invented scores 86 against the light.* entity of the
    same name — above threshold — so only domain confinement stops it from firing
    the wrong device. The step is dropped instead."""
    entities = [{"entity_id": "light.living_room_ceiling", "name": "Living Room Ceiling",
                 "state": "on"}]
    steps = [{"domain": "switch", "service": "turn_off",
              "entity_id": "switch.living_room_ceiling"}]
    assert _validate_steps(steps, entities, []) == []


def test_zigbee_suffixed_entity_resolves_from_natural_guess():
    """The natural guess must resolve onto the hardware-suffixed Zigbee id."""
    entities = [{"entity_id": "fan.master_bedroom_2_gang_1_right", "name": "Master Bedroom Fan",
                 "state": "on"}]
    steps = [{"domain": "fan", "service": "turn_off", "entity_id": "fan.master_bedroom_fan"}]
    out = _validate_steps(steps, entities, [])
    assert out[0]["entity_id"] == "fan.master_bedroom_2_gang_1_right"


def test_domain_prefixed_area_id_still_resolves():
    """area_ids carry no domain, so a model-emitted "light.living_room" must match
    the "living_room" area rather than filter the candidate pool down to nothing."""
    areas = [{"area_id": "living_room", "name": "Living Room"},
             {"area_id": "office", "name": "Office"}]
    steps = [{"domain": "light", "service": "turn_off", "area_id": "light.living_room"}]
    out = _validate_steps(steps, [], areas)
    assert len(out) == 1
    assert out[0]["area_id"] == "living_room"
