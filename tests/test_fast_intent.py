from pipeline.agents.fast_intent import match_fast_intent

E = [
    {"entity_id": "light.kitchen", "name": "Kitchen Light", "state": "off"},
    {"entity_id": "fan.office", "name": "Office Fan", "state": "on"},
    {"entity_id": "light.desk_lamp", "name": "Desk Lamp", "state": "off"},
]
AREAS = [{"area_id": "kitchen", "name": "Kitchen"}]


def test_room_light_resolves_to_area_not_single_entity():
    # "<room> light" must control the whole area (HA expands area_id to every
    # light in the room) — not one guessed entity.
    p = match_fast_intent("turn off the kitchen light", E, AREAS)
    assert p["kind"] == "action" and p["domain"] == "light"
    assert p["service"] == "turn_off" and p["area_id"] == "kitchen"
    assert "entity_id" not in p


def test_specific_device_resolves_to_entity():
    p = match_fast_intent("turn on the desk lamp", E, AREAS)
    assert p["service"] == "turn_on" and p["entity_id"] == "light.desk_lamp"
    assert "area_id" not in p


def test_turn_off_named_entity():
    p = match_fast_intent("turn off the office fan", E, AREAS)
    assert p["service"] == "turn_off" and p["entity_id"] == "fan.office"


def test_toggle():
    p = match_fast_intent("toggle the desk lamp", E, AREAS)
    assert p["service"] == "toggle" and p["entity_id"] == "light.desk_lamp"


def test_volume_set_needs_player():
    p = match_fast_intent("set volume to 40 percent", E, AREAS)
    assert p["domain"] == "media_player" and p["service"] == "volume_set"
    assert p["volume_level"] == 0.4 and p["needs_player"] is True
    assert "entity_id" not in p


def test_media_stop_needs_player():
    p = match_fast_intent("stop the music", E, AREAS)
    assert p["service"] == "media_stop" and p["needs_player"] is True


def test_state_query():
    p = match_fast_intent("is the office fan on", E, AREAS)
    assert p["kind"] == "query" and p["entity_id"] == "fan.office"


def test_ambiguous_returns_none():
    ents = [
        {"entity_id": "light.one", "name": "Light One", "state": "off"},
        {"entity_id": "light.two", "name": "Light Two", "state": "off"},
    ]
    assert match_fast_intent("turn on the light", ents, []) is None


def test_unknown_device_returns_none():
    assert match_fast_intent("turn on the spaceship", E, AREAS) is None


def test_compound_command_returns_none():
    assert match_fast_intent("turn on the kitchen light and the office fan", E, AREAS) is None


def test_play_song_returns_none():
    assert match_fast_intent("play despacito", E, AREAS) is None


def test_dim_not_handled_lights_are_onoff():
    # These lights have no brightness; dimming is not a fast-path intent.
    assert match_fast_intent("dim the kitchen light to 50%", E, AREAS) is None
