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


def test_play_song_is_music_fast_intent():
    m = match_fast_intent("play despacito", E, AREAS)
    assert m is not None and m["kind"] == "music"
    assert m["query"] == "despacito"


def test_dim_not_handled_lights_are_onoff():
    # These lights have no brightness; dimming is not a fast-path intent.
    assert match_fast_intent("dim the kitchen light to 50%", E, AREAS) is None


def test_media_control_declines_when_buried_in_speech():
    """Media phrases name no device, so the fast path would answer TV dialogue
    without the planner's ignore classifier ever seeing it. Past the surrounding
    word allowance the match is declined so the planner arbitrates."""
    for line in ("can you turn it up i can't hear it",
                 "turn it down honey the baby's sleeping",
                 "pause for a second while i get the door",
                 "just stop the music already"):
        assert match_fast_intent(line, E, AREAS) is None, line


def test_media_control_still_fast_for_real_commands():
    """The allowance must not cost the fast path the commands it exists for."""
    for line, service in (("louder", "volume_up"),
                          ("volume up", "volume_up"),
                          ("quieter", "volume_down"),
                          ("turn the volume down", "volume_down"),
                          ("turn it up", "volume_up"),
                          ("pause", "media_pause"),
                          ("stop the music", "media_stop")):
        p = match_fast_intent(line, E, AREAS)
        assert p is not None and p["service"] == service, line
        assert p["needs_player"] is True


def test_volume_percent_declines_when_buried_in_speech():
    p = match_fast_intent("i was telling her to set volume to 40 percent yesterday", E, AREAS)
    assert p is None
    p = match_fast_intent("set volume to 40 percent", E, AREAS)
    assert p["service"] == "volume_set" and p["volume_level"] == 0.4


def test_query_reports_every_tied_device():
    """"the kitchen light" with two kitchen lights is plural, not ambiguous.
    Resolving the tie here keeps the choice away from the planner, which picks by
    entity_id substring and lands on the washing machine light 1 run in 6."""
    ents = [
        {"entity_id": "light.living_room_3_gang_1_center", "name": "Kitchen Light 1", "state": "off"},
        {"entity_id": "light.kitchen_2_gang_right_2", "name": "Kitchen Light 2", "state": "off"},
        {"entity_id": "light.kitchen_2_gang_left", "name": "Washing Machine Light", "state": "off"},
    ]
    p = match_fast_intent("is the kitchen light on", ents, AREAS)
    assert p["kind"] == "query"
    names = [t["name"] for t in p["targets"]]
    assert names == ["Kitchen Light 1", "Kitchen Light 2"]


def test_query_tie_does_not_sweep_in_a_different_device():
    p = match_fast_intent("is the desk lamp on", E, AREAS)
    assert [t["name"] for t in p["targets"]] == ["Desk Lamp"]
