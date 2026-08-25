from pipeline.runner import _filter_entities


def test_partial_match_not_dropped_when_strong_exists():
    """Recall: a 1-keyword-match device must NOT be excluded just because a
    2-keyword-match device exists — the planner needs to see both candidates."""
    entities = [
        {"entity_id": "light.office_desk", "name": "Office Desk Lamp", "state": "on"},
        {"entity_id": "light.office_ceiling", "name": "Office Light", "state": "on"},
    ]
    out = _filter_entities(entities, "turn on the office desk")
    ids = {e["entity_id"] for e in out}
    assert "light.office_desk" in ids
    assert "light.office_ceiling" in ids


def test_fuzzy_candidate_included_when_no_keyword_match():
    """Recall: a device whose name doesn't substring-match the transcript but is a
    close fuzzy match should still be surfaced to the planner."""
    entities = [
        {"entity_id": "light.desk_lamp", "name": "Desk Lamp", "state": "on"},
        {"entity_id": "light.bedside", "name": "Bedside Light", "state": "on"},
    ]
    # "bedside" substring-matches Bedside Light (keyword), Desk Lamp does not;
    # but a misspelling should still pull the right one via fuzzy.
    out = _filter_entities(entities, "turn off the bedsied lite")
    ids = {e["entity_id"] for e in out}
    assert "light.bedside" in ids


def test_no_match_returns_full_list():
    entities = [
        {"entity_id": "light.a", "name": "Kitchen", "state": "on"},
        {"entity_id": "light.b", "name": "Garage", "state": "on"},
    ]
    out = _filter_entities(entities, "play some jazz")
    assert len(out) == 2
