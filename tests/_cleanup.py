"""Live-test cleanup helpers — snapshot device state before tests, restore after."""
import asyncio

# On/off-controllable domains we snapshot + restore around the test session.
# media_player/climate/cover state isn't cleanly restorable via on/off.
_RESTORE_DOMAINS = ("light", "switch", "fan", "input_boolean")

# Entities never touched by restore. atx_control is a momentary machine-power
# switch (toggling it has real side effects); it's surfaced on the HA dashboard
# for manual control instead.
_EXCLUDE = {"switch.atx_control_atx_power"}


def _domain(entity_id: str) -> str:
    return entity_id.split(".")[0]


async def snapshot_states(ha) -> dict[str, str]:
    """Capture the current on/off state of restorable entities. Best-effort."""
    try:
        entities = await ha.get_entities()
    except Exception:
        return {}
    return {e["entity_id"]: e["state"] for e in entities
            if _domain(e["entity_id"]) in _RESTORE_DOMAINS
            and e["entity_id"] not in _EXCLUDE}


async def restore_states(ha, snapshot: dict[str, str], settle_s: float = 2.0) -> None:
    """Revert only the devices the tests actually changed back to their original
    on/off state. Waits `settle_s` so the last test's Zigbee transitions have
    landed before reading current state (avoids restoring against a stale read),
    then touches only genuinely-drifted, non-excluded entities — so unrelated
    devices (water heaters, AC, etc.) are never commanded."""
    if not snapshot:
        return
    if settle_s:
        await asyncio.sleep(settle_s)
    if hasattr(ha, "clear_cache"):
        try:
            ha.clear_cache()
        except Exception:
            pass
    try:
        entities = await ha.get_entities()
    except Exception:
        return
    current = {e["entity_id"]: e["state"] for e in entities}
    for eid, want in snapshot.items():
        if eid in _EXCLUDE or current.get(eid, want) == want:
            continue
        svc = "turn_on" if want == "on" else "turn_off"
        try:
            await ha.call_service(_domain(eid), svc, entity_id=eid)
        except Exception:
            pass
