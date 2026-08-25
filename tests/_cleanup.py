"""Live-test cleanup helpers."""


async def turn_off_all_switches(ha) -> None:
    """Turn off every switch-domain entity so e2e tests don't leave real devices on.

    Best-effort: swallows per-call errors and a missing/unreachable HA.
    """
    try:
        entities = await ha.get_entities()
    except Exception:
        return
    for e in entities:
        if e["entity_id"].startswith("switch."):
            try:
                await ha.call_service("switch", "turn_off", entity_id=e["entity_id"])
            except Exception:
                pass
