import asyncio, json, logging, os
from pipeline.ha_client import HAClient
from pipeline.ollama_client import OllamaClient

log = logging.getLogger("pipeline")

_ZIGBEE_SETTLE_S = float(os.getenv("ZIGBEE_PROPAGATION_MS", "400")) / 1000

_PARTIAL_SYSTEM = (
    "You are a smart home voice assistant. "
    "Compose ONE sentence of plain spoken English reporting what happened. "
    "No markdown, no emojis, no lists."
)


async def _run_step(step: dict, ha: HAClient) -> dict:
    """Execute one step and return a result dict with outcome."""
    domain    = step["domain"]
    service   = step["service"]
    entity_id = step.get("entity_id")
    area_id   = step.get("area_id")

    # For get_state queries, just read and return
    if service == "get_state" and isinstance(entity_id, str):
        try:
            state = await ha.get_state(entity_id)
            return {"entity_id": entity_id, "outcome": "queried", "state": state["state"]}
        except Exception as e:
            return {"entity_id": entity_id, "outcome": "failed", "error": str(e)}

    # Capture state before (only for single-entity — area/list can't be diffed)
    state_before = None
    if isinstance(entity_id, str) and entity_id:
        try:
            state_before = (await ha.get_state(entity_id))["state"]
        except Exception:
            pass

    # Execute
    log.info("EXEC | %s.%s entity=%s area=%s", domain, service, entity_id, area_id)
    try:
        await ha.call_service(domain, service, entity_id=entity_id, area_id=area_id)
    except Exception as e:
        log.warning("EXEC | FAILED %s.%s: %s", domain, service, e)
        return {"entity_id": entity_id or area_id, "outcome": "failed", "error": str(e)}

    # Wait for Zigbee/Z-Wave propagation, then read back state
    if isinstance(entity_id, str) and entity_id:
        await asyncio.sleep(_ZIGBEE_SETTLE_S)
        try:
            state_after = (await ha.get_state(entity_id))["state"]
            outcome = "success" if state_after != state_before else "already"
            return {"entity_id": entity_id, "outcome": outcome,
                    "state_before": state_before, "state_after": state_after}
        except Exception:
            pass

    # Area or list call — no per-entity state diff available
    return {"entity_id": entity_id or area_id, "outcome": "success"}


async def execute(
    intent: str,
    steps: list[dict],
    ha: HAClient,
    ollama: OllamaClient,
    ok_response: str = "",
    already_response: str = "",
    fail_response: str = "",
) -> str:
    # Run all steps in parallel
    results = await asyncio.gather(*[_run_step(s, ha) for s in steps])

    outcomes  = [r["outcome"] for r in results]
    failed    = [r for r in results if r["outcome"] == "failed"]
    n_fail    = len(failed)
    n_total   = len(results)

    log.info("EXEC | outcomes=%s", outcomes)

    if n_fail == 0:
        if all(r["outcome"] == "already" for r in results) and already_response:
            return already_response
        return ok_response or "Done."

    if n_fail == n_total:
        return fail_response or "Sorry, I couldn't complete that."

    # Partial failure — one micro-LLM call for an accurate sentence
    succeeded_ids = [str(r.get("entity_id", "")) for r in results if r["outcome"] != "failed"]
    failed_ids    = [str(r.get("entity_id", "")) for r in failed]
    user = (
        f"Partial result:\n"
        f"Succeeded: {', '.join(succeeded_ids)}\n"
        f"Failed: {', '.join(failed_ids)}\n"
        f"Base response: {ok_response}"
    )
    return await ollama.chat(system=_PARTIAL_SYSTEM, user=user)
