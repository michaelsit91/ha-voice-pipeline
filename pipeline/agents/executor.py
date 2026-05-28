import asyncio, json, logging, os
from pipeline.ha_client import HAClient
from pipeline.ollama_client import OllamaClient
from pipeline.music_assistant_client import MusicAssistantClient
from pipeline.spotify_connect_sync import SpotifyConnectSync, extract_spotify_track_id

log = logging.getLogger("pipeline")

_ZIGBEE_SETTLE_S = float(os.getenv("ZIGBEE_PROPAGATION_MS", "400")) / 1000

_PARTIAL_SYSTEM = (
    "You are a smart home voice assistant. "
    "Compose ONE sentence of plain spoken English reporting what happened. "
    "No markdown, no emojis, no lists."
)


# Step keys that are routing metadata, not HA service data.
_STEP_META_KEYS = frozenset({"domain", "service", "entity_id", "area_id",
                              "query", "artist", "media_type"})


async def _run_step(step: dict, ha: HAClient) -> dict:
    """Execute one step and return a result dict with outcome."""
    domain    = step["domain"]
    service   = step["service"]
    entity_id = step.get("entity_id")
    area_id   = step.get("area_id")
    # Extra keys (e.g. volume_level, brightness_pct) are forwarded to HA as service data.
    extra     = {k: v for k, v in step.items() if k not in _STEP_META_KEYS}

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
    log.info("EXEC | %s.%s entity=%s area=%s extra=%s", domain, service, entity_id, area_id, extra)
    try:
        await ha.call_service(domain, service, entity_id=entity_id, area_id=area_id, **extra)
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


async def _run_music_step(
    step: dict,
    ha: HAClient,
    ma: MusicAssistantClient,
    spotify_sync: SpotifyConnectSync | None = None,
) -> str:
    """Execute one music_assistant.play_media step: search then play."""
    query      = step.get("query", "")
    artist     = step.get("artist") or None
    media_type = step.get("media_type", "track")
    entity_id  = step.get("entity_id")

    if not query:
        return "Sorry, I didn't catch what you wanted to play."

    try:
        results = await ma.search(query, media_type=media_type, artist=artist)
    except Exception as e:
        log.warning("MUSIC | search error: %s", e)
        return "Sorry, Music Assistant isn't responding right now."

    if not results:
        return f"Sorry, I couldn't find {query}."

    best        = results[0]
    uri         = best["uri"]
    track_name  = best["name"]
    artist_name = best["artist"]

    try:
        await ha.call_service(
            "music_assistant", "play_media",
            entity_id=entity_id,
            media_id=uri,
            media_type=media_type,
        )
    except Exception as e:
        log.warning("MUSIC | play_media error: %s", e)
        return "Sorry, I couldn't play that right now."

    # Fire-and-forget: sync to Spotify Connect so the phone shows what's playing
    if spotify_sync is not None and media_type == "track":
        track_id = extract_spotify_track_id(uri)
        if track_id:
            asyncio.create_task(spotify_sync.schedule_sync(track_id))
            log.debug("SPOTIFY_SYNC | scheduled sync for track %s", track_id)

    if artist_name:
        return f"Playing {track_name} by {artist_name}."
    return f"Playing {track_name}."


async def execute(
    intent: str,
    steps: list[dict],
    ha: HAClient,
    ollama: OllamaClient,
    ok_response: str = "",
    already_response: str = "",
    fail_response: str = "",
    ma: MusicAssistantClient | None = None,
    spotify_sync: SpotifyConnectSync | None = None,
) -> str:
    # Music steps are handled separately — branch before HA execution
    music_steps = [s for s in steps if s.get("domain") == "music_assistant"]
    if music_steps:
        if ma is None:
            return "Sorry, Music Assistant is not configured."
        return await _run_music_step(music_steps[0], ha, ma, spotify_sync)

    # Run all HA steps in parallel
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
