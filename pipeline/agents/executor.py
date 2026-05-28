import asyncio, json, logging, os, random
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

# Natural-sounding acknowledgment prefixes added randomly to successful action
# responses so Piper doesn't repeat the exact same phrase every time.
# Empty string entries increase the chance of no prefix (keeps original phrasing).
_REPLY_PREFIXES = [
    "",          # no prefix — keeps original
    "",          # no prefix — keeps original (double-weighted for ~30% no-change rate)
    "Sure, ",
    "Done! ",
    "Got it, ",
    "OK, ",
    "Alright, ",
]


def _vary(text: str) -> str:
    """Randomly prepend one of several natural acknowledgment phrases.

    Applied only to action confirmations before Piper speaks them.
    Empty-prefix entries in _REPLY_PREFIXES mean ~30% of responses keep
    their original phrasing unchanged.
    """
    if not text:
        return text
    prefix = random.choice(_REPLY_PREFIXES)
    if not prefix:
        return text
    # Lowercase the first character of the original sentence so "Sure, The light
    # is on." doesn't happen — it becomes "Sure, the light is on."
    return prefix + text[0].lower() + text[1:]


# Step keys that are routing metadata, not HA service data.
_STEP_META_KEYS = frozenset({"domain", "service", "entity_id", "area_id",
                              "query", "artist", "media_type"})

# Volume step size for "louder" / "quieter" commands.
_VOLUME_STEP = 0.10


async def _run_volume_step(
    ha: HAClient,
    entity_id: str,
    direction: str,  # "volume_up" | "volume_down"
) -> dict:
    """Implement volume_up/down as a precise ±10% volume_set call.

    Reading the current volume from HA and computing the new level ourselves
    gives deterministic 10% steps regardless of the integration's own default.
    """
    try:
        state = await ha.get_state(entity_id)
        current: float = state["attributes"].get("volume_level", 0.5)
        delta = _VOLUME_STEP if direction == "volume_up" else -_VOLUME_STEP
        new_level = round(max(0.0, min(1.0, current + delta)), 2)
        await ha.call_service("media_player", "volume_set",
                              entity_id=entity_id, volume_level=new_level)
        log.info("EXEC | volume %s: %.0f%% → %.0f%%",
                 direction, current * 100, new_level * 100)
        return {"entity_id": entity_id, "outcome": "success",
                "state_before": str(current), "state_after": str(new_level)}
    except Exception as e:
        log.warning("EXEC | FAILED volume step: %s", e)
        return {"entity_id": entity_id, "outcome": "failed", "error": str(e)}


async def _run_step(step: dict, ha: HAClient) -> dict:
    """Execute one step and return a result dict with outcome."""
    domain    = step["domain"]
    service   = step["service"]
    entity_id = step.get("entity_id")
    area_id   = step.get("area_id")
    # Extra keys (e.g. volume_level, brightness_pct) are forwarded to HA as service data.
    extra     = {k: v for k, v in step.items() if k not in _STEP_META_KEYS}

    # volume_up/down → precise 10% steps via volume_set
    if service in ("volume_up", "volume_down") and isinstance(entity_id, str):
        return await _run_volume_step(ha, entity_id, service)

    # For get_state queries, just read and return
    if service == "get_state" and isinstance(entity_id, str):
        try:
            state = await ha.get_state(entity_id)
            return {"entity_id": entity_id, "outcome": "queried", "state": state["state"]}
        except Exception as e:
            return {"entity_id": entity_id, "outcome": "failed", "error": str(e)}

    # Capture state before execution (string or list entity_id; area_id excluded —
    # HA exposes no per-area state endpoint).
    states_before: dict[str, str] = {}
    if isinstance(entity_id, str) and entity_id:
        try:
            states_before[entity_id] = (await ha.get_state(entity_id))["state"]
        except Exception:
            pass
    elif isinstance(entity_id, list) and entity_id:
        try:
            pre = await asyncio.gather(
                *[ha.get_state(eid) for eid in entity_id], return_exceptions=True
            )
            for eid, r in zip(entity_id, pre):
                if not isinstance(r, Exception):
                    states_before[eid] = r["state"]
        except Exception:
            pass

    # Execute
    log.info("EXEC | %s.%s entity=%s area=%s extra=%s", domain, service, entity_id, area_id, extra)
    try:
        await ha.call_service(domain, service, entity_id=entity_id, area_id=area_id, **extra)
    except Exception as e:
        log.warning("EXEC | FAILED %s.%s: %s", domain, service, e)
        return {"entity_id": entity_id or area_id, "outcome": "failed", "error": str(e)}

    # Area calls: HA has no per-area state endpoint, nothing to diff
    if area_id:
        return {"entity_id": area_id, "outcome": "success"}

    # Determine which entity IDs to read back
    target_ids: list[str] = (
        [entity_id] if isinstance(entity_id, str) and entity_id
        else entity_id if isinstance(entity_id, list) and entity_id
        else []
    )
    if not target_ids:
        return {"entity_id": entity_id, "outcome": "success"}

    # Wait for Zigbee/Z-Wave propagation, then read back state
    await asyncio.sleep(_ZIGBEE_SETTLE_S)
    try:
        post = await asyncio.gather(
            *[ha.get_state(eid) for eid in target_ids], return_exceptions=True
        )
        states_after = {eid: r["state"] for eid, r in zip(target_ids, post)
                        if not isinstance(r, Exception)}
    except Exception:
        return {"entity_id": entity_id, "outcome": "success"}

    if not states_after:
        return {"entity_id": entity_id, "outcome": "success"}

    # "already" if every entity with a before/after reading was unchanged
    comparable = [eid for eid in target_ids
                  if eid in states_before and eid in states_after]
    all_unchanged = bool(comparable) and all(
        states_before[eid] == states_after[eid] for eid in comparable
    )
    outcome = "already" if all_unchanged else "success"

    if isinstance(entity_id, str):
        return {"entity_id": entity_id, "outcome": outcome,
                "state_before": states_before.get(entity_id),
                "state_after": states_after.get(entity_id)}
    return {"entity_id": entity_id, "outcome": outcome}


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
            return _vary(already_response)
        return _vary(ok_response) if ok_response else "Done."

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
