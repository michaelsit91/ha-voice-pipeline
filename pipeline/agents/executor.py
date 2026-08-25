import asyncio, json, logging, os, random
from pipeline.ha_client import HAClient
from pipeline.ollama_client import OllamaClient
from pipeline.music_assistant_client import MusicAssistantClient

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

# Services the planner is permitted to emit. The LLM chooses domain+service, so
# this is the safety boundary preventing a prompt-injected transcript from
# reaching a destructive service. music_assistant is handled before _run_step.
_ALLOWED_SERVICES: dict[str, frozenset[str]] = {
    "light":         frozenset({"turn_on", "turn_off", "toggle", "get_state"}),
    "switch":        frozenset({"turn_on", "turn_off", "toggle", "get_state"}),
    "fan":           frozenset({"turn_on", "turn_off", "toggle", "get_state"}),
    "input_boolean": frozenset({"turn_on", "turn_off", "toggle", "get_state"}),
    "cover":         frozenset({"open_cover", "close_cover", "stop_cover", "toggle", "get_state"}),
    "climate":       frozenset({"set_temperature", "set_hvac_mode", "turn_on", "turn_off", "get_state"}),
    "media_player":  frozenset({"media_play", "media_pause", "media_stop",
                                "volume_up", "volume_down", "volume_set", "get_state"}),
}


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

    if service not in _ALLOWED_SERVICES.get(domain, frozenset()):
        log.warning("EXEC | refused disallowed service %s.%s", domain, service)
        return {"entity_id": entity_id or area_id, "outcome": "failed",
                "error": f"service {domain}.{service} not allowed"}

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

    # Readback (state diff for "already" detection) is gated on the settle delay.
    # ZIGBEE_PROPAGATION_MS=0 → skip both the pre-read and the post-read entirely
    # for lowest action latency; the optimistic ok_response is spoken instead.
    readback = _ZIGBEE_SETTLE_S > 0

    # Capture state before execution (string or list entity_id; area_id excluded —
    # HA exposes no per-area state endpoint).
    states_before: dict[str, str] = {}
    if readback and isinstance(entity_id, str) and entity_id:
        try:
            states_before[entity_id] = (await ha.get_state(entity_id))["state"]
        except Exception:
            pass
    elif readback and isinstance(entity_id, list) and entity_id:
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

    # Fast mode: readback disabled — return optimistic success without re-reading
    if not readback:
        return {"entity_id": entity_id, "outcome": "success"}

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


async def _resolve_media(
    query: str,
    artist: str | None,
    media_type: str,
    ma: MusicAssistantClient,
    spotify_search=None,
) -> dict | None:
    """Resolve a heard query to {uri, name, artist} for MA play_media.

    Chain: wide MA search re-ranked with rapidfuzz (best acoustic match wins
    even on misheard STT) → raw MA search (last resort).
    """
    if spotify_search is not None:
        try:
            sp = await spotify_search.search(query, media_type=media_type, artist=artist)
        except Exception as e:
            log.warning("MUSIC | fuzzy search unavailable (%s) — raw MA fallback", e)
            sp = []
        if sp:
            best = sp[0]
            log.info("MUSIC | fuzzy match %r by %r (score %.0f)",
                     best["name"], best["artist"], best["score"])
            return {"uri": best["uri"], "name": best["name"], "artist": best["artist"]}

    try:
        results = await ma.search(query, media_type=media_type, artist=artist)
    except Exception as e:
        log.warning("MUSIC | MA search error: %s", e)
        return None
    return results[0] if results else None


async def _run_music_step(
    step: dict,
    ha: HAClient,
    ma: MusicAssistantClient,
    spotify_search=None,
) -> str:
    """Execute one music_assistant.play_media step: resolve then play."""
    query      = step.get("query", "")
    artist     = step.get("artist") or None
    media_type = step.get("media_type", "track")
    entity_id  = step.get("entity_id")

    if not query:
        return "Sorry, I didn't catch what you wanted to play."
    if not entity_id:
        return "Sorry, I can't find a speaker to play that on."

    best = await _resolve_media(query, artist, media_type, ma, spotify_search)
    if best is None:
        return f"Sorry, I couldn't find {query}."

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
        log.warning("MUSIC | play_media error for %r: %s — MA raw-query fallback", uri, e)
        # Direct spotify:// URI rejected (provider quirk) — resolve via MA search
        try:
            results = await ma.search(track_name or query, media_type=media_type,
                                      artist=artist_name or artist)
            if not results:
                return f"Sorry, I couldn't find {query}."
            uri, track_name, artist_name = (results[0]["uri"], results[0]["name"],
                                            results[0]["artist"])
            await ha.call_service(
                "music_assistant", "play_media",
                entity_id=entity_id,
                media_id=uri,
                media_type=media_type,
            )
        except Exception as e2:
            log.warning("MUSIC | play_media fallback error: %s", e2)
            return "Sorry, I couldn't play that right now."

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
    spotify_search=None,
    entity_names: dict[str, str] | None = None,
) -> str:
    # Music steps are handled separately — branch before HA execution
    music_steps = [s for s in steps if s.get("domain") == "music_assistant"]
    if music_steps:
        if ma is None:
            return "Sorry, Music Assistant is not configured."
        return await _run_music_step(music_steps[0], ha, ma, spotify_search)

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
        # One confirmation register: simple single-step action successes get the
        # same terse ack as the fast path (pre-cached TTS). Multi-step successes
        # keep the informative sentence — the words carry information there.
        if intent == "action" and n_total == 1:
            return "Done."
        return _vary(ok_response) if ok_response else "Done."

    if n_fail == n_total:
        return fail_response or "Sorry, I couldn't complete that."

    # Partial failure — one micro-LLM call for an accurate sentence.
    # Speak friendly names, never raw entity_ids ("living room 3 gang 1 left 3").
    names = entity_names or {}

    def _spoken(r: dict) -> str:
        eid = r.get("entity_id", "")
        if isinstance(eid, list):
            return ", ".join(names.get(e, str(e)) for e in eid)
        return names.get(eid, str(eid))

    succeeded_ids = [_spoken(r) for r in results if r["outcome"] != "failed"]
    failed_ids    = [_spoken(r) for r in failed]
    user = (
        f"Partial result:\n"
        f"Succeeded: {', '.join(succeeded_ids)}\n"
        f"Failed: {', '.join(failed_ids)}\n"
        f"Base response: {ok_response}"
    )
    try:
        return await ollama.chat(system=_PARTIAL_SYSTEM, user=user)
    except Exception as e:
        log.warning("EXEC | partial-failure LLM call failed: %s", e)
        return f"Done, but {n_fail} device(s) didn't respond."
