import asyncio, logging, os, re
from pipeline.ha_client import HAClient
from pipeline.ollama_client import OllamaClient
from pipeline.agents.planner import plan, _HESITATION_PATTERNS
from pipeline.agents.executor import execute, _run_volume_step
from pipeline.agents.fast_intent import match_fast_intent
from pipeline.spotify_connect_sync import SpotifyConnectSync

log = logging.getLogger("pipeline")

# ── Alexa-like deterministic fast path ────────────────────────────────────────
_FAST_PATH_ENABLED = os.getenv("FAST_PATH_ENABLED", "true").lower() in ("true", "1", "yes")
_FAST_MIN_SCORE = int(os.getenv("FAST_PATH_MIN_SCORE", "82"))
_FAST_MARGIN = int(os.getenv("FAST_PATH_MARGIN", "8"))

_STOP_WORDS = {"is","the","a","an","all","on","off","of","and","or","to","in",
               "are","was","it","be","turn","what","how","does","do"}

_MUSIC_CONTROL_SERVICES = frozenset({"media_stop", "media_pause",
                                      "volume_up", "volume_down", "volume_set"})

_CJK_RE = re.compile(r'[一-鿿㐀-䶿＀-￯]')

def _is_cjk(text: str) -> bool:
    """True if transcript contains ≥2 CJK characters (Chinese/Japanese/Korean)."""
    return len(_CJK_RE.findall(text)) >= 2

# Max candidate entities passed to the planner (stays under _build_context's 30 cap).
_FILTER_MAX = 25


def _fuzzy_entity_candidates(
    transcript: str, entities: list[dict], exclude: set[str], threshold: int = 72, limit: int = 8
) -> list[dict]:
    """Entities whose name fuzzily matches the transcript (catches misheard or
    differently-phrased device names that the substring keyword filter misses).
    No-op when rapidfuzz is unavailable."""
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return []
    t = transcript.lower()
    scored = [
        (fuzz.token_set_ratio(t, e["name"].lower()), e)
        for e in entities if e["entity_id"] not in exclude
    ]
    scored = [(s, e) for s, e in scored if s >= threshold]
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:limit]]


def _filter_entities(entities: list[dict], transcript: str) -> list[dict]:
    """Return entities most relevant to this transcript.

    Recall-first: keep ALL keyword matches (strong and partial) plus close fuzzy
    matches the keyword filter missed, so the intended device is never dropped
    before the planner sees it. For CJK transcripts, returns the full list (the
    model handles cross-language mapping e.g. '办公室' → 'Office Light')."""
    if _is_cjk(transcript):
        return entities

    words = {w.lower() for w in re.split(r'\W+', transcript) if len(w) > 2} - _STOP_WORDS
    if not words:
        return entities
    scored = sorted(
        ((e, sum(1 for w in words if w in e["name"].lower())) for e in entities),
        key=lambda x: -x[1],
    )
    matched = [e for e, s in scored if s >= 1]
    matched_ids = {e["entity_id"] for e in matched}
    fuzzy = _fuzzy_entity_candidates(transcript, entities, exclude=matched_ids)
    if matched:
        return (matched + fuzzy)[:_FILTER_MAX]
    # Nothing keyword-matched — surface fuzzy candidates, else the full list
    # (likely a broadcast command like "everything off" or gibberish).
    return fuzzy or entities

async def _query_fast_path(steps: list[dict], entities: list[dict], ha: HAClient) -> str | None:
    """Skip executor LLM for state queries — saves one Ollama round-trip.
    Handles both single and multi-entity queries."""
    if not steps or not all(s.get("service") == "get_state" for s in steps):
        return None
    entity_map = {e["entity_id"]: e["name"] for e in entities}
    # Flatten steps: expand list entity_ids, reject area_id (no HA area state query)
    entity_ids: list[str] = []
    for s in steps:
        eid = s.get("entity_id")
        if isinstance(eid, list):
            entity_ids.extend(eid)
        elif isinstance(eid, str):
            entity_ids.append(eid)
        elif s.get("area_id"):
            # Resolve area_id to matching entities via keyword match on name
            # (HA has no get_state for areas; planner sometimes uses area_id for queries)
            area_kw = s["area_id"].replace("_", " ").lower()
            matched = [e["entity_id"] for e in entities if area_kw in e["name"].lower()]
            if matched:
                entity_ids.extend(matched)
            else:
                return None  # can't resolve, fall through to executor
    if not entity_ids:
        return None
    try:
        states = await asyncio.gather(
            *[ha.get_state(eid) for eid in entity_ids], return_exceptions=True
        )
    except Exception:
        return None
    valid = [(entity_map.get(st["entity_id"], st["entity_id"]), st["state"])
             for st in states if not isinstance(st, Exception)]
    if not valid:
        return None
    if len(valid) == 1:
        name, state = valid[0]
        return _speak_state(name, state)
    on  = [n for n, s in valid if s == "on"]
    off = [n for n, s in valid if s != "on"]
    if not on:
        return "All of those are off."
    if not off:
        return "All of those are on."
    return f"{len(on)} on and {len(off)} off."

def _speak_state(name: str, state: str) -> str:
    """Spoken form of a device state — never read raw HA states aloud."""
    if state in ("unavailable", "unknown"):
        return f"The {name} isn't responding."
    return f"The {name} is {state}."


async def _execute_fast(fast: dict, ha: HAClient, ma, satellite: str | None) -> str | None:
    """Execute a fast-path plan optimistically (no readback) and return the spoken
    response, or None to fall back to the LLM planner (e.g. unresolved MA player)."""
    if fast["kind"] == "query":
        try:
            st = await ha.get_state(fast["entity_id"])
        except Exception:
            return None
        return _speak_state(fast["name"], st["state"])

    domain, service = fast["domain"], fast["service"]
    entity_id = fast.get("entity_id")
    area_id = fast.get("area_id")
    if fast.get("needs_player"):
        entity_id = ma.resolve_player(satellite) if ma is not None else None
        if not entity_id:
            return None  # no MA player resolved — let the LLM path route music
    extra = {k: fast[k] for k in ("volume_level",) if k in fast}
    try:
        if service in ("volume_up", "volume_down"):
            await _run_volume_step(ha, entity_id, service)
        else:
            await ha.call_service(domain, service, entity_id=entity_id, area_id=area_id, **extra)
    except Exception as e:
        log.warning("FAST | execution failed: %s", e)
        return "Sorry, that didn't work."
    return fast["ack"]


async def run_pipeline(
    transcript: str,
    ha: HAClient,
    ollama: OllamaClient,
    ma=None,
    satellite: str | None = None,
    spotify_sync: SpotifyConnectSync | None = None,
) -> str:
    # Hesitation/cancellation check — zero latency, no HA or Ollama calls needed
    if _HESITATION_PATTERNS.search(transcript):
        log.info("PLAN | intent=hesitation  transcript=%r", transcript)
        return "OK."

    entities, areas = await asyncio.gather(ha.get_entities(), ha.get_areas())

    # Alexa-like fast path: confident common commands skip the LLM entirely.
    if _FAST_PATH_ENABLED:
        fast = match_fast_intent(transcript, entities, areas, _FAST_MIN_SCORE, _FAST_MARGIN)
        if fast is not None:
            resp = await _execute_fast(fast, ha, ma, satellite)
            if resp is not None:
                log.info("FAST | %s -> %r", fast.get("service", fast["kind"]), resp)
                return resp

    filtered_entities = _filter_entities(entities, transcript)
    planned = await plan(transcript, filtered_entities, areas, ollama)
    log.info("PLAN | intent=%s corrected=%r steps=%s",
             planned.get("intent"), planned.get("corrected"), planned.get("steps"))
    if not planned.get("steps"):
        return "Sorry, I didn't understand that command."

    # Inject satellite's resolved MA player into music steps (overrides LLM's choice)
    if ma is not None:
        ma_player = ma.resolve_player(satellite)
        if ma_player:
            for step in planned["steps"]:
                if step.get("domain") == "music_assistant":
                    step["entity_id"] = ma_player
                elif (step.get("domain") == "media_player"
                      and step.get("service") in _MUSIC_CONTROL_SERVICES):
                    step["entity_id"] = ma_player

    # Fast path: single-step queries skip the executor LLM entirely
    if planned.get("intent") == "query":
        fast = await _query_fast_path(planned["steps"], entities, ha)
        if fast:
            return fast

    return await execute(
        intent=planned["intent"],
        steps=planned["steps"],
        ha=ha,
        ollama=ollama,
        ok_response=planned.get("ok_response", ""),
        already_response=planned.get("already_response", ""),
        fail_response=planned.get("fail_response", ""),
        ma=ma,
        spotify_sync=spotify_sync,
        entity_names={e["entity_id"]: e["name"] for e in entities},
    )
