import asyncio, logging, re
from pipeline.ha_client import HAClient
from pipeline.ollama_client import OllamaClient
from pipeline.agents.planner import plan, _HESITATION_PATTERNS
from pipeline.agents.executor import execute
from pipeline.spotify_connect_sync import SpotifyConnectSync

log = logging.getLogger("pipeline")

_STOP_WORDS = {"is","the","a","an","all","on","off","of","and","or","to","in",
               "are","was","it","be","turn","what","how","does","do"}

_MUSIC_CONTROL_SERVICES = frozenset({"media_stop", "media_pause"})

_CJK_RE = re.compile(r'[一-鿿㐀-䶿＀-￯]')

def _is_cjk(text: str) -> bool:
    """True if transcript contains ≥2 CJK characters (Chinese/Japanese/Korean)."""
    return len(_CJK_RE.findall(text)) >= 2

def _filter_entities(entities: list[dict], transcript: str) -> list[dict]:
    """Return entities most relevant to this transcript using keyword scoring.
    For CJK transcripts, bypasses English keyword filter and returns all entities —
    the model can map e.g. '办公室' → 'Office Light' from the full list context."""
    if _is_cjk(transcript):
        return entities  # full list: model handles cross-language entity mapping

    words = {w.lower() for w in re.split(r'\W+', transcript) if len(w) > 2} - _STOP_WORDS
    if not words:
        return entities
    scored = sorted(
        [(e, sum(1 for w in words if w in e["name"].lower())) for e in entities],
        key=lambda x: -x[1],
    )
    strong = [e for e, s in scored if s >= 2]
    if strong:
        return strong  # 2+ keyword matches — precise set, use only these
    # No strong matches: include entities with at least 1 keyword match
    partial = [e for e, s in scored if s >= 1]
    if partial:
        return partial[:15]
    # Nothing matches (gibberish input): return top 15 by score as fallback
    return [e for e, _ in scored[:15]]

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
        return f"The {name} is {state}."
    on  = [n for n, s in valid if s == "on"]
    off = [n for n, s in valid if s != "on"]
    if not on:
        return "All of those are off."
    if not off:
        return "All of those are on."
    return f"{len(on)} on and {len(off)} off."

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
    )
