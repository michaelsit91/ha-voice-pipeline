import json, logging, re
from pipeline.ollama_client import OllamaClient

log = logging.getLogger("pipeline")

# Words that signal the user is pausing, retracting, or didn't mean to issue a command.
# Detected before the LLM is called — zero latency, no Ollama round-trip.
_HESITATION_PATTERNS = re.compile(
    r"\b(hold on|hold up|wait|never mind|nevermind|forget it|forget that|"
    r"cancel|stop|abort|actually|scratch that|no wait|hang on)\b"
    # Chinese hesitation / cancellation words (no \b needed for CJK)
    r"|等一下|等等|算了|不对|取消|停一下|不用了|算了吧|不是这个",
    re.IGNORECASE,
)

_HESITATION_RESPONSE = {"corrected": "", "intent": "hesitation", "steps": [],
                        "ok_response": "OK.", "fail_response": ""}

# JSON schema enforced by Ollama — prevents malformed output entirely.
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "corrected":     {"type": "string"},
        "intent":        {"type": "string", "enum": ["action", "query"]},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "domain":    {"type": "string"},
                    "service":   {"type": "string"},
                    "entity_id": {"type": "string"},
                    "area_id":   {"type": "string"},
                },
                "required": ["domain", "service"],
            },
        },
        "ok_response":      {"type": "string"},
        "already_response": {"type": "string"},
        "fail_response":    {"type": "string"},
    },
    "required": ["corrected", "intent", "steps", "ok_response", "already_response", "fail_response"],
}

_SYSTEM = """\
You are a voice command parser for a smart home.
Given a voice transcript (which may contain STT errors) and a list of known devices, return a JSON object.

JSON shape — return ONLY this, no markdown, no explanation:
{
  "corrected": "<transcript with STT errors fixed>",
  "intent": "action" | "query",
  "steps": [
    {"domain": "<domain>", "service": "<service>", "entity_id": "<string or array>"}
  ],
  "ok_response":      "<1 sentence spoken confirmation assuming all steps succeed>",
  "already_response": "<1 sentence spoken if device is already in the requested state>",
  "fail_response":    "<1 sentence spoken apology assuming all steps fail>"
}

Rules:
- intent "query": user asks about current state (is X on? what is X set to?)
- intent "action": user wants to change something
- entity_id MUST be an exact entity_id from the device list — never invent one
- domain is the prefix before the dot in entity_id (e.g. entity_id "light.kitchen_1" → domain "light")
- Room/area is determined by entity_id (e.g. "fan.living_room_*" = living room, "light.kitchen_*" = kitchen). Friendly names may be mislabeled — always trust entity_id over name for room inference.
- For queries: use service "get_state"
- For actions: use the appropriate service (turn_on, turn_off, toggle, media_play, media_pause, etc.)

CRITICAL — multi-device consolidation:
- When the user targets devices in a named area ("living room lights", "all kitchen fans"):
  emit ONE step using "area_id" from the Areas table instead of "entity_id".
  Example: {"domain": "light", "service": "turn_off", "area_id": "living_room"}
- When targeting multiple specific devices not in a single area:
  emit ONE step with "entity_id" as a JSON array.
  Example: {"domain": "light", "service": "turn_off", "entity_id": ["light.desk", "light.lamp"]}
- NEVER emit more than one step per domain+service combination.
- For queries (get_state): ALWAYS use a single entity_id string. NEVER use area_id. NEVER use an array. One entity_id per step.

Response rules:
- ALWAYS write ok_response, already_response and fail_response in English, regardless of the transcript language.
- ok_response: short, natural, spoken. Name what changed. e.g. "The office light is now on."
- already_response: spoken sentence for when the device is already in the requested state. e.g. "The office light is already on."
- fail_response: honest, spoken apology. e.g. "Sorry, I couldn't reach the office light."
- For query intent: ok_response, already_response and fail_response may all be empty strings "".

- Order steps so they can execute independently (no dependencies between steps)
- WARNING: If you are unsure which entity_id to use, pick the closest matching one from the list. NEVER make up an entity_id or area_id.

--- EXAMPLES ---

Devices: light.office_light,Office Light,on | fan.living_room_fan,Living Room Fan,off
Areas: living_room,Living Room | office,Office

Transcript: turn on the office lite
{"corrected":"turn on the office light","intent":"action","steps":[{"domain":"light","service":"turn_on","entity_id":"light.office_light"}],"ok_response":"The office light is now on.","already_response":"The office light is already on.","fail_response":"Sorry, I couldn't turn on the office light."}

Transcript: 打开客厅风扇
{"corrected":"打开客厅风扇","intent":"action","steps":[{"domain":"fan","service":"turn_on","area_id":"living_room"}],"ok_response":"The living room fan is now on.","fail_response":"Sorry, I couldn't turn on the living room fan."}

Transcript: tun off the oface silin fan
{"corrected":"turn off the office ceiling fan","intent":"action","steps":[{"domain":"fan","service":"turn_off","entity_id":"fan.office_fan"}],"ok_response":"The office ceiling fan is now off.","fail_response":"Sorry, I couldn't turn off the office ceiling fan."}

Transcript: is the office light on
{"corrected":"is the office light on","intent":"query","steps":[{"domain":"light","service":"get_state","entity_id":"light.office_light"}],"ok_response":"","fail_response":""}

Transcript: turn off all living room lights and the office fan
{"corrected":"turn off all living room lights and the office fan","intent":"action","steps":[{"domain":"light","service":"turn_off","area_id":"living_room"},{"domain":"fan","service":"turn_off","entity_id":"fan.office_fan"}],"ok_response":"Living room lights and office fan are now off.","fail_response":"Sorry, I couldn't turn those off."}

Transcript: dim the kitchen light to fifty percent
{"corrected":"dim the kitchen light to 50%","intent":"action","steps":[{"domain":"light","service":"turn_on","entity_id":"light.kitchen_light","brightness_pct":50}],"ok_response":"Kitchen light dimmed to 50%.","fail_response":"Sorry, I couldn't dim the kitchen light."}
"""

_STOP_WORDS = {"the", "a", "an", "is", "on", "off", "all", "are", "in", "turn",
               "to", "and", "of", "my", "please", "can", "you", "me"}


_CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿]")


def _filter_entities(transcript: str, entities: list[dict], max_entities: int = 30) -> list[dict]:
    """Return entities most likely relevant to the transcript via keyword overlap.
    For CJK transcripts, keyword matching against English names is useless, so
    skip filtering and return all entities (capped at max_entities)."""
    if _CJK_RE.search(transcript):
        return entities[:max_entities]

    words = {w.lower() for w in transcript.split() if w.lower() not in _STOP_WORDS}
    if not words:
        return entities[:max_entities]

    def score(e: dict) -> int:
        target = (e["entity_id"] + " " + e.get("name", "")).lower()
        return sum(1 for w in words if w in target)

    scored = sorted(entities, key=score, reverse=True)
    matched = [e for e in scored if score(e) > 0]
    if len(matched) >= max_entities:
        return matched[:max_entities]
    extras = [e for e in scored if score(e) == 0]
    return (matched + extras)[:max_entities]


def _build_context(entities: list[dict], areas: list[dict], transcript: str = "") -> str:
    area_rows = ["area_id,name"] + [f"{a['area_id']},{a['name']}" for a in areas]
    filtered = _filter_entities(transcript, entities) if transcript else entities
    rows = ["entity_id,name,state"]
    for e in filtered:
        rows.append(f"{e['entity_id']},{e['name']},{e['state']}")
    return "Areas:\n" + "\n".join(area_rows) + "\n\nDevices:\n" + "\n".join(rows)


def _fuzzy_resolve(candidate: str, valid_ids: set[str]) -> str | None:
    """Return the best fuzzy match for candidate in valid_ids, or None if below threshold."""
    if not valid_ids:
        return None
    try:
        from rapidfuzz import process, fuzz
        match, score, _ = process.extractOne(
            candidate, valid_ids, scorer=fuzz.token_set_ratio
        )
        if score >= 80:
            return match
    except ImportError:
        pass
    return None


def _validate_steps(steps: list[dict], entities: list[dict], areas: list[dict]) -> list[dict]:
    """Post-validate entity_id / area_id in each step; fuzzy-fix or drop bad ones."""
    valid_entity_ids = {e["entity_id"] for e in entities}
    valid_area_ids   = {a["area_id"]   for a in areas}
    cleaned = []
    for step in steps:
        eid = step.get("entity_id")
        aid = step.get("area_id")

        # Unwrap nested {"entity_id": "..."} dicts that some models emit under schema mode
        if isinstance(eid, dict):
            eid = eid.get("entity_id") or next(iter(eid.values()), None)
            if eid is not None:
                step = {**step, "entity_id": eid}

        # Validate / fix entity_id
        if eid is not None:
            if isinstance(eid, list):
                fixed = [e if e in valid_entity_ids else _fuzzy_resolve(e, valid_entity_ids)
                         for e in eid]
                fixed = [e for e in fixed if e is not None]
                if not fixed:
                    log.warning("PLAN | dropped step — all entity_ids invalid: %s", eid)
                    continue
                step = {**step, "entity_id": fixed if len(fixed) > 1 else fixed[0]}
            elif isinstance(eid, str) and eid not in valid_entity_ids:
                resolved = _fuzzy_resolve(eid, valid_entity_ids)
                if resolved:
                    log.info("PLAN | fuzzy fix entity_id %r → %r", eid, resolved)
                    step = {**step, "entity_id": resolved}
                else:
                    log.warning("PLAN | dropped step — unknown entity_id %r", eid)
                    continue

        # Validate / fix area_id
        if aid is not None and aid not in valid_area_ids:
            resolved = _fuzzy_resolve(aid, valid_area_ids)
            if resolved:
                log.info("PLAN | fuzzy fix area_id %r → %r", aid, resolved)
                step = {**step, "area_id": resolved}
            else:
                log.warning("PLAN | dropped step — unknown area_id %r", aid)
                continue

        cleaned.append(step)

    # Consolidate multiple steps with the same (domain, service) and only entity_ids
    # (no area_id) into a single step with entity_id as a list.
    merged: list[dict] = []
    seen: dict[tuple, int] = {}  # (domain, service) -> index in merged
    for step in cleaned:
        if step.get("area_id"):
            merged.append(step)
            continue
        key = (step.get("domain"), step.get("service"))
        eid = step.get("entity_id")
        if key in seen:
            existing = merged[seen[key]]
            existing_eid = existing.get("entity_id")
            if existing_eid is None:
                merged[seen[key]] = {**existing, "entity_id": eid}
            elif isinstance(existing_eid, list):
                merged[seen[key]] = {**existing, "entity_id": existing_eid + [eid]}
            else:
                merged[seen[key]] = {**existing, "entity_id": [existing_eid, eid]}
        else:
            seen[key] = len(merged)
            merged.append(step)

    return merged


async def plan(
    transcript: str,
    entities: list[dict],
    areas: list[dict],
    ollama: OllamaClient,
) -> dict:
    if _HESITATION_PATTERNS.search(transcript):
        return _HESITATION_RESPONSE

    context = _build_context(entities, areas, transcript)
    user    = f"{context}\n\nTranscript: {transcript}"

    # Two attempts: first with JSON schema mode, retry on parse failure.
    for attempt in range(2):
        raw = await ollama.chat(system=_SYSTEM, user=user, format=_RESPONSE_SCHEMA)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            result = json.loads(raw)
            break
        except json.JSONDecodeError:
            if attempt == 0:
                log.warning("PLAN | JSONDecodeError on attempt 1, retrying: %r", raw[:120])
                # Second attempt: tighter user prompt
                user = (f"{context}\n\nTranscript: {transcript}\n\n"
                        "Reply with ONLY the JSON object, no other text.")
            else:
                log.error("PLAN | JSONDecodeError on attempt 2, giving up: %r", raw[:120])
                return {"corrected": transcript, "intent": "action", "steps": [],
                        "ok_response": "", "fail_response": "", "parse_error": raw}

    result.setdefault("corrected", transcript)
    result.setdefault("intent", "action")
    result.setdefault("steps", [])
    result.setdefault("ok_response", "")
    result.setdefault("already_response", "")
    result.setdefault("fail_response", "")

    # Post-validate entity/area ids — fuzzy-fix or drop bad ones
    result["steps"] = _validate_steps(result["steps"], entities, areas)

    return result
