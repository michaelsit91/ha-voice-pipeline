import json, logging, re
from pipeline.ollama_client import OllamaClient

log = logging.getLogger("pipeline")

# Words that signal the user is pausing, retracting, or didn't mean to issue a command.
# Detected before the LLM is called — zero latency, no Ollama round-trip.
_HESITATION_PATTERNS = re.compile(
    r"\b(hold on|hold up|wait|never mind|nevermind|forget it|forget that|"
    r"cancel|stop(?!\s+(?:\w+\s+)*(?:music|playing|song|track))|abort|"
    r"actually|scratch that|no wait|hang on)\b"
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
        "intent":        {"type": "string", "enum": ["action", "query", "ignore"]},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "domain":        {"type": "string"},
                    "service":       {"type": "string"},
                    "entity_id":     {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                    "area_id":       {"type": "string"},
                    "query":         {"type": "string"},
                    "artist":        {"type": "string"},
                    "media_type":    {"type": "string"},
                    "volume_level":  {"type": "number"},
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
  "intent": "action" | "query" | "ignore",
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
- intent "ignore": the transcript is NOT addressed to a smart-home assistant — TV/movie dialogue,
  background conversation between people, song lyrics, or a meaningless fragment picked up by a
  false wake ("flable suit of jade", "I told you he'd be back tomorrow", "she never saw it coming").
  For "ignore": steps MUST be [] and all responses MUST be "". When the transcript plausibly
  commands or asks about a known device — even garbled — prefer "action"/"query" over "ignore".
  Brevity is NOT evidence of non-direction: a short phrase naming no device but asking for a
  change in the room ("make it dark", "louder", "turn it off", "too bright") is a command.
  Choose "ignore" only on positive evidence — the speech is ABOUT someone or something else
  (third-party narrative, past tense, dialogue, lyrics) rather than addressed to you.
- entity_id MUST be an exact entity_id from the device list — never invent one
- domain is the prefix before the dot in entity_id (e.g. entity_id "light.kitchen_1" → domain "light")
- Room membership comes from each device's area_id column, NOT from entity names/ids. Some lights are wired to switches in OTHER rooms, so an entity like "light.living_room_*" may actually live in the kitchen area — entity names are unreliable for room inference. For ANY whole-room command ("the kitchen light", "living room lights", "all office fans"), emit ONE step with "area_id" from the Areas table and NO entity_id; Home Assistant then controls every matching device in that area. NEVER guess a single entity_id for a room command. When a query needs ONE device in a room, pick it by its area_id column and name — an id containing "kitchen" may belong to another room, and the kitchen's light may have a living_room_* id.
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

MUSIC COMMANDS:
- When the user wants to play a song, artist, album, or playlist, emit ONE music_assistant.play_media step.
- entity_id: use the media_player entity with mass_player_type player from the Devices list.
- query: STT-corrected search term using your knowledge of music (e.g. 'blainding lites' -> 'Blinding Lights').
- media_type: 'track' for a specific song, 'artist' for 'play X' or 'something by X', 'album' for album, 'playlist' for playlist.
- artist: only include when the user explicitly names an artist alongside a title.
- ok_response: 'Playing {query}.' -- keep it short.

MUSIC CONTROL COMMANDS:
- "stop the music" / "stop playing" / "stop that song" → emit ONE step: domain=media_player, service=media_stop.
- "pause" / "pause the music" / "pause that" → emit ONE step: domain=media_player, service=media_pause.
- entity_id: use the media_player entity with mass_player_type player from the Devices list.
- ok_response: "Music stopped." for stop; "Paused." for pause.
- already_response: "" (no already-state applies for stop/pause).

VOLUME COMMANDS:
- "volume up" / "louder" / "turn it up" / "a bit louder" → emit ONE step: domain=media_player, service=volume_up.
- "volume down" / "quieter" / "turn it down" / "lower the volume" → emit ONE step: domain=media_player, service=volume_down.
- "set volume to X%" / "volume X percent" / "X percent volume" → emit ONE step: domain=media_player, service=volume_set, volume_level=X/100 (float 0.0–1.0).
- entity_id: use the media_player entity with mass_player_type player from the Devices list.
- ok_response: "Volume up." / "Volume down." / "Volume set to X%."
- already_response: "" (no already-state applies for volume commands).

--- EXAMPLES ---

Devices: light.office_light,Office Light | fan.living_room_fan,Living Room Fan | light.kitchen_ceiling,Kitchen Ceiling
Areas: living_room,Living Room | office,Office | kitchen,Kitchen

Transcript: turn on the office lite
{"corrected":"turn on the office light","intent":"action","steps":[{"domain":"light","service":"turn_on","area_id":"office"}],"ok_response":"The office light is now on.","already_response":"","fail_response":"Sorry, I couldn't turn on the office light."}

Transcript: turn off the kitchen light
{"corrected":"turn off the kitchen light","intent":"action","steps":[{"domain":"light","service":"turn_off","area_id":"kitchen"}],"ok_response":"The kitchen light is now off.","already_response":"","fail_response":"Sorry, I couldn't turn off the kitchen light."}

Transcript: 打开客厅风扇
{"corrected":"打开客厅风扇","intent":"action","steps":[{"domain":"fan","service":"turn_on","area_id":"living_room"}],"ok_response":"The living room fan is now on.","fail_response":"Sorry, I couldn't turn on the living room fan."}

Transcript: tun off the oface silin fan
{"corrected":"turn off the office ceiling fan","intent":"action","steps":[{"domain":"fan","service":"turn_off","entity_id":"fan.office_fan"}],"ok_response":"The office ceiling fan is now off.","fail_response":"Sorry, I couldn't turn off the office ceiling fan."}

Transcript: is the office light on
{"corrected":"is the office light on","intent":"query","steps":[{"domain":"light","service":"get_state","entity_id":"light.office_light"}],"ok_response":"","fail_response":""}

Transcript: flable suit of jade
{"corrected":"","intent":"ignore","steps":[],"ok_response":"","already_response":"","fail_response":""}

Transcript: honey I told you he was coming back tomorrow
{"corrected":"","intent":"ignore","steps":[],"ok_response":"","already_response":"","fail_response":""}

Transcript: make it dark
{"corrected":"make it dark","intent":"action","steps":[{"domain":"light","service":"turn_off","area_id":"living_room"}],"ok_response":"Lights off.","already_response":"","fail_response":"Sorry, I couldn't turn the lights off."}

Transcript: turn off all living room lights and the office fan
{"corrected":"turn off all living room lights and the office fan","intent":"action","steps":[{"domain":"light","service":"turn_off","area_id":"living_room"},{"domain":"fan","service":"turn_off","entity_id":"fan.office_fan"}],"ok_response":"Living room lights and office fan are now off.","fail_response":"Sorry, I couldn't turn those off."}

Transcript: turn off everything
{"corrected":"turn off everything","intent":"action","steps":[{"domain":"light","service":"turn_off","entity_id":"light.office_light"},{"domain":"fan","service":"turn_off","entity_id":"fan.living_room_fan"}],"ok_response":"Everything is off.","already_response":"","fail_response":"Sorry, I couldn't turn everything off."}

Devices: fan.living_room_fan,Living Room Fan | fan.master_bedroom_fan,Master Bedroom Fan | fan.office_fan,Office Fan | fan.guest_room_fan,Guest Room Fan
Areas: living_room,Living Room | master_bedroom,Master Bedroom | office,Office | guest_room,Guest Room

Transcript: toggle all the fans
{"corrected":"toggle all the fans","intent":"action","steps":[{"domain":"fan","service":"toggle","entity_id":["fan.living_room_fan","fan.master_bedroom_fan","fan.office_fan","fan.guest_room_fan"]}],"ok_response":"All fans toggled.","already_response":"","fail_response":"Sorry, I couldn't toggle the fans."}

Transcript: turn off all the fans
{"corrected":"turn off all the fans","intent":"action","steps":[{"domain":"fan","service":"turn_off","entity_id":["fan.living_room_fan","fan.master_bedroom_fan","fan.office_fan","fan.guest_room_fan"]}],"ok_response":"All fans are now off.","already_response":"All fans are already off.","fail_response":"Sorry, I couldn't turn off the fans."}

Devices: media_player.respeaker_lite_media_player_2,Spotify
Transcript: play blinding lights
{"corrected":"play Blinding Lights","intent":"action","steps":[{"domain":"music_assistant","service":"play_media","entity_id":"media_player.respeaker_lite_media_player_2","query":"Blinding Lights","media_type":"track"}],"ok_response":"Playing Blinding Lights.","already_response":"","fail_response":"Sorry, I couldn't play that."}

Transcript: play something by the weeknd
{"corrected":"play something by The Weeknd","intent":"action","steps":[{"domain":"music_assistant","service":"play_media","entity_id":"media_player.respeaker_lite_media_player_2","query":"The Weeknd","media_type":"artist"}],"ok_response":"Playing The Weeknd.","already_response":"","fail_response":"Sorry, I couldn't play that."}

Transcript: play hotel california by the eagles
{"corrected":"play Hotel California by the Eagles","intent":"action","steps":[{"domain":"music_assistant","service":"play_media","entity_id":"media_player.respeaker_lite_media_player_2","query":"Hotel California","artist":"Eagles","media_type":"track"}],"ok_response":"Playing Hotel California by the Eagles.","already_response":"","fail_response":"Sorry, I couldn't play that."}

Transcript: stop the music
{"corrected":"stop the music","intent":"action","steps":[{"domain":"media_player","service":"media_stop","entity_id":"media_player.respeaker_lite_media_player_2"}],"ok_response":"Music stopped.","already_response":"","fail_response":"Sorry, I couldn't stop the music."}

Transcript: pause
{"corrected":"pause","intent":"action","steps":[{"domain":"media_player","service":"media_pause","entity_id":"media_player.respeaker_lite_media_player_2"}],"ok_response":"Paused.","already_response":"","fail_response":"Sorry, I couldn't pause the music."}

Transcript: louder
{"corrected":"louder","intent":"action","steps":[{"domain":"media_player","service":"volume_up","entity_id":"media_player.respeaker_lite_media_player_2"}],"ok_response":"Volume up.","already_response":"","fail_response":"Sorry, I couldn't raise the volume."}

Transcript: turn the volume down
{"corrected":"turn the volume down","intent":"action","steps":[{"domain":"media_player","service":"volume_down","entity_id":"media_player.respeaker_lite_media_player_2"}],"ok_response":"Volume down.","already_response":"","fail_response":"Sorry, I couldn't lower the volume."}

Transcript: set volume to 40 percent
{"corrected":"set volume to 40%","intent":"action","steps":[{"domain":"media_player","service":"volume_set","entity_id":"media_player.respeaker_lite_media_player_2","volume_level":0.4}],"ok_response":"Volume set to 40%.","already_response":"","fail_response":"Sorry, I couldn't set the volume."}
"""

def _build_context(entities: list[dict], areas: list[dict]) -> str:
    """Build the prompt context from the (already-filtered) entity list.
    States are intentionally omitted: under the HA roster cache they can be stale
    and must not bias entity/intent selection — the executor's readback is the
    authority on current state. Caps at 30 entities for the context budget."""
    area_of = {eid: a["area_id"] for a in areas for eid in a.get("entities", ())}
    area_rows = ["area_id,name"] + [f"{a['area_id']},{a['name']}" for a in areas]
    rows = ["entity_id,name,area_id"]
    for e in entities[:30]:
        rows.append(f"{e['entity_id']},{e['name']},{area_of.get(e['entity_id'], '')}")
    return "Areas:\n" + "\n".join(area_rows) + "\n\nDevices:\n" + "\n".join(rows)


_FUZZY_THRESHOLD = 80


def _fuzzy_resolve(candidate: str, valid_ids: set[str]) -> str | None:
    """Return the best fuzzy match for candidate in valid_ids, or None if below threshold.

    Matching is confined to the candidate's own domain, so a hallucinated
    ``fan.*`` id can never resolve to a ``light.*`` entity and fire the wrong
    action. WRatio is used over token_set_ratio because Zigbee ids carry hardware
    suffixes ("fan.master_bedroom_2_gang_1_right") that token scorers penalise
    heavily — the natural guess "fan.master_bedroom_fan" scored 76 and was
    dropped, which cost a full thinking retry to re-derive the same answer.
    """
    if not valid_ids:
        return None
    # Confinement only applies to domain-qualified sets (entity_ids). area_ids
    # carry no domain, so a model-emitted "light.living_room" must still match
    # "living_room" rather than filter the pool down to nothing.
    domain = candidate.split(".")[0] if "." in candidate else ""
    if domain and any("." in v for v in valid_ids):
        pool = {e for e in valid_ids if e.split(".")[0] == domain}
        if not pool:
            return None
    else:
        pool = valid_ids
    try:
        from rapidfuzz import process, fuzz
        match, score, _ = process.extractOne(candidate, pool, scorer=fuzz.WRatio)
        if score >= _FUZZY_THRESHOLD:
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
        # Music Assistant steps: entity_id is injected by the runner — skip validation
        if step.get("domain") == "music_assistant":
            cleaned.append(step)
            continue
        eid = step.get("entity_id")
        aid = step.get("area_id")

        # Unwrap nested {"entity_id": "..."} dicts that some models emit under schema mode
        if isinstance(eid, dict):
            eid = eid.get("entity_id") or next(iter(eid.values()), None)
            if eid is not None:
                step = {**step, "entity_id": eid}

        # Split comma-joined entity_id strings some models emit ("light.a, light.b")
        # so each id is validated and kept instead of dropped as one unknown id.
        if isinstance(eid, str) and "," in eid:
            eid = [p.strip() for p in eid.split(",") if p.strip()]
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
        # Normalise eid to a flat list for consistent merging
        new_eids: list[str] = eid if isinstance(eid, list) else ([eid] if eid is not None else [])
        if key in seen:
            existing = merged[seen[key]]
            existing_eid = existing.get("entity_id")
            if existing_eid is None:
                merged[seen[key]] = {**existing, "entity_id": eid}
            elif isinstance(existing_eid, list):
                merged[seen[key]] = {**existing, "entity_id": existing_eid + new_eids}
            else:
                merged[seen[key]] = {**existing, "entity_id": [existing_eid] + new_eids}
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

    context   = _build_context(entities, areas)
    base_user = f"{context}\n\nTranscript: {transcript}"

    # Adaptive two-pass: the fast path runs with thinking OFF for low latency. Only
    # if the first pass fails to parse OR yields no actionable steps do we retry
    # with thinking ON (slower, but higher planning accuracy) — so the thinking
    # cost is paid only when the cheap pass wasn't good enough.
    result: dict | None = None
    for attempt in range(2):
        think = attempt == 1
        user = base_user if attempt == 0 else (
            f"{base_user}\n\nReply with ONLY the JSON object, no other text."
        )
        raw = await ollama.chat(system=_SYSTEM, user=user, format=_RESPONSE_SCHEMA, think=think)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            if attempt == 0:
                log.warning("PLAN | JSONDecodeError on attempt 1, retrying with thinking: %r", raw[:120])
                continue
            log.error("PLAN | JSONDecodeError on attempt 2, giving up: %r", raw[:120])
            return {"corrected": transcript, "intent": "action", "steps": [],
                    "ok_response": "", "fail_response": "", "parse_error": raw}

        parsed.setdefault("corrected", transcript)
        parsed.setdefault("intent", "action")
        parsed.setdefault("steps", [])
        parsed.setdefault("ok_response", "")
        parsed.setdefault("already_response", "")
        parsed.setdefault("fail_response", "")

        # Post-validate entity/area ids — fuzzy-fix or drop bad ones
        parsed["steps"] = _validate_steps(parsed["steps"], entities, areas)
        result = parsed

        # "ignore" is a deliberate empty plan (TV noise / non-directed speech) —
        # don't burn the thinking retry on it.
        if result["steps"] or result["intent"] == "ignore" or attempt == 1:
            return result
        log.info("PLAN | no actionable steps on attempt 1 — retrying with thinking")

    return result
