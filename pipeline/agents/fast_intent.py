"""Deterministic intent matcher — the Alexa-like fast path.

Maps the most common voice commands (on/off/toggle, volume, media stop/pause,
state queries) directly to a Home Assistant action WITHOUT calling the LLM, but
only when a single entity target matches confidently. Anything ambiguous,
compound, or unrecognised returns None so the caller falls back to the planner.
"""
import re

# Words stripped from the transcript before fuzzy-matching the device target.
_FILLER = {
    "turn", "on", "off", "switch", "shut", "the", "a", "an", "toggle", "dim",
    "set", "to", "is", "are", "what", "whats", "please", "of", "my", "in", "it",
    "percent", "down", "up", "lower", "raise", "make", "can", "you", "all", "every",
}

# Intents valid for the fast path, by entity domain (others fall back to the LLM).
_TOGGLEABLE = {"light", "switch", "fan", "input_boolean"}

_PCT = r"(\d{1,3})\s*(?:percent|%)"
# Empty on purpose: fast-path successes are acknowledged by the satellite's
# success chime (firmware on_end), not by spoken TTS — snappier, Alexa-like.
_ACK = ""


def _target_text(transcript: str) -> str:
    """The probable device-name portion: transcript minus command/filler words."""
    toks = [
        w for w in re.split(r"\W+", transcript.lower())
        if w and not w.isdigit() and w not in _FILLER
    ]
    return " ".join(toks)


def _score(a: str, b: str) -> float:
    """Fuzzy similarity 0–100. Uses rapidfuzz in production; falls back to the
    stdlib difflib ratio if rapidfuzz isn't installed (degraded but functional)."""
    try:
        from rapidfuzz import fuzz
        return fuzz.token_set_ratio(a, b)
    except ImportError:
        import difflib
        return difflib.SequenceMatcher(None, a, b).ratio() * 100


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("_", " ").strip().lower())


def _resolve_area(place: str, areas: list[dict]):
    """Exact-ish area match for a room name. Uses a length-sensitive ratio (not
    token-set) so 'kitchen' matches the Kitchen area but 'kitchen ceiling' does
    NOT (that's a specific entity, handled by _resolve_entity)."""
    if not place:
        return None
    import difflib
    p = _norm(place)
    best, best_s = None, 0.0
    for a in areas:
        for cand in (a.get("name", ""), a.get("area_id", "")):
            n = _norm(cand)
            s = 100.0 if n == p else difflib.SequenceMatcher(None, p, n).ratio() * 100
            if s > best_s:
                best_s, best = s, a
    return best if best_s >= 90 else None


def _resolve_entity(target: str, entities: list[dict], min_score: int, margin: int):
    """Best confident entity for the target name, or None if absent/ambiguous."""
    if not target:
        return None
    scored = sorted(
        ((_score(target, e["name"].lower()), e) for e in entities),
        key=lambda x: -x[0],
    )
    if not scored:
        return None
    best_score, best = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0
    if best_score >= min_score and (best_score - second) >= margin:
        return best
    return None


_PLAY_RE = re.compile(r"^(?:play|put on)\s+(.+?)[.!?]?$")
# Queries too vague to resolve deterministically — let the LLM interpret them.
# Covers bare requests ("play music") and mood requests ("play something relaxing").
_VAGUE_PLAY = re.compile(
    r"^(?:some\s+)?(?:music|songs?|radio)$"
    r"|^(?:something|anything|a song)(?:\s+.+)?$"
)
_MEDIA_TYPE_PREFIX = re.compile(
    r"^(?:the\s+)?(album|playlist|artist)\s+(.+)$"
)
_ARTIST_STYLE = re.compile(
    r"^(?:some|songs?\s+by|music\s+by|tracks?\s+by)\s+(.+)$"
)


def _match_play(t: str) -> dict | None:
    """Deterministic 'play <query> [by <artist>]' → music step (skips the LLM).

    Vague requests ('play some music', 'play something relaxing') return None
    so the planner can interpret them.
    """
    m = _PLAY_RE.match(t)
    if m is None:
        return None
    rest = m.group(1).strip()
    if _VAGUE_PLAY.match(rest):
        return None

    media_type = "track"
    tm = _MEDIA_TYPE_PREFIX.match(rest)
    if tm:
        media_type, rest = tm.group(1), tm.group(2).strip()
    else:
        am = _ARTIST_STYLE.match(rest)
        if am:
            media_type, rest = "artist", am.group(1).strip()

    artist = None
    if media_type == "track":
        parts = re.split(r"\s+by\s+", rest, maxsplit=1)
        if len(parts) == 2:
            rest, artist = parts[0].strip(), parts[1].strip()

    if not rest:
        return None
    return {"kind": "music", "query": rest, "artist": artist,
            "media_type": media_type, "needs_player": True}


def match_fast_intent(
    transcript: str,
    entities: list[dict],
    areas: list[dict],
    min_score: int = 82,
    margin: int = 8,
) -> dict | None:
    """Return a fast-path plan dict, or None to fall back to the LLM planner."""
    t = transcript.lower().strip()

    # Multi-clause commands need the planner.
    if " and " in t:
        return None

    # ── Media controls — no named device target; the runner injects the player ──
    m = re.search(r"volume\b.*?" + _PCT, t) or re.search(_PCT + r"\s*volume", t)
    if m:
        level = max(0, min(100, int(m.group(1)))) / 100
        return {"kind": "action", "domain": "media_player", "service": "volume_set",
                "volume_level": round(level, 2), "needs_player": True, "ack": _ACK}
    if re.search(r"\b(volume up|louder|turn it up|turn up the volume)\b", t):
        return {"kind": "action", "domain": "media_player", "service": "volume_up",
                "needs_player": True, "ack": _ACK}
    if re.search(r"\b(volume down|quieter|softer|turn it down|lower the volume)\b", t):
        return {"kind": "action", "domain": "media_player", "service": "volume_down",
                "needs_player": True, "ack": _ACK}
    if re.search(r"\bstop\b", t):  # bare "stop" is caught as hesitation upstream
        return {"kind": "action", "domain": "media_player", "service": "media_stop",
                "needs_player": True, "ack": _ACK}
    if re.search(r"\bpause\b", t):
        return {"kind": "action", "domain": "media_player", "service": "media_pause",
                "needs_player": True, "ack": _ACK}

    music = _match_play(t)
    if music is not None:
        return music

    # ── Device-targeted intents ─────────────────────────────────────────────────
    kind = service = None
    if re.search(r"\b(turn|switch)\s+on\b", t):
        kind, service = "action", "turn_on"
    elif re.search(r"\b(turn|switch|shut)\s+off\b", t):
        kind, service = "action", "turn_off"
    elif re.search(r"\btoggle\b", t):
        kind, service = "action", "toggle"
    elif re.search(r"^\s*(is|are)\b.*\b(on|off)\b", t) or re.search(r"\bwhat('?s| is)\b", t):
        kind, service = "query", "get_state"
    if kind is None:
        return None

    # Room command: "<area> light(s)/fan(s)" → target the whole area so HA controls
    # every entity of that domain in the room, regardless of misleading entity_ids.
    if kind == "action":
        domain_word = ("light" if re.search(r"\blights?\b", t)
                       else "fan" if re.search(r"\bfans?\b", t) else None)
        if domain_word:
            place = re.sub(r"\b(lights?|fans?)\b", " ", _target_text(t))
            area = _resolve_area(re.sub(r"\s+", " ", place).strip(), areas)
            if area is not None:
                return {"kind": "action", "domain": domain_word, "service": service,
                        "area_id": area["area_id"], "ack": _ACK}

    entity = _resolve_entity(_target_text(t), entities, min_score, margin)
    if entity is None:
        return None
    domain = entity["entity_id"].split(".")[0]

    if kind == "query":
        return {"kind": "query", "entity_id": entity["entity_id"], "name": entity["name"]}

    # Action: intent must be valid for the matched domain.
    if domain not in _TOGGLEABLE:
        return None
    return {"kind": "action", "domain": domain, "service": service,
            "entity_id": entity["entity_id"], "ack": _ACK}
