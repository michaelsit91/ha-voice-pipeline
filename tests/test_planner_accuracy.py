"""
Planner accuracy benchmark — model-agnostic scoring across 6 command categories.

Run with default model (qwen3:9b):
    cd ha-voice-pipeline && export $(cat config.env | xargs)
    pytest tests/test_planner_accuracy.py -v -s

Compare with a different model:
    MODEL=qwen3:4b pytest tests/test_planner_accuracy.py -v -s

Each test case runs RUNS=3 times (LLM is stochastic) and reports per-run and aggregate scores.
Final summary printed at end: accuracy per category and overall.
"""
import asyncio, json, os, re, time, pytest
from pipeline.ha_client import HAClient
from pipeline.ollama_client import OllamaClient
from pipeline.agents.planner import plan

HA_URL     = os.getenv("HA_URL",     "http://localhost:8123")
HA_TOKEN   = os.getenv("HA_TOKEN",   "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
RUNS       = int(os.getenv("BENCH_RUNS", "3"))

# BENCH_MODEL: use a specific Ollama model name for benchmarking.
# Must be the EXACT Ollama model tag (e.g. "qwen3:4b", "qwen3:1.7b"), NOT an alias.
# This avoids touching the 'default' alias so wyoming-voice and the
# production pipeline keep using their assigned model uninterrupted.
# Leave unset to benchmark the same model as production (default alias).
MODEL = os.getenv("BENCH_MODEL", os.getenv("MODEL", "default"))

# Populated dynamically by the _populate_known_ids fixture before tests run
_known_ids: set[str] = set()

# ── Test fixture ─────────────────────────────────────────────────────────────
# Each case:
#   transcript      : raw voice input (may include STT errors)
#   intent          : expected "action" | "query"
#   check           : callable(result) -> (bool, str)  — True = pass, str = failure reason
#   category        : label for grouping in the summary report
#   description     : human-readable explanation of what's being tested

def _step_domains(result):
    return {s.get("domain") for s in result.get("steps", [])}

def _step_services(result):
    return {s.get("service") for s in result.get("steps", [])}

def _all_entity_ids(result):
    ids = set()
    for s in result.get("steps", []):
        eid = s.get("entity_id")
        if isinstance(eid, str):
            ids.add(eid)
        elif isinstance(eid, list):
            ids.update(eid)
        if "area_id" in s:
            ids.add(f"area:{s['area_id']}")
    return ids

def _valid_json(result):
    return (
        "corrected" in result
        and "intent" in result
        and "steps" in result
        and isinstance(result["steps"], list)
    )

def _entities_known(result):
    ids = _all_entity_ids(result)
    if not _known_ids:
        return True  # no known set loaded yet — skip validation
    return all(
        eid in _known_ids or eid.startswith("area:")
        for eid in ids
        if not eid.startswith("area:")
    )

CASES = [
    # ── Category 1: Simple single-entity actions ──────────────────────────────
    {
        "category": "simple_action",
        "description": "Turn on one specific light",
        "transcript": "turn on the office light",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and "light" in _step_domains(r)
            and "turn_on" in _step_services(r)
            and any("office" in eid for eid in _all_entity_ids(r)),
            "expected light.turn_on with 'office' in entity",
        ),
    },
    {
        "category": "simple_action",
        "description": "Turn off the kitchen ceiling light",
        "transcript": "turn off the kitchen ceiling light",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and "turn_off" in _step_services(r)
            and any("kitchen" in eid for eid in _all_entity_ids(r)),
            "expected turn_off with 'kitchen' entity",
        ),
    },
    {
        "category": "simple_action",
        "description": "Turn on the hallway light",
        "transcript": "turn on the hallway light",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and "turn_on" in _step_services(r)
            and any("hallway" in eid for eid in _all_entity_ids(r)),
            "expected turn_on with 'hallway' entity",
        ),
    },

    # ── Category 2: Fan control ───────────────────────────────────────────────
    {
        "category": "fan_control",
        "description": "Turn on office fan",
        "transcript": "turn on the office fan",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and "fan" in _step_domains(r)
            and "turn_on" in _step_services(r)
            and any("office" in eid for eid in _all_entity_ids(r)),
            "expected fan.turn_on with 'office' entity",
        ),
    },
    {
        "category": "fan_control",
        "description": "Toggle bedroom fan",
        "transcript": "toggle the bedroom fan",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and "fan" in _step_domains(r)
            and any("master_bedroom" in eid or "bedroom" in eid
                    for eid in _all_entity_ids(r)),
            "expected fan entity with 'bedroom'",
        ),
    },

    # ── Category 3: Status queries ────────────────────────────────────────────
    {
        "category": "status_query",
        "description": "Is the office light on",
        "transcript": "is the office light on",
        "intent": "query",
        "check": lambda r: (
            _valid_json(r)
            and r.get("intent") == "query"
            and "get_state" in _step_services(r)
            and any("office" in eid for eid in _all_entity_ids(r)),
            "expected query intent with get_state and 'office' entity",
        ),
    },
    {
        "category": "status_query",
        "description": "Is the kitchen ceiling light on",
        "transcript": "is the kitchen ceiling light on",
        "intent": "query",
        "check": lambda r: (
            _valid_json(r)
            and r.get("intent") == "query"
            and any("kitchen" in eid for eid in _all_entity_ids(r)),
            "expected query with kitchen entity",
        ),
    },

    # ── Category 4: STT correction ────────────────────────────────────────────
    {
        "category": "stt_correction",
        "description": "Garbled verb + device type (office intact)",
        "transcript": "tern on the office lait",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and "turn_on" in _step_services(r)
            and any("office" in eid for eid in _all_entity_ids(r)),
            "expected corrected to turn_on office light",
        ),
    },
    {
        "category": "stt_correction",
        "description": "Garbled room name approximation",
        "transcript": "turn on the kitchin light",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and "turn_on" in _step_services(r)
            and any("kitchen" in eid for eid in _all_entity_ids(r)),
            "expected corrected to kitchen light",
        ),
    },
    {
        "category": "stt_correction",
        "description": "Multiple phonetic errors in one utterance",
        "transcript": "tun of the oface seeling fan",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and "fan" in _step_domains(r)
            and any("office" in eid for eid in _all_entity_ids(r)),
            "expected fan entity with office, despite heavy STT errors",
        ),
    },

    # ── Category 5: Multi-device (consolidated steps) ─────────────────────────
    {
        "category": "multi_device",
        "description": "All lights in a named room — expects 1 consolidated step",
        "transcript": "turn off all the living room lights",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and len(r.get("steps", [])) == 1
            and "turn_off" in _step_services(r)
            and (
                any("area_id" in s for s in r["steps"])
                or any(isinstance(s.get("entity_id"), list) for s in r["steps"])
            ),
            "expected 1 consolidated step with area_id or list entity_id",
        ),
    },
    {
        "category": "multi_device",
        "description": "All bedroom lights — expects 1 step",
        "transcript": "turn on all the bedroom lights",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and len(r.get("steps", [])) == 1
            and "turn_on" in _step_services(r),
            "expected 1 consolidated step for bedroom lights",
        ),
    },

    # ── Category 6: Complex / multi-domain ───────────────────────────────────
    {
        "category": "complex",
        "description": "Two devices same domain — light and fan in same command",
        "transcript": "turn on the office light and the office fan",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and len(r.get("steps", [])) >= 1
            and any("office" in eid for eid in _all_entity_ids(r)),
            "expected at least one step targeting office entities",
        ),
    },
    {
        "category": "complex",
        "description": "Cross-domain: lights off in one room, fan off in another",
        "transcript": "turn off the guest room lights and the living room fan",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and len(r.get("steps", [])) >= 1
            and any("guest" in eid or "living" in eid
                    for eid in _all_entity_ids(r)),
            "expected steps covering guest room and/or living room",
        ),
    },
    {
        "category": "complex",
        "description": "Implicit 'make it dark' phrasing — natural language intent",
        "transcript": "make the guest room dark",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and "turn_off" in _step_services(r)
            and any("guest" in eid for eid in _all_entity_ids(r)),
            "expected turn_off targeting guest room entities",
        ),
    },
    {
        "category": "complex",
        "description": "Toggle all fans — should consolidate",
        "transcript": "toggle all the fans",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and "fan" in _step_domains(r)
            and len(r.get("steps", [])) <= 2,
            "expected fan steps, consolidated to ≤2",
        ),
    },
    {
        "category": "complex",
        "description": "Status query for multiple lights in a room",
        "transcript": "are the hallway lights on",
        "intent": "query",
        "check": lambda r: (
            _valid_json(r)
            and r.get("intent") == "query"
            and any("hallway" in eid for eid in _all_entity_ids(r)),
            "expected query with hallway entity",
        ),
    },

    # ── Category 7: Ambiguous commands ───────────────────────────────────────
    # No specific target — success criteria are structural (valid JSON, correct
    # intent, a coherent response string) rather than entity-specific.
    # The planner is allowed to pick ANY device or broadcast to all; we do NOT
    # require entities to be in KNOWN since LED rings etc. may appear.
    {
        "category": "ambiguous",
        "description": "No room specified — 'turn on the light'",
        "transcript": "turn on the light",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and r.get("intent") == "action"
            and "turn_on" in _step_services(r)
            and len(r.get("steps", [])) >= 1
            and bool(r.get("ok_response")),
            "expected action intent, turn_on, ≥1 step, non-empty ok_response",
        ),
    },
    {
        "category": "ambiguous",
        "description": "No room specified — 'turn off the fan'",
        "transcript": "turn off the fan",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and r.get("intent") == "action"
            and "fan" in _step_domains(r)
            and "turn_off" in _step_services(r),
            "expected action with fan domain and turn_off",
        ),
    },
    {
        "category": "ambiguous",
        "description": "Everything off — broad broadcast command",
        "transcript": "turn off everything",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and r.get("intent") == "action"
            and "turn_off" in _step_services(r)
            and len(r.get("steps", [])) >= 1,
            "expected at least one turn_off step",
        ),
    },
    {
        "category": "ambiguous",
        "description": "Implicit lighting intent — 'make it dark'",
        "transcript": "make it dark",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and r.get("intent") == "action"
            and "turn_off" in _step_services(r),
            "expected action with turn_off (darkness = lights off)",
        ),
    },
    {
        "category": "complex",
        "description": "Mixed STT errors + multi-device",
        "transcript": "tun off the guets rum lites and the fan",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and "turn_off" in _step_services(r)
            and any("guest" in eid or "fan" in eid
                    for eid in _all_entity_ids(r)),
            "expected turn_off with guest room or fan entities despite STT errors",
        ),
    },

    # ── Category 8: Chinese (ZH) commands ────────────────────────────────────
    {
        "category": "zh_action",
        "description": "ZH: turn on office light (打开办公室灯)",
        "transcript": "打开办公室灯",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and r.get("intent") == "action"
            and "turn_on" in _step_services(r)
            and any("office" in eid for eid in _all_entity_ids(r)),
            "expected turn_on with office entity from ZH transcript",
        ),
    },
    {
        "category": "zh_action",
        "description": "ZH: turn off living room fan (关掉客厅风扇)",
        "transcript": "关掉客厅风扇",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and r.get("intent") == "action"
            and "turn_off" in _step_services(r)
            and any("living" in eid or "area:living" in eid
                    for eid in _all_entity_ids(r)),
            "expected turn_off with living_room entity/area from ZH transcript",
        ),
    },
    {
        "category": "zh_query",
        "description": "ZH: query office light state (办公室灯开着吗)",
        "transcript": "办公室灯开着吗",
        "intent": "query",
        "check": lambda r: (
            _valid_json(r)
            and r.get("intent") == "query"
            and "get_state" in _step_services(r)
            and any("office" in eid for eid in _all_entity_ids(r)),
            "expected query with get_state and office entity from ZH transcript",
        ),
    },

    # ── Category 9: Music playback ────────────────────────────────────────────
    {
        "category": "music_play",
        "description": "Play a specific song by title",
        "transcript": "play Blinding Lights",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and r.get("intent") == "action"
            and any(s.get("domain") == "music_assistant" for s in r.get("steps", []))
            and any(s.get("service") == "play_media" for s in r.get("steps", []))
            and any(s.get("query") for s in r.get("steps", [])),
            "expected music_assistant.play_media step with non-empty query",
        ),
    },
    {
        "category": "music_play",
        "description": "Play song with STT errors (blainding lites)",
        "transcript": "play blainding lites",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and r.get("intent") == "action"
            and any(s.get("domain") == "music_assistant" for s in r.get("steps", []))
            and any(
                "blind" in s.get("query", "").lower() or "light" in s.get("query", "").lower()
                for s in r.get("steps", [])
            ),
            "expected music_assistant step, query corrected toward 'Blinding Lights'",
        ),
    },
    {
        "category": "music_play",
        "description": "Play by artist (something by X)",
        "transcript": "play something by The Weeknd",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and r.get("intent") == "action"
            and any(s.get("domain") == "music_assistant" for s in r.get("steps", []))
            and any(
                s.get("media_type") == "artist"
                or "weeknd" in s.get("query", "").lower()
                for s in r.get("steps", [])
            ),
            "expected music_assistant step targeting artist The Weeknd",
        ),
    },
    {
        "category": "music_play",
        "description": "Play song with artist named (title by artist)",
        "transcript": "play Hotel California by the Eagles",
        "intent": "action",
        "check": lambda r: (
            _valid_json(r)
            and r.get("intent") == "action"
            and any(s.get("domain") == "music_assistant" for s in r.get("steps", []))
            and any(
                "california" in s.get("query", "").lower()
                or "hotel" in s.get("query", "").lower()
                for s in r.get("steps", [])
            ),
            "expected music_assistant step with 'Hotel California' in query",
        ),
    },
]

# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def ha():
    return HAClient(HA_URL, HA_TOKEN)

@pytest.fixture(scope="module")
def ollama():
    return OllamaClient(OLLAMA_URL, MODEL)

@pytest.fixture(scope="module")
def ha_context(ha, event_loop):
    entities = event_loop.run_until_complete(ha.get_entities())
    areas    = event_loop.run_until_complete(ha.get_areas())
    return {"entities": entities, "areas": areas}

# ── Scoring ───────────────────────────────────────────────────────────────────
_results_store: dict[str, list[dict]] = {}

def _score_run(case: dict, result: dict) -> dict:
    check_fn = case["check"]
    check_result = check_fn(result)
    if isinstance(check_result, tuple):
        passed, reason = check_result
    else:
        passed, reason = check_result, ""

    return {
        "json_valid":      _valid_json(result),
        "intent_correct":  result.get("intent") == case["intent"],
        "check_passed":    bool(passed),
        "check_reason":    reason if not passed else "",
        "has_ok_response": bool(result.get("ok_response")),
        "entities_known":  _entities_known(result),
        "steps_count":     len(result.get("steps", [])),
        "latency_s":       0.0,   # filled in by test runner
        "result":          result,
    }

# ── Test runner ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("case", CASES, ids=[c["description"][:50] for c in CASES])
async def test_planner_case(case, ollama, ha_context):
    entities = ha_context["entities"]
    areas    = ha_context["areas"]
    key      = case["description"]

    run_scores = []
    print(f"\n  [{case['category']}] {case['description']}")
    print(f"  transcript: {case['transcript']!r}")

    for i in range(RUNS):
        t0     = time.perf_counter()
        result = await plan(case["transcript"], entities, areas, ollama)
        lat    = time.perf_counter() - t0
        score  = _score_run(case, result)
        score["latency_s"] = lat
        run_scores.append(score)
        status = "✓" if score["check_passed"] else "✗"
        print(f"  run {i+1}: {status}  {lat:.2f}s  intent={result.get('intent'):8s}  "
              f"steps={score['steps_count']}  entities={list(_all_entity_ids(result))[:3]}")
        if not score["check_passed"]:
            print(f"         reason: {score['check_reason']}")

    _results_store[key] = run_scores

    lats      = [s["latency_s"] for s in run_scores]
    pass_rate = sum(1 for s in run_scores if s["check_passed"]) / len(run_scores)
    print(f"  pass={pass_rate:.0%}  lat={sum(lats)/len(lats):.2f}s avg  "
          f"[{min(lats):.2f}–{max(lats):.2f}s]")

    # Soft pass: >50% of runs correct (LLM is stochastic)
    assert pass_rate >= 0.5, (
        f"Pass rate {pass_rate:.0%} below 50% threshold for: {case['description']}\n"
        f"Last result: {json.dumps(run_scores[-1]['result'], indent=2)}"
    )


# ── Known-ID loader (session-scoped, fires before any test) ──────────────────
@pytest.fixture(scope="session", autouse=True)
def _populate_known_ids(ha, event_loop):
    """Fetch live entity list from HA and populate the known entity ID set."""
    global _known_ids
    entities = event_loop.run_until_complete(ha.get_entities())
    _known_ids = {e["entity_id"] for e in entities}
    print(f"\n  [benchmark] loaded {len(_known_ids)} entity IDs from HA")


# ── Summary report (session-scoped autouse fixture, fires after all tests) ─────
@pytest.fixture(scope="session", autouse=True)
def _print_accuracy_summary():
    yield  # run all tests first

    if not _results_store:
        return

    by_cat: dict[str, list] = {}
    for key, runs in _results_store.items():
        case = next(c for c in CASES if c["description"] == key)
        cat  = case["category"]
        by_cat.setdefault(cat, []).extend(runs)

    W = 78
    print("\n" + "=" * W)
    print(f"  PLANNER BENCHMARK  — model: {MODEL}  runs/case: {RUNS}")
    print("=" * W)
    print(f"  {'category':<20} {'pass%':>6}  {'intent%':>7}  {'json%':>6}  {'ents%':>6}  "
          f"{'avg(s)':>7}  {'min(s)':>6}  {'max(s)':>6}")
    print(f"  {'-'*(W-2)}")

    total_runs = []
    for cat, runs in sorted(by_cat.items()):
        pass_r   = sum(1 for r in runs if r["check_passed"])   / len(runs)
        intent_r = sum(1 for r in runs if r["intent_correct"]) / len(runs)
        json_r   = sum(1 for r in runs if r["json_valid"])     / len(runs)
        ents_r   = sum(1 for r in runs if r["entities_known"]) / len(runs)
        lats     = [r["latency_s"] for r in runs]
        avg_l, min_l, max_l = sum(lats)/len(lats), min(lats), max(lats)
        print(f"  {cat:<20} {pass_r:>5.0%}   {intent_r:>6.0%}   {json_r:>5.0%}   {ents_r:>5.0%}  "
              f"  {avg_l:>6.2f}s  {min_l:>5.2f}s  {max_l:>5.2f}s")
        total_runs.extend(runs)

    print(f"  {'-'*(W-2)}")
    overall  = sum(1 for r in total_runs if r["check_passed"])   / len(total_runs)
    intent_o = sum(1 for r in total_runs if r["intent_correct"]) / len(total_runs)
    json_o   = sum(1 for r in total_runs if r["json_valid"])     / len(total_runs)
    ents_o   = sum(1 for r in total_runs if r["entities_known"]) / len(total_runs)
    all_lats = [r["latency_s"] for r in total_runs]
    avg_o, min_o, max_o = sum(all_lats)/len(all_lats), min(all_lats), max(all_lats)
    print(f"  {'OVERALL':<20} {overall:>5.0%}   {intent_o:>6.0%}   {json_o:>5.0%}   {ents_o:>5.0%}  "
          f"  {avg_o:>6.2f}s  {min_o:>5.2f}s  {max_o:>5.2f}s")
    print("=" * W)
    print(f"  To compare: BENCH_MODEL=qwen3:4b pytest tests/test_planner_accuracy.py -v -s")
