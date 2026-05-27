# Music Stop & Pause Commands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add "stop the music" and "pause" voice commands that stop or pause the MA player on the issuing satellite.

**Architecture:** Two files change. The hesitation pattern in `planner.py` gets a negative lookahead so "stop the music" reaches the planner. The planner gets music control rules + examples so it emits `media_player.media_stop` / `media_player.media_pause` steps. The runner extends its MA player injection to cover those two services. The executor needs no changes — it already handles arbitrary `media_player` calls.

**Tech Stack:** Python 3.12, pytest, pytest-asyncio, unittest.mock — same as the rest of the test suite.

---

## File Map

| File | Change |
|---|---|
| `pipeline/agents/planner.py` | Fix `_HESITATION_PATTERNS`; add music control rule + 2 examples to `_SYSTEM` |
| `pipeline/runner.py` | Extend MA player injection to `media_player.media_stop` and `media_player.media_pause` |
| `tests/test_music_control.py` | New — all tests for this feature |

---

### Task 1: Fix hesitation pattern so "stop the music" is not swallowed

**Files:**
- Modify: `pipeline/agents/planner.py:8-14`
- Test: `tests/test_music_control.py`

`_HESITATION_PATTERNS` is compiled in `planner.py` and imported by `runner.py`. The word `stop` in the pattern intercepts "stop the music" before the planner runs. A negative lookahead fixes this: bare "stop" stays a hesitation word; "stop [the] music/playing/song/track" passes through.

- [ ] **Step 1: Write failing tests**

Create `tests/test_music_control.py`:

```python
"""Tests for music stop/pause voice commands."""
import pytest
import re
from pipeline.agents.planner import _HESITATION_PATTERNS


# ── Task 1: hesitation pattern ────────────────────────────────────────────────

def test_stop_the_music_not_hesitation():
    assert not _HESITATION_PATTERNS.search("stop the music")

def test_stop_playing_not_hesitation():
    assert not _HESITATION_PATTERNS.search("stop playing")

def test_stop_that_song_not_hesitation():
    assert not _HESITATION_PATTERNS.search("stop that song")

def test_stop_the_track_not_hesitation():
    assert not _HESITATION_PATTERNS.search("stop the track")

def test_bare_stop_is_still_hesitation():
    assert _HESITATION_PATTERNS.search("stop")

def test_stop_in_sentence_is_still_hesitation():
    assert _HESITATION_PATTERNS.search("actually stop")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/vertiq/ha-voice-pipeline
python3 -m pytest tests/test_music_control.py -v 2>&1 | head -30
```

Expected: 4 tests FAIL (`stop the music`, `stop playing`, `stop that song`, `stop the track` are incorrectly matched as hesitation).

- [ ] **Step 3: Apply the hesitation pattern fix**

In `pipeline/agents/planner.py`, change the `_HESITATION_PATTERNS` definition from:

```python
_HESITATION_PATTERNS = re.compile(
    r"\b(hold on|hold up|wait|never mind|nevermind|forget it|forget that|"
    r"cancel|stop|abort|actually|scratch that|no wait|hang on)\b"
    # Chinese hesitation / cancellation words (no \b needed for CJK)
    r"|等一下|等等|算了|不对|取消|停一下|不用了|算了吧|不是这个",
    re.IGNORECASE,
)
```

to:

```python
_HESITATION_PATTERNS = re.compile(
    r"\b(hold on|hold up|wait|never mind|nevermind|forget it|forget that|"
    r"cancel|stop(?!\s+(?:the\s+)?(?:music|playing|song|track))|abort|"
    r"actually|scratch that|no wait|hang on)\b"
    # Chinese hesitation / cancellation words (no \b needed for CJK)
    r"|等一下|等等|算了|不对|取消|停一下|不用了|算了吧|不是这个",
    re.IGNORECASE,
)
```

The `(?!\s+(?:the\s+)?(?:music|playing|song|track))` negative lookahead prevents `stop` from matching when followed by optional "the" and then a music-related word.

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_music_control.py -v 2>&1 | head -20
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Verify no existing tests regressed**

```bash
python3 -m pytest tests/ -v --ignore=tests/test_agents.py 2>&1 | tail -10
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add pipeline/agents/planner.py tests/test_music_control.py
git commit -m "feat: fix hesitation pattern — stop the music passes through to planner"
```

---

### Task 2: Add music control rules and examples to the planner prompt

**Files:**
- Modify: `pipeline/agents/planner.py:93-133` (the `_SYSTEM` string)
- Test: `tests/test_music_control.py` (append to existing file)

The planner's `_SYSTEM` prompt already has a `MUSIC COMMANDS:` section. Add a `MUSIC CONTROL COMMANDS:` block immediately after it, and add two examples to the `--- EXAMPLES ---` section.

- [ ] **Step 1: Write failing unit tests for planner output (append to `tests/test_music_control.py`)**

```python
# ── Task 2: planner music control rules ───────────────────────────────────────

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock


def _make_ollama_returning(json_obj):
    """Return a mock OllamaClient whose .chat() resolves to the given JSON string."""
    ollama = MagicMock()
    ollama.chat = AsyncMock(return_value=json.dumps(json_obj))
    return ollama


_MUSIC_ENTITY = {
    "entity_id": "media_player.respeaker_lite_media_player_2",
    "name": "Spotify",
    "state": "playing",
}

_MEDIA_STOP_PLAN = {
    "corrected": "stop the music",
    "intent": "action",
    "steps": [{
        "domain": "media_player",
        "service": "media_stop",
        "entity_id": "media_player.respeaker_lite_media_player_2",
    }],
    "ok_response": "Music stopped.",
    "already_response": "",
    "fail_response": "Sorry, I couldn't stop the music.",
}

_MEDIA_PAUSE_PLAN = {
    "corrected": "pause",
    "intent": "action",
    "steps": [{
        "domain": "media_player",
        "service": "media_pause",
        "entity_id": "media_player.respeaker_lite_media_player_2",
    }],
    "ok_response": "Paused.",
    "already_response": "",
    "fail_response": "Sorry, I couldn't pause the music.",
}


@pytest.mark.asyncio
async def test_planner_stop_music_emits_media_stop():
    from pipeline.agents.planner import plan
    ollama = _make_ollama_returning(_MEDIA_STOP_PLAN)
    result = await plan("stop the music", [_MUSIC_ENTITY], [], ollama)
    steps = result["steps"]
    assert len(steps) == 1
    assert steps[0]["domain"] == "media_player"
    assert steps[0]["service"] == "media_stop"


@pytest.mark.asyncio
async def test_planner_pause_emits_media_pause():
    from pipeline.agents.planner import plan
    ollama = _make_ollama_returning(_MEDIA_PAUSE_PLAN)
    result = await plan("pause", [_MUSIC_ENTITY], [], ollama)
    steps = result["steps"]
    assert len(steps) == 1
    assert steps[0]["domain"] == "media_player"
    assert steps[0]["service"] == "media_pause"


@pytest.mark.asyncio
async def test_planner_stop_music_ok_response():
    from pipeline.agents.planner import plan
    ollama = _make_ollama_returning(_MEDIA_STOP_PLAN)
    result = await plan("stop the music", [_MUSIC_ENTITY], [], ollama)
    assert result["ok_response"] == "Music stopped."


@pytest.mark.asyncio
async def test_planner_pause_ok_response():
    from pipeline.agents.planner import plan
    ollama = _make_ollama_returning(_MEDIA_PAUSE_PLAN)
    result = await plan("pause", [_MUSIC_ENTITY], [], ollama)
    assert result["ok_response"] == "Paused."
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_music_control.py::test_planner_stop_music_emits_media_stop \
                  tests/test_music_control.py::test_planner_pause_emits_media_pause -v 2>&1 | head -20
```

These tests use a mocked Ollama that returns the expected JSON — they should PASS immediately since the planner parses the JSON and `_validate_steps` lets `media_player` steps through. This verifies the planner pipeline handles the new step shape correctly.

Expected: PASS (the plan validates and parses `media_player.media_stop/pause` steps fine with no code changes yet). If they fail, the mock JSON is malformed — check the step schema.

- [ ] **Step 3: Add the music control rules block to `_SYSTEM` in `pipeline/agents/planner.py`**

Find the line `ok_response: 'Playing {query}.' -- keep it short.` and add the new block immediately after it (before `--- EXAMPLES ---`):

```python
MUSIC CONTROL COMMANDS:
- "stop the music" / "stop playing" / "stop that song" → emit ONE step: domain=media_player, service=media_stop.
- "pause" / "pause the music" / "pause that" → emit ONE step: domain=media_player, service=media_pause.
- entity_id: use the media_player entity with mass_player_type player from the Devices list.
- ok_response: "Music stopped." for stop; "Paused." for pause.
```

The full block to insert (replace the existing `ok_response: 'Playing {query}.' -- keep it short.` line):

```python
- ok_response: 'Playing {query}.' -- keep it short.

MUSIC CONTROL COMMANDS:
- "stop the music" / "stop playing" / "stop that song" → emit ONE step: domain=media_player, service=media_stop.
- "pause" / "pause the music" / "pause that" → emit ONE step: domain=media_player, service=media_pause.
- entity_id: use the media_player entity with mass_player_type player from the Devices list.
- ok_response: "Music stopped." for stop; "Paused." for pause.
```

- [ ] **Step 4: Add two examples to the `--- EXAMPLES ---` section in `_SYSTEM`**

Find the last example in `_SYSTEM`:

```
Transcript: play hotel california by the eagles
{"corrected":"play Hotel California by the Eagles",...}
"""
```

Append before the closing `"""`:

```python

Transcript: stop the music
{"corrected":"stop the music","intent":"action","steps":[{"domain":"media_player","service":"media_stop","entity_id":"media_player.respeaker_lite_media_player_2"}],"ok_response":"Music stopped.","already_response":"","fail_response":"Sorry, I couldn't stop the music."}

Transcript: pause
{"corrected":"pause","intent":"action","steps":[{"domain":"media_player","service":"media_pause","entity_id":"media_player.respeaker_lite_media_player_2"}],"ok_response":"Paused.","already_response":"","fail_response":"Sorry, I couldn't pause the music."}
```

- [ ] **Step 5: Run all music control tests**

```bash
python3 -m pytest tests/test_music_control.py -v 2>&1 | tail -20
```

Expected: all 10 tests PASS.

- [ ] **Step 6: Verify no regressions**

```bash
python3 -m pytest tests/ -v --ignore=tests/test_agents.py 2>&1 | tail -10
```

Expected: all previously passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add pipeline/agents/planner.py tests/test_music_control.py
git commit -m "feat: add music control rules to planner — stop and pause commands"
```

---

### Task 3: Runner injects satellite MA player into media_stop and media_pause steps

**Files:**
- Modify: `pipeline/runner.py:108-114`
- Test: `tests/test_music_control.py` (append to existing file)

The runner currently injects the satellite's MA player `entity_id` only into `music_assistant` domain steps. Extend it to also inject for `media_player.media_stop` and `media_player.media_pause` steps.

- [ ] **Step 1: Write failing tests (append to `tests/test_music_control.py`)**

```python
# ── Task 3: runner injects MA player into media_stop / media_pause ────────────

from unittest.mock import patch
from pipeline.runner import run_pipeline
from pipeline.music_assistant_client import MusicAssistantClient


def _make_ha_runner(entities=None, areas=None):
    ha = MagicMock()
    ha.get_entities = AsyncMock(return_value=entities or [
        {"entity_id": "media_player.respeaker_lite_media_player_2",
         "name": "Spotify", "state": "playing"},
    ])
    ha.get_areas = AsyncMock(return_value=areas or [])
    return ha


def _make_ma_runner(player="media_player.respeaker_lite_media_player_2"):
    ma = MagicMock(spec=MusicAssistantClient)
    ma.resolve_player = MagicMock(return_value=player)
    return ma


def _plan_with_service(service: str):
    return {
        "corrected": "stop the music",
        "intent": "action",
        "steps": [{
            "domain": "media_player",
            "service": service,
            "entity_id": "media_player.wrong_player",
        }],
        "ok_response": "Music stopped.", "already_response": "", "fail_response": "Sorry.",
    }


@pytest.mark.asyncio
async def test_runner_injects_ma_player_into_media_stop():
    ha = _make_ha_runner()
    ollama = _make_ollama_returning(_plan_with_service("media_stop"))
    ma = _make_ma_runner("media_player.respeaker_lite_media_player_2")

    with patch("pipeline.runner.execute") as mock_exec:
        mock_exec.return_value = "Music stopped."
        await run_pipeline("stop the music", ha, ollama, ma=ma, satellite="respeaker_lite")

    called_steps = mock_exec.call_args.kwargs["steps"]
    assert called_steps[0]["entity_id"] == "media_player.respeaker_lite_media_player_2"


@pytest.mark.asyncio
async def test_runner_injects_ma_player_into_media_pause():
    ha = _make_ha_runner()
    ollama = _make_ollama_returning(_plan_with_service("media_pause"))
    ma = _make_ma_runner("media_player.respeaker_lite_media_player_2")

    with patch("pipeline.runner.execute") as mock_exec:
        mock_exec.return_value = "Paused."
        await run_pipeline("pause", ha, ollama, ma=ma, satellite="respeaker_lite")

    called_steps = mock_exec.call_args.kwargs["steps"]
    assert called_steps[0]["entity_id"] == "media_player.respeaker_lite_media_player_2"


@pytest.mark.asyncio
async def test_runner_does_not_inject_unrelated_media_player_steps():
    """media_player.turn_on (e.g. TV) must NOT get the MA player injected."""
    ha = _make_ha_runner(entities=[
        {"entity_id": "media_player.tv", "name": "TV", "state": "off"},
        {"entity_id": "media_player.respeaker_lite_media_player_2", "name": "Spotify", "state": "idle"},
    ])
    plan_result = {
        "corrected": "turn on the TV",
        "intent": "action",
        "steps": [{"domain": "media_player", "service": "turn_on", "entity_id": "media_player.tv"}],
        "ok_response": "TV on.", "already_response": "", "fail_response": "Sorry.",
    }
    ollama = _make_ollama_returning(plan_result)
    ma = _make_ma_runner("media_player.respeaker_lite_media_player_2")

    with patch("pipeline.runner.execute") as mock_exec:
        mock_exec.return_value = "TV on."
        await run_pipeline("turn on the TV", ha, ollama, ma=ma, satellite="respeaker_lite")

    called_steps = mock_exec.call_args.kwargs["steps"]
    assert called_steps[0]["entity_id"] == "media_player.tv"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_music_control.py::test_runner_injects_ma_player_into_media_stop \
                  tests/test_music_control.py::test_runner_injects_ma_player_into_media_pause -v 2>&1 | head -20
```

Expected: both FAIL — entity_id is still `"media_player.wrong_player"` because the runner doesn't inject for these services yet.

- [ ] **Step 3: Extend runner injection in `pipeline/runner.py`**

Find the injection block:

```python
    # Inject satellite's resolved MA player into music steps (overrides LLM's choice)
    if ma is not None:
        ma_player = ma.resolve_player(satellite)
        if ma_player:
            for step in planned["steps"]:
                if step.get("domain") == "music_assistant":
                    step["entity_id"] = ma_player
```

Replace with:

```python
    # Inject satellite's resolved MA player into music steps (overrides LLM's choice)
    _MUSIC_CONTROL_SERVICES = {"media_stop", "media_pause"}
    if ma is not None:
        ma_player = ma.resolve_player(satellite)
        if ma_player:
            for step in planned["steps"]:
                if step.get("domain") == "music_assistant":
                    step["entity_id"] = ma_player
                elif (step.get("domain") == "media_player"
                      and step.get("service") in _MUSIC_CONTROL_SERVICES):
                    step["entity_id"] = ma_player
```

- [ ] **Step 4: Run all music control tests**

```bash
python3 -m pytest tests/test_music_control.py -v 2>&1 | tail -20
```

Expected: all 13 tests PASS.

- [ ] **Step 5: Verify no regressions across full test suite**

```bash
python3 -m pytest tests/ -v --ignore=tests/test_agents.py 2>&1 | tail -15
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add pipeline/runner.py tests/test_music_control.py
git commit -m "feat: runner injects satellite MA player into media_stop and media_pause steps"
```

---

## Dashboard Assessment

A web UI dashboard was raised as a possible addition. **It's not worth it for this project right now.** Here's why:

- The pipeline is a pure voice interface — the interaction surface is audio, not a screen.
- All operational visibility already exists: structured logs with timestamps, intent, corrected transcript, latency, and outcome appear in `docker logs ha-voice-pipeline`.
- The planner accuracy benchmarks (`tests/test_planner_accuracy.py`) cover regression testing for command understanding.
- A dashboard would require a frontend framework, WebSocket or SSE plumbing for live logs, and ongoing maintenance — none of which delivers user-facing value for a voice assistant.

If observability becomes a pain point (e.g. debugging multi-satellite issues), the right move is a lightweight log aggregator (Loki + Grafana, already available in many HA setups) rather than a bespoke dashboard.
