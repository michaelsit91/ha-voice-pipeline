# Music Stop & Pause Commands — Design Spec
**Date:** 2026-05-27

## Overview

Add voice-triggered stop and pause controls for music playback. Saying "stop the music" or "pause" stops or pauses the MA player on the satellite that issued the command.

---

## 1. Hesitation Pattern Fix

### Problem
`\bstop\b` is in `_HESITATION_PATTERNS`. "Stop the music" is intercepted before the planner runs and returns `"OK."` instead of stopping playback.

### Solution
Apply a negative lookahead so bare "stop" (cancellation intent) remains a hesitation word, but "stop the music / playing / song / track" passes through to the planner.

**Before:**
```
cancel|stop|abort
```

**After:**
```
cancel|stop(?!\s+(?:the\s+)?(?:music|playing|song|track))|abort
```

Same change is applied in both `pipeline/agents/planner.py` and `pipeline/runner.py` (both import / define the pattern).

---

## 2. Planner Changes

### New music control rule (system prompt addition)
```
MUSIC CONTROL COMMANDS:
- "stop the music" / "stop playing" / "stop that song" → media_player.media_stop on the MA player entity.
- "pause" / "pause the music" / "pause that" → media_player.media_pause on the MA player entity.
- entity_id: use the media_player entity with mass_player_type player from the Devices list.
- ok_response: "Music stopped." for stop; "Paused." for pause.
```

### New examples

```
Devices: media_player.respeaker_lite_media_player_2,Spotify,playing
Transcript: stop the music
{"corrected":"stop the music","intent":"action","steps":[{"domain":"media_player","service":"media_stop","entity_id":"media_player.respeaker_lite_media_player_2"}],"ok_response":"Music stopped.","already_response":"","fail_response":"Sorry, I couldn't stop the music."}

Transcript: pause
{"corrected":"pause","intent":"action","steps":[{"domain":"media_player","service":"media_pause","entity_id":"media_player.respeaker_lite_media_player_2"}],"ok_response":"Paused.","already_response":"","fail_response":"Sorry, I couldn't pause the music."}
```

### No schema changes needed
`domain`, `service`, and `entity_id` are already in the step schema.

---

## 3. Runner Changes

The runner already injects the satellite's resolved MA player `entity_id` into `music_assistant` steps. Extend this to also cover `media_player.media_stop` and `media_player.media_pause` steps, so the entity_id is always the correct satellite player regardless of what the LLM picks.

```python
_MUSIC_CONTROL_SERVICES = {"media_stop", "media_pause"}

for step in planned["steps"]:
    if step.get("domain") == "music_assistant":
        step["entity_id"] = ma_player
    elif (step.get("domain") == "media_player"
          and step.get("service") in _MUSIC_CONTROL_SERVICES):
        step["entity_id"] = ma_player
```

---

## 4. Executor

No changes. `_run_step` already handles `media_player` domain calls via `ha.call_service`. Stop/pause arrive here and execute normally. State-before/after diff correctly detects the transition (`playing` → `paused` / `idle`).

---

## 5. Files Changed

| File | Change |
|---|---|
| `pipeline/agents/planner.py` | Hesitation pattern fix; add music control rules + 2 examples |
| `pipeline/runner.py` | Extend MA player injection to `media_player.media_stop/pause` |

---

## 6. Testing

- Unit test: hesitation pattern does NOT match "stop the music", "stop playing", "stop that song"
- Unit test: hesitation pattern DOES match bare "stop", "stop abort"
- Unit test: runner injects MA player into `media_player.media_stop` steps
- Unit test: runner injects MA player into `media_player.media_pause` steps
- Planner accuracy test cases: "stop the music" → `media_stop`; "pause" → `media_pause`
- Integration: `POST /v1/chat/completions?satellite=respeaker_lite` with "stop the music" → `media_player.respeaker_lite_media_player_2` receives `media_stop`
