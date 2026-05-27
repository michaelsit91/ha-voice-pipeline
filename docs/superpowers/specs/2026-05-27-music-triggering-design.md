# Music Triggering via Voice — Design Spec
**Date:** 2026-05-27

## Overview

Add voice-triggered music playback to the HA voice pipeline. Saying "play [song/artist/playlist]" searches Music Assistant (Spotify backend) and plays the result on the satellite that issued the command.

---

## 1. Satellite Routing

### Problem
HA's OpenAI conversation integration does not forward which satellite triggered the command. The pipeline receives only the transcript text.

### Solution: `?satellite=` query param
Each satellite's conversation endpoint URL is configured in HA to include a slug identifying it:

| Satellite | HA endpoint URL |
|---|---|
| ReSpeaker Lite | `http://pipeline:18795/v1/chat/completions?satellite=respeaker_lite` |
| HA Voice | `http://pipeline:18795/v1/chat/completions?satellite=home_assistant_voice_09d0e0` |

The pipeline reads `?satellite=` from the request query string and resolves it to the correct MA player. Falls back to the first discovered MA player if param is absent.

### Auto-discovery (no env entries needed)
At startup, the pipeline queries HA's device and entity registries to build the slug → MA player map automatically:

1. For each `assist_satellite.*` entity → get its `device_id`
2. Find the `media_player.*` (non-MA) on the same device — this is the physical player
3. Find the `music_assistant` platform `media_player.*` whose `active_queue` matches that physical player (name-based fallback when idle)
4. Derive slug from the satellite entity_id: `assist_satellite.X_assist_satellite` → `X`

**Discovered mapping (current):**

| slug | Satellite | MA Player |
|---|---|---|
| `respeaker_lite` | `assist_satellite.respeaker_lite_assist_satellite` | `media_player.respeaker_lite_media_player_2` |
| `home_assistant_voice_09d0e0` | `assist_satellite.home_assistant_voice_09d0e0_assist_satellite` | `media_player.home_assistant_voice_media_player` |

Only one new env var required:
```
MA_CONFIG_ENTRY_ID=01KNSX10VSWX7AB1N449NJ0AQB
```

---

## 2. Planner Changes

### New step schema fields
Music steps use `domain=music_assistant`, `service=play_media`, plus three new optional fields:

```json
{
  "domain": "music_assistant",
  "service": "play_media",
  "entity_id": "<satellite MA player entity_id>",
  "query": "We Are the Champions",
  "artist": "Queen",
  "media_type": "track"
}
```

- `query` — the search term (STT-corrected by the LLM using song/artist knowledge)
- `artist` — optional, only set when explicitly named by the user
- `media_type` — `"track"` (default), `"artist"`, `"album"`, or `"playlist"` inferred from phrasing

### JSON schema extension
`query`, `artist`, `media_type` added as optional string fields to the step schema.

### Planner prompt additions
- New rule: when intent is music playback, emit a single `music_assistant.play_media` step
- `entity_id` = the MA player from the device context (injected by the runner)
- `media_type` inference rules and examples added

### Example transcripts → steps

| Transcript | Query | Artist | MediaType |
|---|---|---|---|
| "play Blinding Lights" | `"Blinding Lights"` | — | `track` |
| "play something by The Weeknd" | `"The Weeknd"` | — | `artist` |
| "play my chill playlist" | `"chill"` | — | `playlist` |
| "play Starboy by The Weeknd" | `"Starboy"` | `"The Weeknd"` | `track` |
| "play Hotel California by the Eagles" | `"Hotel California"` | `"Eagles"` | `track` |

### STT error handling — two layers
1. **LLM correction:** the planner uses its training knowledge of songs/artists to fix phonetic STT errors (e.g. "blinding lice" → "Blinding Lights", "the weakened" → "The Weeknd") before writing `query`
2. **MA fuzzy search:** Spotify's search API handles remaining imperfections naturally

---

## 3. MusicAssistantClient

New class in `pipeline/music_assistant_client.py`.

### Responsibilities
- Auto-discover satellite → MA player map from HA registries at startup
- `search(name, media_type, artist, config_entry_id, limit)` → returns list of `{uri, name, artist}` dicts
- Called by the executor; no play logic here

### HA API used
```
POST /api/services/music_assistant/search?return_response
{
  "config_entry_id": "...",
  "name": "We Are the Champions",
  "media_type": ["track"],
  "artist": "Queen",
  "limit": 3
}
```
Returns track URIs like `spotify--GoM6sQqz://track/1lCRw5FEZ1gPDNPzy1K4zW`.

### Auto-discovery method
`discover_satellite_players(ha_url, token) -> dict[str, str]`
Returns `{slug: ma_entity_id}`. Called once on startup, cached in `MusicAssistantClient`.

---

## 4. Executor Changes

### Music step detection
```python
if step["domain"] == "music_assistant" and step["service"] == "play_media":
    return await _run_music_step(step, ha, ma_client)
```

### `_run_music_step` flow
```
1. Extract query, artist, media_type from step
2. ma_client.search(query, media_type, artist) → results
3. If no results → return "Sorry, I couldn't find {query}."
4. Pick first result → uri, track_name, artist_name
5. ha.call_service("music_assistant", "play_media",
       entity_id=step["entity_id"], media_id=uri, media_type=media_type)
6. Return f"Playing {track_name} by {artist_name}."
```

### Error responses

| Failure | Spoken response |
|---|---|
| Search returns 0 results | `"Sorry, I couldn't find {query}."` |
| `play_media` call fails | `"Sorry, I couldn't play that right now."` |
| MA integration exception | `"Sorry, Music Assistant isn't responding."` |

---

## 5. Server Changes

`pipeline/agents/server.py`:
- Read `satellite` from query params: `request.query_params.get("satellite")`
- Pass satellite slug through `run_pipeline(transcript, ha, ollama, satellite=slug)`
- Runner resolves slug → MA player entity_id via `ma_client`, injects it into the device context

---

## 6. Runner Changes

`pipeline/runner.py`:
- Accept optional `satellite` param
- Inject MA player entity_id into entity list as a `media_player` entry so the planner can reference it
- Pass `ma_client` to executor

---

## 7. Files Changed

| File | Change |
|---|---|
| `pipeline/music_assistant_client.py` | **New** — MA search + satellite auto-discovery |
| `pipeline/agents/server.py` | Read `?satellite=` query param |
| `pipeline/runner.py` | Accept satellite, inject MA player into context |
| `pipeline/agents/planner.py` | Extend schema + prompt for music steps |
| `pipeline/agents/executor.py` | Branch music steps to `_run_music_step` |
| `pipeline/ha_client.py` | Add `call_service` kwargs passthrough for `media_id` (already supports `**kwargs`) |
| `config.env` / `config.env.example` | Add `MA_CONFIG_ENTRY_ID` |

---

## 8. Testing

- Unit test `MusicAssistantClient.discover_satellite_players` against mocked WS responses
- Unit test `_run_music_step` with mocked search results (found / not found / exception)
- Integration test: `"play We Are the Champions"` with `?satellite=respeaker_lite` → correct player called
- Planner accuracy test cases for music transcripts including STT-mangled inputs
