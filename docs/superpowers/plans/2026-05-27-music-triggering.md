# Music Triggering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add voice-triggered music playback — "play [song]" searches Music Assistant and plays on the satellite that issued the command.

**Architecture:** New `MusicAssistantClient` auto-discovers satellite→MA-player mapping from HA states at startup. Planner emits `music_assistant.play_media` steps with a `query` field. Executor branches on domain, calls MA search, then plays the top result. Server reads `?satellite=` query param to route to the right player.

**Tech Stack:** Python 3.12, httpx (already in requirements), FastAPI, Home Assistant REST API, Music Assistant HA integration.

**Spec:** `docs/superpowers/specs/2026-05-27-music-triggering-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `pipeline/music_assistant_client.py` | **Create** | Auto-discovery, satellite→player map, MA search |
| `tests/test_music_assistant_client.py` | **Create** | Unit tests for discovery + search (mocked httpx) |
| `pipeline/agents/planner.py` | **Modify** | Add `query`/`artist`/`media_type` to step schema + music prompt rules |
| `tests/test_planner_accuracy.py` | **Modify** | Add `music_play` category cases |
| `pipeline/agents/executor.py` | **Modify** | Add `_run_music_step`, branch on `domain == "music_assistant"` |
| `tests/test_music_executor.py` | **Create** | Unit tests for music executor (mocked MA + HA) |
| `pipeline/runner.py` | **Modify** | Accept `ma` + `satellite` params, inject MA player into music steps |
| `pipeline/agents/server.py` | **Modify** | Init `_ma`, run discovery on startup, read `?satellite=` param |
| `config.env.example` | **Modify** | Add `MA_CONFIG_ENTRY_ID` |

---

## Task 1: `MusicAssistantClient` — slug helpers, discovery, resolve_player

**Files:**
- Create: `pipeline/music_assistant_client.py`
- Create: `tests/test_music_assistant_client.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_music_assistant_client.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pipeline.music_assistant_client import (
    MusicAssistantClient,
    _satellite_slug,
    _physical_player_slug,
    _discover_satellite_players,
)

# ── slug helpers ──────────────────────────────────────────────────────────────

def test_satellite_slug_respeaker():
    assert _satellite_slug("assist_satellite.respeaker_lite_assist_satellite") == "respeaker_lite"

def test_satellite_slug_ha_voice():
    assert _satellite_slug(
        "assist_satellite.home_assistant_voice_09d0e0_assist_satellite"
    ) == "home_assistant_voice_09d0e0"

def test_physical_player_slug_plain():
    assert _physical_player_slug("media_player.respeaker_lite_media_player") == "respeaker_lite"

def test_physical_player_slug_numbered():
    assert _physical_player_slug("media_player.respeaker_lite_media_player_2") == "respeaker_lite"

def test_physical_player_slug_ha_voice():
    assert _physical_player_slug(
        "media_player.home_assistant_voice_09d0e0_media_player"
    ) == "home_assistant_voice_09d0e0"

# ── discovery ─────────────────────────────────────────────────────────────────

_MOCK_STATES = [
    {"entity_id": "assist_satellite.respeaker_lite_assist_satellite",
     "state": "idle", "attributes": {}},
    {"entity_id": "assist_satellite.home_assistant_voice_09d0e0_assist_satellite",
     "state": "idle", "attributes": {}},
    {"entity_id": "media_player.respeaker_lite_media_player",
     "state": "idle", "attributes": {}},
    {"entity_id": "media_player.home_assistant_voice_09d0e0_media_player",
     "state": "idle", "attributes": {}},
    {"entity_id": "media_player.respeaker_lite_media_player_2",
     "state": "playing",
     "attributes": {"mass_player_type": "player",
                    "active_queue": "media_player.respeaker_lite_media_player"}},
    {"entity_id": "media_player.home_assistant_voice_media_player",
     "state": "idle",
     "attributes": {"mass_player_type": "player",
                    "active_queue": "media_player.home_assistant_voice_09d0e0_media_player"}},
]

def _mock_http(states):
    mock_resp = MagicMock()
    mock_resp.json.return_value = states
    mock_resp.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client

@pytest.mark.asyncio
async def test_discover_maps_both_satellites():
    with patch("pipeline.music_assistant_client.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_http(_MOCK_STATES)
        result = await _discover_satellite_players(
            "http://ha:8123", {"Authorization": "Bearer x"}
        )
    assert result["respeaker_lite"] == "media_player.respeaker_lite_media_player_2"
    assert result["home_assistant_voice_09d0e0"] == "media_player.home_assistant_voice_media_player"

@pytest.mark.asyncio
async def test_discover_fallback_when_active_queue_null():
    states = []
    for s in _MOCK_STATES:
        s2 = {"entity_id": s["entity_id"], "state": s["state"],
              "attributes": dict(s["attributes"])}
        if s2["attributes"].get("mass_player_type") == "player":
            s2["attributes"]["active_queue"] = None
        states.append(s2)

    with patch("pipeline.music_assistant_client.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_http(states)
        result = await _discover_satellite_players(
            "http://ha:8123", {"Authorization": "Bearer x"}
        )
    assert result.get("respeaker_lite") == "media_player.respeaker_lite_media_player_2"

# ── resolve_player ────────────────────────────────────────────────────────────

def test_resolve_player_known_slug():
    ma = MusicAssistantClient("http://ha:8123", "token", "entry-id")
    ma._satellite_map = {
        "respeaker_lite": "media_player.respeaker_lite_media_player_2",
        "ha_voice": "media_player.home_assistant_voice_media_player",
    }
    assert ma.resolve_player("respeaker_lite") == "media_player.respeaker_lite_media_player_2"

def test_resolve_player_unknown_slug_falls_back_to_first():
    ma = MusicAssistantClient("http://ha:8123", "token", "entry-id")
    ma._satellite_map = {"respeaker_lite": "media_player.respeaker_lite_media_player_2"}
    assert ma.resolve_player("unknown") == "media_player.respeaker_lite_media_player_2"

def test_resolve_player_none_falls_back_to_first():
    ma = MusicAssistantClient("http://ha:8123", "token", "entry-id")
    ma._satellite_map = {"respeaker_lite": "media_player.respeaker_lite_media_player_2"}
    assert ma.resolve_player(None) == "media_player.respeaker_lite_media_player_2"

def test_resolve_player_empty_map_returns_none():
    ma = MusicAssistantClient("http://ha:8123", "token", "entry-id")
    assert ma.resolve_player("respeaker_lite") is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/vertiq/ha-voice-pipeline && pytest tests/test_music_assistant_client.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'pipeline.music_assistant_client'`

- [ ] **Step 3: Create `pipeline/music_assistant_client.py`**

```python
import logging, re
import httpx

log = logging.getLogger("pipeline")


def _satellite_slug(entity_id: str) -> str:
    """assist_satellite.respeaker_lite_assist_satellite → respeaker_lite"""
    s = re.sub(r"^assist_satellite\.", "", entity_id)
    return re.sub(r"_assist_satellite$", "", s)


def _physical_player_slug(entity_id: str) -> str:
    """media_player.respeaker_lite_media_player(_N)? → respeaker_lite"""
    s = re.sub(r"^media_player\.", "", entity_id)
    return re.sub(r"_media_player(_\d+)?$", "", s)


async def _discover_satellite_players(url: str, hdrs: dict) -> dict[str, str]:
    """Build {satellite_slug: ma_player_entity_id} from HA /api/states."""
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{url}/api/states", headers=hdrs, timeout=10)
        r.raise_for_status()
    states = r.json()

    satellite_slugs = {
        _satellite_slug(s["entity_id"])
        for s in states
        if s["entity_id"].startswith("assist_satellite.")
    }

    ma_players = [
        s for s in states
        if s["entity_id"].startswith("media_player.")
        and s.get("attributes", {}).get("mass_player_type") == "player"
    ]

    result: dict[str, str] = {}

    # Primary: match via active_queue attribute (set when player is active)
    for ma in ma_players:
        aq = ma.get("attributes", {}).get("active_queue") or ""
        if aq.startswith("media_player."):
            slug = _physical_player_slug(aq)
            if slug in satellite_slugs:
                result[slug] = ma["entity_id"]

    # Fallback: slug match on entity_id for idle players
    for ma in ma_players:
        if ma["entity_id"] in result.values():
            continue
        ma_slug = _physical_player_slug(ma["entity_id"])
        for sat_slug in satellite_slugs:
            if sat_slug not in result and sat_slug == ma_slug:
                result[sat_slug] = ma["entity_id"]

    return result


class MusicAssistantClient:
    def __init__(self, ha_url: str, token: str, config_entry_id: str):
        self._url = ha_url.rstrip("/")
        self._hdrs = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self._config_entry_id = config_entry_id
        self._satellite_map: dict[str, str] = {}

    async def discover(self) -> None:
        """Populate satellite→player map from HA. Call once at startup."""
        self._satellite_map = await _discover_satellite_players(
            self._url, self._hdrs
        )
        log.info(
            "MUSIC | discovered %d satellite player(s): %s",
            len(self._satellite_map),
            self._satellite_map,
        )

    def resolve_player(self, satellite_slug: str | None) -> str | None:
        """Return MA player entity_id for slug, or first discovered player."""
        if satellite_slug and satellite_slug in self._satellite_map:
            return self._satellite_map[satellite_slug]
        if self._satellite_map:
            return next(iter(self._satellite_map.values()))
        return None

    async def search(
        self,
        name: str,
        media_type: str = "track",
        artist: str | None = None,
        limit: int = 3,
    ) -> list[dict]:
        """Search MA. Returns list of {uri, name, artist} dicts."""
        payload: dict = {
            "config_entry_id": self._config_entry_id,
            "name": name,
            "media_type": [media_type],
            "limit": limit,
        }
        if artist:
            payload["artist"] = artist
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{self._url}/api/services/music_assistant/search",
                headers=self._hdrs,
                json=payload,
                params={"return_response": ""},
                timeout=10,
            )
            r.raise_for_status()
        sr = r.json().get("service_response", {})
        items = sr.get(f"{media_type}s", [])
        results = []
        for item in items:
            uri = item.get("uri", "")
            n = item.get("name", "")
            if item.get("media_type") == "artist":
                a = n
            else:
                artists = item.get("artists", [])
                a = artists[0].get("name", "") if artists else ""
            results.append({"uri": uri, "name": n, "artist": a})
        return results
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/vertiq/ha-voice-pipeline && pytest tests/test_music_assistant_client.py -v
```

Expected: all 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/vertiq/ha-voice-pipeline
git add pipeline/music_assistant_client.py tests/test_music_assistant_client.py
git commit -m "feat: add MusicAssistantClient with satellite auto-discovery and MA search"
```

---

## Task 2: Add `search()` tests for artist/empty responses

**Files:**
- Modify: `tests/test_music_assistant_client.py`

- [ ] **Step 1: Add search tests to `tests/test_music_assistant_client.py`**

Append to the end of the existing test file:

```python
# ── search ────────────────────────────────────────────────────────────────────

def _mock_http_post(response_body):
    mock_resp = MagicMock()
    mock_resp.json.return_value = response_body
    mock_resp.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client

@pytest.mark.asyncio
async def test_search_track_returns_name_and_artist():
    ma = MusicAssistantClient("http://ha:8123", "token", "entry-123")
    body = {
        "service_response": {
            "tracks": [{
                "media_type": "track",
                "uri": "spotify://track/abc123",
                "name": "Blinding Lights",
                "artists": [{"name": "The Weeknd", "media_type": "artist"}],
            }],
            "artists": [], "albums": [], "playlists": [],
        }
    }
    with patch("pipeline.music_assistant_client.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_http_post(body)
        results = await ma.search("Blinding Lights", media_type="track")
    assert len(results) == 1
    assert results[0] == {
        "uri": "spotify://track/abc123",
        "name": "Blinding Lights",
        "artist": "The Weeknd",
    }

@pytest.mark.asyncio
async def test_search_artist_uses_name_as_artist():
    ma = MusicAssistantClient("http://ha:8123", "token", "entry-123")
    body = {
        "service_response": {
            "artists": [{
                "media_type": "artist",
                "uri": "spotify://artist/xyz",
                "name": "The Weeknd",
            }],
            "tracks": [], "albums": [], "playlists": [],
        }
    }
    with patch("pipeline.music_assistant_client.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_http_post(body)
        results = await ma.search("The Weeknd", media_type="artist")
    assert results[0]["name"] == "The Weeknd"
    assert results[0]["artist"] == "The Weeknd"

@pytest.mark.asyncio
async def test_search_empty_returns_empty_list():
    ma = MusicAssistantClient("http://ha:8123", "token", "entry-123")
    body = {
        "service_response": {
            "tracks": [], "artists": [], "albums": [], "playlists": [],
        }
    }
    with patch("pipeline.music_assistant_client.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_http_post(body)
        results = await ma.search("xyzzy404notfound", media_type="track")
    assert results == []
```

- [ ] **Step 2: Run tests**

```bash
cd /home/vertiq/ha-voice-pipeline && pytest tests/test_music_assistant_client.py -v
```

Expected: all 15 tests PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/vertiq/ha-voice-pipeline
git add tests/test_music_assistant_client.py
git commit -m "test: add search() unit tests for MusicAssistantClient"
```

---

## Task 3: Extend planner schema + prompt for music commands

**Files:**
- Modify: `pipeline/agents/planner.py`
- Modify: `tests/test_planner_accuracy.py`

- [ ] **Step 1: Add music benchmark cases to `tests/test_planner_accuracy.py`**

In `test_planner_accuracy.py`, find the `CASES = [` list and append these entries before the closing `]`:

```python
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
```

- [ ] **Step 2: Run new cases to confirm they fail**

```bash
cd /home/vertiq/ha-voice-pipeline && export $(cat config.env | xargs) && \
  pytest tests/test_planner_accuracy.py -v -s -k "music_play" 2>&1 | tail -20
```

Expected: FAIL — `music_assistant` domain not in steps (planner doesn't know about it yet).

- [ ] **Step 3: Extend `_RESPONSE_SCHEMA` in `pipeline/agents/planner.py`**

In `planner.py`, find the step item schema under `"items"` inside `"steps"` and replace it:

```python
# BEFORE
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
```

```python
# AFTER
"items": {
    "type": "object",
    "properties": {
        "domain":     {"type": "string"},
        "service":    {"type": "string"},
        "entity_id":  {"type": "string"},
        "area_id":    {"type": "string"},
        "query":      {"type": "string"},
        "artist":     {"type": "string"},
        "media_type": {"type": "string"},
    },
    "required": ["domain", "service"],
},
```

- [ ] **Step 4: Add music rules + examples to `_SYSTEM` in `pipeline/agents/planner.py`**

In `_SYSTEM`, find the line `--- EXAMPLES ---` and insert the following block **immediately before** it:

```python
# BEFORE (find this exact line)
"--- EXAMPLES ---\n"
```

```python
# AFTER (replace with this)
"MUSIC COMMANDS:\n"
"- When the user wants to play a song, artist, album, or playlist, emit ONE music_assistant.play_media step.\n"
"- entity_id: use the media_player entity with mass_player_type player from the Devices list.\n"
"- query: STT-corrected search term using your knowledge of music (e.g. 'blainding lites' → 'Blinding Lights').\n"
"- media_type: 'track' for a specific song, 'artist' for 'play X' or 'something by X', 'album' for album, 'playlist' for playlist.\n"
"- artist: only include when the user explicitly names an artist alongside a title.\n"
"- ok_response: 'Playing {query}.' — do NOT include the actual resolved track name.\n"
"- For music steps, entity_id validation is relaxed — use the best media_player from the list.\n"
"\n"
"--- EXAMPLES ---\n"
```

- [ ] **Step 5: Add music examples to `_SYSTEM` in `pipeline/agents/planner.py`**

In `_SYSTEM`, find the end of the examples section (just before the closing `"""`) and add:

```python
# Find this line (last example before closing """):
"Transcript: dim the kitchen light to fifty percent\n"
'{\"corrected\":\"dim the kitchen light to 50%\",\"intent\":\"action\",\"steps\":[{\"domain\":\"light\",\"service\":\"turn_on\",\"entity_id\":\"light.kitchen_light\",\"brightness_pct\":50}],\"ok_response\":\"Kitchen light dimmed to 50%.\",\"fail_response\":\"Sorry, I couldn\'t dim the kitchen light.\"}\n'
'"""'
```

```python
# Replace closing with additional examples then close:
"Transcript: dim the kitchen light to fifty percent\n"
'{\"corrected\":\"dim the kitchen light to 50%\",\"intent\":\"action\",\"steps\":[{\"domain\":\"light\",\"service\":\"turn_on\",\"entity_id\":\"light.kitchen_light\",\"brightness_pct\":50}],\"ok_response\":\"Kitchen light dimmed to 50%.\",\"fail_response\":\"Sorry, I couldn\'t dim the kitchen light.\"}\n'
"\n"
"Devices: media_player.respeaker_lite_media_player_2,Spotify,playing\n"
"Transcript: play blinding lights\n"
'{\"corrected\":\"play Blinding Lights\",\"intent\":\"action\",\"steps\":[{\"domain\":\"music_assistant\",\"service\":\"play_media\",\"entity_id\":\"media_player.respeaker_lite_media_player_2\",\"query\":\"Blinding Lights\",\"media_type\":\"track\"}],\"ok_response\":\"Playing Blinding Lights.\",\"already_response\":\"\",\"fail_response\":\"Sorry, I couldn\'t play that.\"}\n'
"\n"
"Transcript: play something by the weeknd\n"
'{\"corrected\":\"play something by The Weeknd\",\"intent\":\"action\",\"steps\":[{\"domain\":\"music_assistant\",\"service\":\"play_media\",\"entity_id\":\"media_player.respeaker_lite_media_player_2\",\"query\":\"The Weeknd\",\"media_type\":\"artist\"}],\"ok_response\":\"Playing The Weeknd.\",\"already_response\":\"\",\"fail_response\":\"Sorry, I couldn\'t play that.\"}\n'
"\n"
"Transcript: play hotel california by the eagles\n"
'{\"corrected\":\"play Hotel California by the Eagles\",\"intent\":\"action\",\"steps\":[{\"domain\":\"music_assistant\",\"service\":\"play_media\",\"entity_id\":\"media_player.respeaker_lite_media_player_2\",\"query\":\"Hotel California\",\"artist\":\"Eagles\",\"media_type\":\"track\"}],\"ok_response\":\"Playing Hotel California by the Eagles.\",\"already_response\":\"\",\"fail_response\":\"Sorry, I couldn\'t play that.\"}\n'
'"""'
```

- [ ] **Step 6: Skip entity validation for `music_assistant` steps in `_validate_steps`**

In `planner.py`, find `_validate_steps`. After `cleaned = []` and `for step in steps:`, add a bypass at the top of the loop body:

```python
# BEFORE
    cleaned = []
    for step in steps:
        eid = step.get("entity_id")
```

```python
# AFTER
    cleaned = []
    for step in steps:
        # Music Assistant steps: entity_id is injected by the runner — skip validation
        if step.get("domain") == "music_assistant":
            cleaned.append(step)
            continue
        eid = step.get("entity_id")
```

- [ ] **Step 7: Run music planner accuracy tests**

```bash
cd /home/vertiq/ha-voice-pipeline && export $(cat config.env | xargs) && \
  pytest tests/test_planner_accuracy.py -v -s -k "music_play" 2>&1 | tail -30
```

Expected: ≥50% pass rate on each music_play case (LLM stochastic). Look for `music_assistant` in domain and non-empty `query`.

- [ ] **Step 8: Commit**

```bash
cd /home/vertiq/ha-voice-pipeline
git add pipeline/agents/planner.py tests/test_planner_accuracy.py
git commit -m "feat: extend planner schema and prompt for music_assistant.play_media steps"
```

---

## Task 4: Add `_run_music_step` to executor

**Files:**
- Create: `tests/test_music_executor.py`
- Modify: `pipeline/agents/executor.py`

- [ ] **Step 1: Write failing tests in `tests/test_music_executor.py`**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pipeline.agents.executor import _run_music_step, execute
from pipeline.music_assistant_client import MusicAssistantClient


def _make_ma(search_results):
    ma = MagicMock(spec=MusicAssistantClient)
    ma.search = AsyncMock(return_value=search_results)
    return ma


def _make_ha(call_ok=True):
    ha = MagicMock()
    if call_ok:
        ha.call_service = AsyncMock(return_value={})
    else:
        ha.call_service = AsyncMock(side_effect=Exception("HA unreachable"))
    return ha


# ── _run_music_step ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_music_step_plays_top_result():
    step = {
        "domain": "music_assistant",
        "service": "play_media",
        "entity_id": "media_player.respeaker_lite_media_player_2",
        "query": "Blinding Lights",
        "media_type": "track",
    }
    ma = _make_ma([{"uri": "spotify://track/abc", "name": "Blinding Lights", "artist": "The Weeknd"}])
    ha = _make_ha()

    result = await _run_music_step(step, ha, ma)

    ma.search.assert_awaited_once_with("Blinding Lights", media_type="track", artist=None)
    ha.call_service.assert_awaited_once_with(
        "music_assistant", "play_media",
        entity_id="media_player.respeaker_lite_media_player_2",
        media_id="spotify://track/abc",
        media_type="track",
    )
    assert result == "Playing Blinding Lights by The Weeknd."


@pytest.mark.asyncio
async def test_music_step_no_results_returns_not_found():
    step = {
        "domain": "music_assistant",
        "service": "play_media",
        "entity_id": "media_player.respeaker_lite_media_player_2",
        "query": "xyzzy404",
        "media_type": "track",
    }
    ma = _make_ma([])
    ha = _make_ha()

    result = await _run_music_step(step, ha, ma)

    ha.call_service.assert_not_awaited()
    assert "couldn't find" in result.lower()
    assert "xyzzy404" in result


@pytest.mark.asyncio
async def test_music_step_play_failure_returns_error():
    step = {
        "domain": "music_assistant",
        "service": "play_media",
        "entity_id": "media_player.respeaker_lite_media_player_2",
        "query": "Blinding Lights",
        "media_type": "track",
    }
    ma = _make_ma([{"uri": "spotify://track/abc", "name": "Blinding Lights", "artist": "The Weeknd"}])
    ha = _make_ha(call_ok=False)

    result = await _run_music_step(step, ha, ma)

    assert "couldn't play" in result.lower()


@pytest.mark.asyncio
async def test_music_step_ma_exception_returns_error():
    step = {
        "domain": "music_assistant",
        "service": "play_media",
        "entity_id": "media_player.respeaker_lite_media_player_2",
        "query": "Blinding Lights",
        "media_type": "track",
    }
    ma = MagicMock(spec=MusicAssistantClient)
    ma.search = AsyncMock(side_effect=Exception("MA down"))
    ha = _make_ha()

    result = await _run_music_step(step, ha, ma)

    ha.call_service.assert_not_awaited()
    assert "responding" in result.lower() or "couldn't" in result.lower()


@pytest.mark.asyncio
async def test_music_step_passes_artist_to_search():
    step = {
        "domain": "music_assistant",
        "service": "play_media",
        "entity_id": "media_player.respeaker_lite_media_player_2",
        "query": "Hotel California",
        "artist": "Eagles",
        "media_type": "track",
    }
    ma = _make_ma([{"uri": "spotify://track/hc", "name": "Hotel California", "artist": "Eagles"}])
    ha = _make_ha()

    await _run_music_step(step, ha, ma)

    ma.search.assert_awaited_once_with("Hotel California", media_type="track", artist="Eagles")


@pytest.mark.asyncio
async def test_music_step_no_artist_in_result_omits_by():
    step = {
        "domain": "music_assistant",
        "service": "play_media",
        "entity_id": "media_player.respeaker_lite_media_player_2",
        "query": "chill vibes",
        "media_type": "playlist",
    }
    ma = _make_ma([{"uri": "spotify://playlist/xyz", "name": "Chill Vibes", "artist": ""}])
    ha = _make_ha()

    result = await _run_music_step(step, ha, ma)

    assert result == "Playing Chill Vibes."
    assert " by " not in result


# ── execute() branches music steps ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_routes_music_step_through_ma():
    steps = [{
        "domain": "music_assistant",
        "service": "play_media",
        "entity_id": "media_player.respeaker_lite_media_player_2",
        "query": "Blinding Lights",
        "media_type": "track",
    }]
    ma = _make_ma([{"uri": "spotify://track/abc", "name": "Blinding Lights", "artist": "The Weeknd"}])
    ha = _make_ha()
    ollama = MagicMock()

    result = await execute(
        intent="action", steps=steps, ha=ha, ollama=ollama,
        ok_response="Playing.", ma=ma,
    )

    assert "Blinding Lights" in result
    ma.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_music_step_without_ma_returns_error():
    steps = [{
        "domain": "music_assistant",
        "service": "play_media",
        "entity_id": "media_player.respeaker_lite_media_player_2",
        "query": "Blinding Lights",
        "media_type": "track",
    }]
    ha = _make_ha()
    ollama = MagicMock()

    result = await execute(
        intent="action", steps=steps, ha=ha, ollama=ollama,
        ok_response="Playing.", ma=None,
    )

    assert "not configured" in result.lower() or "sorry" in result.lower()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/vertiq/ha-voice-pipeline && pytest tests/test_music_executor.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name '_run_music_step'`

- [ ] **Step 3: Add `_run_music_step` and update `execute()` in `pipeline/agents/executor.py`**

At the top of `executor.py`, add the import after existing imports:

```python
# Add after existing imports
from pipeline.music_assistant_client import MusicAssistantClient
```

Add `_run_music_step` function before `execute()`:

```python
async def _run_music_step(
    step: dict,
    ha: HAClient,
    ma: MusicAssistantClient,
) -> str:
    """Execute one music_assistant.play_media step: search then play."""
    query      = step.get("query", "")
    artist     = step.get("artist") or None
    media_type = step.get("media_type", "track")
    entity_id  = step.get("entity_id")

    if not query:
        return "Sorry, I didn't catch what you wanted to play."

    try:
        results = await ma.search(query, media_type=media_type, artist=artist)
    except Exception as e:
        log.warning("MUSIC | search error: %s", e)
        return "Sorry, Music Assistant isn't responding right now."

    if not results:
        return f"Sorry, I couldn't find {query}."

    best       = results[0]
    uri        = best["uri"]
    track_name = best["name"]
    artist_name = best["artist"]

    try:
        await ha.call_service(
            "music_assistant", "play_media",
            entity_id=entity_id,
            media_id=uri,
            media_type=media_type,
        )
    except Exception as e:
        log.warning("MUSIC | play_media error: %s", e)
        return "Sorry, I couldn't play that right now."

    if artist_name:
        return f"Playing {track_name} by {artist_name}."
    return f"Playing {track_name}."
```

Update the `execute()` signature and add music branch at the top of the function body:

```python
# BEFORE
async def execute(
    intent: str,
    steps: list[dict],
    ha: HAClient,
    ollama: OllamaClient,
    ok_response: str = "",
    already_response: str = "",
    fail_response: str = "",
) -> str:
    # Run all steps in parallel
    results = await asyncio.gather(*[_run_step(s, ha) for s in steps])
```

```python
# AFTER
async def execute(
    intent: str,
    steps: list[dict],
    ha: HAClient,
    ollama: OllamaClient,
    ok_response: str = "",
    already_response: str = "",
    fail_response: str = "",
    ma: MusicAssistantClient | None = None,
) -> str:
    # Music steps are handled separately — branch before HA execution
    music_steps = [s for s in steps if s.get("domain") == "music_assistant"]
    if music_steps:
        if ma is None:
            return "Sorry, Music Assistant is not configured."
        return await _run_music_step(music_steps[0], ha, ma)

    # Run all HA steps in parallel
    results = await asyncio.gather(*[_run_step(s, ha) for s in steps])
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/vertiq/ha-voice-pipeline && pytest tests/test_music_executor.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Confirm existing executor tests still pass**

```bash
cd /home/vertiq/ha-voice-pipeline && pytest tests/test_agents.py -v 2>&1 | tail -15
```

Expected: all pre-existing tests PASS (no regression).

- [ ] **Step 6: Commit**

```bash
cd /home/vertiq/ha-voice-pipeline
git add pipeline/agents/executor.py tests/test_music_executor.py
git commit -m "feat: add _run_music_step to executor with search-then-play flow"
```

---

## Task 5: Update runner to accept `ma` + `satellite`, inject MA player

**Files:**
- Create: `tests/test_music_runner.py`
- Modify: `pipeline/runner.py`

- [ ] **Step 1: Write failing tests in `tests/test_music_runner.py`**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pipeline.runner import run_pipeline
from pipeline.music_assistant_client import MusicAssistantClient


def _make_ha(entities=None, areas=None):
    ha = MagicMock()
    ha.get_entities = AsyncMock(return_value=entities or [
        {"entity_id": "media_player.respeaker_lite_media_player_2",
         "name": "Spotify", "state": "playing"},
    ])
    ha.get_areas = AsyncMock(return_value=areas or [])
    return ha


def _make_ollama(plan_result):
    ollama = MagicMock()
    ollama.chat = AsyncMock(return_value=__import__("json").dumps(plan_result))
    return ollama


def _make_ma(player="media_player.respeaker_lite_media_player_2"):
    ma = MagicMock(spec=MusicAssistantClient)
    ma.resolve_player = MagicMock(return_value=player)
    ma.search = AsyncMock(return_value=[
        {"uri": "spotify://track/abc", "name": "Blinding Lights", "artist": "The Weeknd"}
    ])
    return ma


@pytest.mark.asyncio
async def test_runner_injects_satellite_player_into_music_step():
    """Runner overrides entity_id in music steps with the resolved satellite player."""
    plan_result = {
        "corrected": "play Blinding Lights",
        "intent": "action",
        "steps": [{
            "domain": "music_assistant",
            "service": "play_media",
            "entity_id": "media_player.wrong_player",   # planner got it wrong
            "query": "Blinding Lights",
            "media_type": "track",
        }],
        "ok_response": "Playing Blinding Lights.",
        "already_response": "",
        "fail_response": "Sorry.",
    }
    ha = _make_ha()
    ollama = _make_ollama(plan_result)
    ma = _make_ma("media_player.respeaker_lite_media_player_2")
    ha.call_service = AsyncMock(return_value={})

    with patch("pipeline.runner.execute") as mock_exec:
        mock_exec.return_value = "Playing Blinding Lights by The Weeknd."
        await run_pipeline("play Blinding Lights", ha, ollama, ma=ma, satellite="respeaker_lite")

    # The step passed to execute should have the resolved entity_id
    called_steps = mock_exec.call_args.kwargs["steps"]
    assert called_steps[0]["entity_id"] == "media_player.respeaker_lite_media_player_2"
    ma.resolve_player.assert_called_once_with("respeaker_lite")


@pytest.mark.asyncio
async def test_runner_passes_ma_to_execute():
    plan_result = {
        "corrected": "play Blinding Lights",
        "intent": "action",
        "steps": [{
            "domain": "music_assistant",
            "service": "play_media",
            "entity_id": "media_player.respeaker_lite_media_player_2",
            "query": "Blinding Lights",
            "media_type": "track",
        }],
        "ok_response": "Playing.", "already_response": "", "fail_response": "Sorry.",
    }
    ha = _make_ha()
    ollama = _make_ollama(plan_result)
    ma = _make_ma()

    with patch("pipeline.runner.execute") as mock_exec:
        mock_exec.return_value = "Playing Blinding Lights by The Weeknd."
        await run_pipeline("play Blinding Lights", ha, ollama, ma=ma, satellite="respeaker_lite")

    assert mock_exec.call_args.kwargs.get("ma") is ma


@pytest.mark.asyncio
async def test_runner_works_without_ma_for_non_music_commands():
    """Non-music commands still work when ma=None."""
    plan_result = {
        "corrected": "turn on the office light",
        "intent": "action",
        "steps": [{"domain": "light", "service": "turn_on",
                   "entity_id": "light.office_light"}],
        "ok_response": "The office light is on.",
        "already_response": "Already on.", "fail_response": "Failed.",
    }
    ha = _make_ha(entities=[{"entity_id": "light.office_light", "name": "Office Light", "state": "off"}])
    ha.call_service = AsyncMock(return_value={})
    ha.get_state = AsyncMock(return_value={"entity_id": "light.office_light", "state": "on", "attributes": {}})
    ollama = _make_ollama(plan_result)

    result = await run_pipeline("turn on the office light", ha, ollama, ma=None, satellite=None)

    assert isinstance(result, str) and len(result) > 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/vertiq/ha-voice-pipeline && pytest tests/test_music_runner.py -v 2>&1 | head -20
```

Expected: `TypeError: run_pipeline() got an unexpected keyword argument 'ma'`

- [ ] **Step 3: Update `pipeline/runner.py`**

Change the `run_pipeline` signature and add MA player injection:

```python
# BEFORE
async def run_pipeline(transcript: str, ha: HAClient, ollama: OllamaClient) -> str:
```

```python
# AFTER
async def run_pipeline(
    transcript: str,
    ha: HAClient,
    ollama: OllamaClient,
    ma=None,
    satellite: str | None = None,
) -> str:
```

After the line `if not planned.get("steps"):` block (but before the fast-path query check), add:

```python
    # Inject satellite's resolved MA player into music steps (overrides LLM's choice)
    if ma is not None:
        ma_player = ma.resolve_player(satellite)
        if ma_player:
            for step in planned["steps"]:
                if step.get("domain") == "music_assistant":
                    step["entity_id"] = ma_player
```

Pass `ma` through to `execute()`. Find the `return await execute(` call and update it:

```python
# BEFORE
    return await execute(
        intent=planned["intent"],
        steps=planned["steps"],
        ha=ha,
        ollama=ollama,
        ok_response=planned.get("ok_response", ""),
        already_response=planned.get("already_response", ""),
        fail_response=planned.get("fail_response", ""),
    )
```

```python
# AFTER
    return await execute(
        intent=planned["intent"],
        steps=planned["steps"],
        ha=ha,
        ollama=ollama,
        ok_response=planned.get("ok_response", ""),
        already_response=planned.get("already_response", ""),
        fail_response=planned.get("fail_response", ""),
        ma=ma,
    )
```

- [ ] **Step 4: Run tests**

```bash
cd /home/vertiq/ha-voice-pipeline && pytest tests/test_music_runner.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Confirm existing pipeline tests still pass**

```bash
cd /home/vertiq/ha-voice-pipeline && export $(cat config.env | xargs) && \
  pytest tests/test_pipeline.py -v 2>&1 | tail -15
```

Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/vertiq/ha-voice-pipeline
git add pipeline/runner.py tests/test_music_runner.py
git commit -m "feat: runner accepts satellite param and injects MA player into music steps"
```

---

## Task 6: Update server — init `_ma`, startup discovery, `?satellite=` param

**Files:**
- Modify: `pipeline/agents/server.py`
- Create: `tests/test_music_server.py`

- [ ] **Step 1: Write failing tests in `tests/test_music_server.py`**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("pipeline.agents.server.run_pipeline") as mock_run, \
         patch("pipeline.agents.server._ma") as mock_ma:
        mock_run.return_value = "Playing Blinding Lights by The Weeknd."
        mock_ma.discover = AsyncMock()
        from pipeline.agents.server import app
        yield TestClient(app), mock_run


def test_satellite_param_passed_to_run_pipeline(client):
    tc, mock_run = client
    tc.post(
        "/v1/chat/completions?satellite=respeaker_lite",
        json={"messages": [{"role": "user", "content": "play Blinding Lights"}]},
    )
    _, kwargs = mock_run.call_args
    assert kwargs.get("satellite") == "respeaker_lite"


def test_no_satellite_param_passes_none(client):
    tc, mock_run = client
    tc.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "turn on the light"}]},
    )
    _, kwargs = mock_run.call_args
    assert kwargs.get("satellite") is None


def test_response_contains_pipeline_output(client):
    tc, mock_run = client
    r = tc.post(
        "/v1/chat/completions?satellite=respeaker_lite",
        json={"messages": [{"role": "user", "content": "play Blinding Lights"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Playing Blinding Lights by The Weeknd."
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/vertiq/ha-voice-pipeline && pytest tests/test_music_server.py -v 2>&1 | head -20
```

Expected: FAIL — `run_pipeline` is not called with `satellite` kwarg.

- [ ] **Step 3: Update `pipeline/agents/server.py`**

Add `MusicAssistantClient` import after existing imports:

```python
from pipeline.music_assistant_client import MusicAssistantClient
```

After the `_ollama = OllamaClient(...)` block, add `_ma` init:

```python
_ma = MusicAssistantClient(
    os.getenv("HA_URL",            ""),
    os.getenv("HA_TOKEN",          ""),
    os.getenv("MA_CONFIG_ENTRY_ID", ""),
)
```

In the existing `_startup_warmup` function, add MA discovery after the Ollama warmup try/except:

```python
# Add at the end of _startup_warmup, after the existing try/except block:
    try:
        await _ma.discover()
    except Exception as e:
        log.warning("MUSIC | discovery failed at startup: %s", e)
```

In `chat_completions`, read the `satellite` query param and pass it to `run_pipeline`:

```python
# BEFORE
    text = await run_pipeline(transcript, _ha, _ollama)
```

```python
# AFTER
    satellite = request.query_params.get("satellite")
    text = await run_pipeline(transcript, _ha, _ollama, ma=_ma, satellite=satellite)
```

- [ ] **Step 4: Run tests**

```bash
cd /home/vertiq/ha-voice-pipeline && pytest tests/test_music_server.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/vertiq/ha-voice-pipeline
git add pipeline/agents/server.py tests/test_music_server.py
git commit -m "feat: server reads ?satellite= param and initializes MusicAssistantClient on startup"
```

---

## Task 7: Config files

**Files:**
- Modify: `config.env.example`
- Modify: `config.env`

- [ ] **Step 1: Update `config.env.example`**

```bash
# BEFORE (last line of config.env.example)
ZIGBEE_PROPAGATION_MS=400
```

```bash
# AFTER
ZIGBEE_PROPAGATION_MS=400
MA_CONFIG_ENTRY_ID=your_music_assistant_config_entry_id_here
```

- [ ] **Step 2: Add `MA_CONFIG_ENTRY_ID` to `config.env`**

```bash
# BEFORE (last line of config.env)
ZIGBEE_PROPAGATION_MS=400
```

```bash
# AFTER
ZIGBEE_PROPAGATION_MS=400
MA_CONFIG_ENTRY_ID=01KNSX10VSWX7AB1N449NJ0AQB
```

- [ ] **Step 3: Rebuild and restart the container**

```bash
cd /home/vertiq/ha-voice-pipeline
docker compose down && docker compose up -d --build
docker logs ha-voice-pipeline --follow 2>&1 | head -30
```

Expected log lines include:
```
MUSIC | discovered 2 satellite player(s): {'respeaker_lite': 'media_player.respeaker_lite_media_player_2', 'home_assistant_voice_09d0e0': 'media_player.home_assistant_voice_media_player'}
```

- [ ] **Step 4: Run full test suite**

```bash
cd /home/vertiq/ha-voice-pipeline && export $(cat config.env | xargs) && \
  pytest tests/ -v --ignore=tests/test_planner_accuracy.py --ignore=tests/test_latency.py 2>&1 | tail -30
```

Expected: all unit tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/vertiq/ha-voice-pipeline
git add config.env.example config.env
git commit -m "config: add MA_CONFIG_ENTRY_ID for Music Assistant integration"
```

---

## HA Configuration (manual step — not automated)

After deployment, update each satellite's conversation agent URL in HA:

- **ReSpeaker Lite pipeline:** `http://<pipeline-host>:18795/v1/chat/completions?satellite=respeaker_lite`
- **HA Voice pipeline:** `http://<pipeline-host>:18795/v1/chat/completions?satellite=home_assistant_voice_09d0e0`

This is done in HA → Settings → Voice Assistants → [each pipeline] → Conversation Agent → OpenAI URL.

---

## Self-Review

**Spec coverage check:**
- ✅ Satellite auto-discovery via HA states (Tasks 1–2)
- ✅ `MA_CONFIG_ENTRY_ID` only env var needed (Task 7)
- ✅ `?satellite=` query param routing (Task 6)
- ✅ Planner schema + prompt for music steps with `query`/`artist`/`media_type` (Task 3)
- ✅ STT error correction via LLM knowledge (Task 3 — prompt instructs LLM)
- ✅ Executor: search → play → spoken confirmation with actual track name (Task 4)
- ✅ Error responses for no-results, play failure, MA unreachable (Task 4)
- ✅ `_validate_steps` skips music_assistant domain (Task 3)
- ✅ Fallback to first MA player when `satellite` param absent (Task 1)

**No placeholders found.**

**Type consistency:**
- `MusicAssistantClient.search()` returns `list[dict]` with keys `uri`, `name`, `artist` — used consistently in `_run_music_step`
- `execute()` new `ma` kwarg is `MusicAssistantClient | None` — matches usage in `runner.py` and tests
- `run_pipeline()` new params `ma` and `satellite` match server call and runner tests
