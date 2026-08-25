# Production Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the pipeline from "working home lab project" to production-ready: connection pooling, observability endpoints, input validation, optional auth, and complete test coverage.

**Architecture:** No structural changes to the pipeline flow. All changes are within existing components — client internals (connection pooling), server endpoints (new health/status/reload), and test completeness. The public HTTP API contract is preserved.

**Tech Stack:** Python 3.12, FastAPI 0.115, httpx 0.27, pytest-asyncio, FastAPI TestClient (starlette)

---

## File Map

| File | Change |
|---|---|
| `requirements-dev.txt` | Add `python-dotenv>=1.0.0` |
| `pipeline/ha_client.py` | Add `self._client`, `async def close()` |
| `pipeline/ollama_client.py` | Add `self._client`, `async def close()` |
| `pipeline/music_assistant_client.py` | Add `self._client`, `async def close()`, update `_discover_satellite_players` signature |
| `pipeline/server.py` | Lifespan cleanup, input validation, `/health/deep`, `/status`, `/reload`, API key auth, `_resolve_ollama_url()` |
| `tests/test_music_assistant_client.py` | Update mock pattern (no more `patch(httpx.AsyncClient)`) |
| `tests/test_pipeline.py` | Remove duplicate `ha()` / `ollama()` fixture definitions |
| `tests/test_agents.py` | Replace hardcoded `light.kitchen_ceiling` with dynamic entity |
| `tests/test_server_contract.py` | New: contract + error path tests for all HTTP endpoints |

---

### Task 1: Add python-dotenv to requirements-dev.txt

**Files:**
- Modify: `requirements-dev.txt`

`conftest.py` imports `from dotenv import load_dotenv` but `python-dotenv` is absent from the dev install manifest. `pip install -r requirements-dev.txt` leaves the package missing and all tests fail at collection.

- [ ] **Step 1.1: Verify the gap**

```bash
grep -r "dotenv" tests/ requirements*.txt
```

Expected: `conftest.py` imports it; neither `requirements.txt` nor `requirements-dev.txt` list it.

- [ ] **Step 1.2: Add the dependency**

Edit `requirements-dev.txt` to:

```text
-r requirements.txt
pytest==8.3.2
pytest-asyncio==0.24.0
websockets>=12.0
python-dotenv>=1.0.0
```

- [ ] **Step 1.3: Verify tests collect cleanly**

```bash
python3 -m pytest tests/ --co -q --ignore=tests/test_planner_accuracy.py --ignore=tests/test_latency.py 2>&1 | tail -5
```

Expected: `N tests collected` with no import errors.

- [ ] **Step 1.4: Commit**

```bash
git add requirements-dev.txt
git commit -m "fix(deps): add python-dotenv to requirements-dev.txt"
```

---

### Task 2: Connection pooling — HAClient

**Files:**
- Modify: `pipeline/ha_client.py`

Currently every method does `async with httpx.AsyncClient() as c:` which performs a full TCP setup/teardown per call. Each pipeline command makes 4–6 HA calls. Storing a long-lived `AsyncClient` eliminates this overhead (10–50ms per request).

- [ ] **Step 2.1: Write a failing unit test for pooling**

Add to `tests/test_ha_client.py`:

```python
def test_ha_client_reuses_http_client():
    """HAClient must share one httpx.AsyncClient across calls, not create one per call."""
    from pipeline.ha_client import HAClient
    ha = HAClient("http://test", "token")
    assert hasattr(ha, "_client"), "HAClient must store an httpx.AsyncClient instance"
    import httpx
    assert isinstance(ha._client, httpx.AsyncClient)
```

- [ ] **Step 2.2: Run to confirm it fails**

```bash
python3 -m pytest tests/test_ha_client.py::test_ha_client_reuses_http_client -v
```

Expected: FAILED — `AssertionError: HAClient must store an httpx.AsyncClient instance`

- [ ] **Step 2.3: Rewrite `pipeline/ha_client.py`**

```python
import httpx

CONTROLLABLE_DOMAINS = {"light", "switch", "fan", "media_player", "climate", "cover", "input_boolean"}


class HAClient:
    def __init__(self, ha_url: str, token: str):
        self._url = ha_url.rstrip("/")
        self._hdrs = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        self._client = httpx.AsyncClient()

    async def close(self) -> None:
        await self._client.aclose()

    async def get_entities(self) -> list[dict]:
        r = await self._client.get(f"{self._url}/api/states", headers=self._hdrs, timeout=10)
        r.raise_for_status()
        raw = [
            {
                "entity_id": s["entity_id"],
                "name": s["attributes"].get("friendly_name", s["entity_id"]),
                "state": s["state"],
                "_attr_count": len(s.get("attributes", {})),
            }
            for s in r.json()
            if s["entity_id"].split(".")[0] in CONTROLLABLE_DOMAINS
            and s["state"] != "unavailable"
        ]
        by_name: dict[str, dict] = {}
        for e in raw:
            existing = by_name.get(e["name"])
            if existing is None or e["_attr_count"] > existing["_attr_count"]:
                by_name[e["name"]] = e
        return [{"entity_id": e["entity_id"], "name": e["name"], "state": e["state"]}
                for e in by_name.values()]

    async def get_areas(self) -> list[dict]:
        _TMPL = (
            '{% set r = namespace(a=[]) %}'
            '{% for aid in areas() %}'
            '{% set r.a = r.a + [{"area_id": aid, "name": area_name(aid)}] %}'
            '{% endfor %}{{ r.a | tojson }}'
        )
        r = await self._client.post(
            f"{self._url}/api/template",
            headers=self._hdrs,
            json={"template": _TMPL},
            timeout=10,
        )
        if r.status_code in (404, 400):
            return []
        r.raise_for_status()
        import json
        return json.loads(r.text)

    async def get_state(self, entity_id: str) -> dict:
        r = await self._client.get(
            f"{self._url}/api/states/{entity_id}",
            headers=self._hdrs,
            timeout=10,
        )
        r.raise_for_status()
        s = r.json()
        return {
            "entity_id": s["entity_id"],
            "state": s["state"],
            "attributes": s.get("attributes", {}),
        }

    async def call_service(
        self,
        domain: str,
        service: str,
        entity_id: str | list[str] | None = None,
        *,
        area_id: str | None = None,
        **kwargs,
    ) -> dict:
        payload: dict = {**kwargs}
        if entity_id is not None:
            payload["entity_id"] = entity_id
        if area_id is not None:
            payload["area_id"] = area_id
        r = await self._client.post(
            f"{self._url}/api/services/{domain}/{service}",
            headers=self._hdrs,
            json=payload,
            timeout=15,
        )
        r.raise_for_status()
        return r.json() if r.content else {}
```

- [ ] **Step 2.4: Run the new test and all existing HA client tests**

```bash
python3 -m pytest tests/test_ha_client.py -v
```

Expected: all pass (live integration tests still hit real HA; the unit test passes).

- [ ] **Step 2.5: Run full unit suite to confirm no regressions**

```bash
python3 -m pytest tests/ --ignore=tests/test_planner_accuracy.py --ignore=tests/test_latency.py -q
```

Expected: all pass.

- [ ] **Step 2.6: Commit**

```bash
git add pipeline/ha_client.py tests/test_ha_client.py
git commit -m "perf(ha_client): reuse httpx.AsyncClient — connection pooling"
```

---

### Task 3: Connection pooling — OllamaClient

**Files:**
- Modify: `pipeline/ollama_client.py`

Same pattern as Task 2. OllamaClient makes one call per pipeline command but warmup calls also hit it; pooling avoids repeated TLS on every request.

- [ ] **Step 3.1: Write a failing unit test**

Add to `tests/test_ollama_client.py`:

```python
def test_ollama_client_reuses_http_client():
    """OllamaClient must store one httpx.AsyncClient instance."""
    from pipeline.ollama_client import OllamaClient
    ollama = OllamaClient("http://test:11434", "model")
    import httpx
    assert isinstance(ollama._client, httpx.AsyncClient)
```

- [ ] **Step 3.2: Run to confirm it fails**

```bash
python3 -m pytest tests/test_ollama_client.py::test_ollama_client_reuses_http_client -v
```

Expected: FAILED.

- [ ] **Step 3.3: Rewrite `pipeline/ollama_client.py`**

```python
import json, httpx
from typing import Callable

class OllamaClient:
    def __init__(self, ollama_url: str, model: str):
        self.url   = ollama_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(timeout=60)

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(self, system: str, user: str, format: dict | str | None = None) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        body: dict = {"model": self.model, "messages": messages,
                      "stream": False, "think": False}
        if format is not None:
            body["format"] = format
        r = await self._client.post(f"{self.url}/api/chat", json=body)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    async def chat_with_tools(
        self,
        system: str,
        user: str,
        tools: list[dict],
        tool_handler: Callable[[str, dict], str],
        max_rounds: int = 6,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        for _ in range(max_rounds):
            r = await self._client.post(
                f"{self.url}/api/chat",
                json={"model": self.model, "messages": messages,
                      "tools": tools, "stream": False, "think": False},
            )
            r.raise_for_status()
            msg = r.json()["message"]
            messages.append(msg)

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                return msg.get("content", "").strip()

            for tc in tool_calls:
                fn   = tc["function"]
                name = fn["name"]
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                result = tool_handler(name, args)
                messages.append({
                    "role":    "tool",
                    "content": result if isinstance(result, str) else json.dumps(result),
                })

        return messages[-1].get("content", "").strip()
```

- [ ] **Step 3.4: Run tests**

```bash
python3 -m pytest tests/test_ollama_client.py -v
```

Expected: all pass.

- [ ] **Step 3.5: Commit**

```bash
git add pipeline/ollama_client.py tests/test_ollama_client.py
git commit -m "perf(ollama_client): reuse httpx.AsyncClient — connection pooling"
```

---

### Task 4: Connection pooling — MusicAssistantClient

**Files:**
- Modify: `pipeline/music_assistant_client.py`
- Modify: `tests/test_music_assistant_client.py`

The module-level `_discover_satellite_players` function currently creates its own `httpx.AsyncClient`. After this change it accepts the client from the instance, so existing tests that patch `httpx.AsyncClient` need updating.

- [ ] **Step 4.1: Write a failing unit test**

Add to `tests/test_music_assistant_client.py`:

```python
def test_ma_client_reuses_http_client():
    """MusicAssistantClient must store one httpx.AsyncClient instance."""
    from pipeline.music_assistant_client import MusicAssistantClient
    import httpx
    ma = MusicAssistantClient("http://test", "token", "entry")
    assert isinstance(ma._client, httpx.AsyncClient)
```

- [ ] **Step 4.2: Run to confirm it fails**

```bash
python3 -m pytest tests/test_music_assistant_client.py::test_ma_client_reuses_http_client -v
```

Expected: FAILED.

- [ ] **Step 4.3: Rewrite `pipeline/music_assistant_client.py`**

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


async def _discover_satellite_players(
    url: str, hdrs: dict, client: httpx.AsyncClient
) -> dict[str, str]:
    """Build {satellite_slug: ma_player_entity_id} from HA /api/states."""
    r = await client.get(f"{url}/api/states", headers=hdrs, timeout=10)
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

    for ma in ma_players:
        aq = ma.get("attributes", {}).get("active_queue") or ""
        if aq.startswith("media_player."):
            slug = _physical_player_slug(aq)
            if slug in satellite_slugs:
                result[slug] = ma["entity_id"]

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
        self._client = httpx.AsyncClient()

    async def close(self) -> None:
        await self._client.aclose()

    async def discover(self) -> None:
        """Populate satellite→player map from HA. Call once at startup."""
        self._satellite_map = await _discover_satellite_players(
            self._url, self._hdrs, self._client
        )
        log.info(
            "MUSIC | discovered %d satellite player(s): %s",
            len(self._satellite_map),
            self._satellite_map,
        )

    def resolve_player(self, satellite_slug: str | None) -> str | None:
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
        payload: dict = {
            "config_entry_id": self._config_entry_id,
            "name": name,
            "media_type": [media_type],
            "limit": limit,
        }
        if artist:
            payload["artist"] = artist
        r = await self._client.post(
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

- [ ] **Step 4.4: Update `tests/test_music_assistant_client.py` mock pattern**

The two tests that used `patch("pipeline.music_assistant_client.httpx.AsyncClient")` now pass a mock client directly. Replace the `_mock_http` and `_mock_http_post` helpers and their test usages:

```python
def _mock_http(states):
    mock_resp = MagicMock()
    mock_resp.json.return_value = states
    mock_resp.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    return mock_client


def _mock_http_post(response_body):
    mock_resp = MagicMock()
    mock_resp.json.return_value = response_body
    mock_resp.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    return mock_client
```

Update `test_discover_maps_both_satellites`:

```python
@pytest.mark.asyncio
async def test_discover_maps_both_satellites():
    mock_client = _mock_http(_MOCK_STATES)
    result = await _discover_satellite_players(
        "http://ha:8123", {"Authorization": "Bearer x"}, mock_client
    )
    assert result["respeaker_lite"] == "media_player.respeaker_lite_media_player_2"
    assert result["home_assistant_voice_09d0e0"] == "media_player.home_assistant_voice_media_player"
```

Update `test_discover_fallback_when_active_queue_null`:

```python
@pytest.mark.asyncio
async def test_discover_fallback_when_active_queue_null():
    states = []
    for s in _MOCK_STATES:
        s2 = {"entity_id": s["entity_id"], "state": s["state"],
              "attributes": dict(s["attributes"])}
        if s2["attributes"].get("mass_player_type") == "player":
            s2["attributes"]["active_queue"] = None
        states.append(s2)
    mock_client = _mock_http(states)
    result = await _discover_satellite_players(
        "http://ha:8123", {"Authorization": "Bearer x"}, mock_client
    )
    assert result.get("respeaker_lite") == "media_player.respeaker_lite_media_player_2"
```

Update `test_search_track_returns_name_and_artist`:

```python
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
    ma._client = _mock_http_post(body)
    results = await ma.search("Blinding Lights", media_type="track")
    assert len(results) == 1
    assert results[0] == {
        "uri": "spotify://track/abc123",
        "name": "Blinding Lights",
        "artist": "The Weeknd",
    }
```

Update `test_search_artist_uses_name_as_artist`:

```python
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
    ma._client = _mock_http_post(body)
    results = await ma.search("The Weeknd", media_type="artist")
    assert results[0]["name"] == "The Weeknd"
    assert results[0]["artist"] == "The Weeknd"
```

Update `test_search_empty_returns_empty_list`:

```python
@pytest.mark.asyncio
async def test_search_empty_returns_empty_list():
    ma = MusicAssistantClient("http://ha:8123", "token", "entry-123")
    body = {
        "service_response": {
            "tracks": [], "artists": [], "albums": [], "playlists": [],
        }
    }
    ma._client = _mock_http_post(body)
    results = await ma.search("xyzzy404notfound", media_type="track")
    assert results == []
```

Remove the two `patch("pipeline.music_assistant_client.httpx.AsyncClient")` context managers — they are no longer needed.

- [ ] **Step 4.5: Run tests**

```bash
python3 -m pytest tests/test_music_assistant_client.py -v
```

Expected: all pass.

- [ ] **Step 4.6: Commit**

```bash
git add pipeline/music_assistant_client.py tests/test_music_assistant_client.py
git commit -m "perf(music_assistant_client): reuse httpx.AsyncClient — connection pooling"
```

---

### Task 5: Lifespan client cleanup

**Files:**
- Modify: `pipeline/server.py`

After connection pooling, each client holds an open `httpx.AsyncClient`. The lifespan teardown (after `yield`) must close all three clients so the event loop doesn't warn about unclosed connections on shutdown.

- [ ] **Step 5.1: Write a failing test**

Add to `tests/test_server_contract.py` (creating the file if it doesn't exist yet):

```python
"""Contract and error-path tests for pipeline/server.py HTTP endpoints."""
import os
import pytest
from unittest.mock import AsyncMock, patch

# Set env vars before importing server (module-level init reads them)
os.environ.setdefault("HA_URL",     "http://test-ha:8123")
os.environ.setdefault("OLLAMA_URL", "http://test-ollama:11434")
os.environ.setdefault("HA_TOKEN",   "test-token")
os.environ.setdefault("MODEL",      "test-model")

from fastapi.testclient import TestClient
from pipeline.server import app


def test_health_endpoint_returns_ok():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
```

- [ ] **Step 5.2: Run to see it pass (health works without lifespan)**

```bash
python3 -m pytest tests/test_server_contract.py::test_health_endpoint_returns_ok -v
```

Expected: PASS (this baseline test passes before lifespan cleanup).

- [ ] **Step 5.3: Update server.py lifespan to close clients**

In `pipeline/server.py`, update the `_lifespan` function. Change the `yield` block to:

```python
@asynccontextmanager
async def _lifespan(app: FastAPI):
    """One-time warmup on startup. GPU clock is locked by systemd nvidia-clocks.service,
    so no periodic keepalive needed — wyoming-voice fires AudioStart warmup per command."""
    if not os.getenv("HA_URL"):
        raise RuntimeError("HA_URL environment variable is required")
    if not os.getenv("OLLAMA_URL"):
        raise RuntimeError("OLLAMA_URL environment variable is required")
    import httpx
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            await c.post(f"{_ollama.url}/api/chat",
                         json={"model": _ollama.model, "messages": [], "keep_alive": -1,
                               "options": {"num_ctx": 8192}})
    except Exception:
        pass
    try:
        await _ma.discover()
        log.info("MUSIC | satellite_map: %s", _ma._satellite_map)
    except Exception as e:
        log.warning("MUSIC | discovery failed at startup: %s", e)
    yield
    # Graceful shutdown: close pooled HTTP clients
    await _ha.close()
    await _ollama.close()
    await _ma.close()
    if _spotify_sync is not None and _spotify_sync._client is not None:
        await _spotify_sync._client.aclose()
```

- [ ] **Step 5.4: Run the suite**

```bash
python3 -m pytest tests/ --ignore=tests/test_planner_accuracy.py --ignore=tests/test_latency.py -q
```

Expected: all pass.

- [ ] **Step 5.5: Commit**

```bash
git add pipeline/server.py tests/test_server_contract.py
git commit -m "feat(server): close pooled httpx clients on lifespan shutdown"
```

---

### Task 6: Input validation — transcript length

**Files:**
- Modify: `pipeline/server.py`
- Modify: `tests/test_server_contract.py`

Without a length limit, a 50 KB ambient speech capture triggers a full LLM call. Cap transcripts at 500 characters; return HTTP 400.

- [ ] **Step 6.1: Write a failing test**

Add to `tests/test_server_contract.py`:

```python
def test_transcript_too_long_returns_400():
    """Transcripts longer than 500 chars must be rejected with HTTP 400."""
    client = TestClient(app)
    long_text = "x" * 501
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": long_text}]
    })
    assert r.status_code == 400
    assert "too long" in r.json().get("error", "").lower()
```

- [ ] **Step 6.2: Run to confirm it fails**

```bash
python3 -m pytest tests/test_server_contract.py::test_transcript_too_long_returns_400 -v
```

Expected: FAILED — gets 200 not 400.

- [ ] **Step 6.3: Add validation to `chat_completions` in server.py**

In `pipeline/server.py`, inside `async def chat_completions(request: Request):`, add the length check immediately after extracting `transcript`:

```python
    if not transcript:
        return JSONResponse({"error": "no user message"}, status_code=400)

    _MAX_TRANSCRIPT_CHARS = 500
    if len(transcript) > _MAX_TRANSCRIPT_CHARS:
        return JSONResponse({"error": "transcript too long"}, status_code=400)
```

- [ ] **Step 6.4: Run test**

```bash
python3 -m pytest tests/test_server_contract.py::test_transcript_too_long_returns_400 -v
```

Expected: PASS.

- [ ] **Step 6.5: Commit**

```bash
git add pipeline/server.py tests/test_server_contract.py
git commit -m "feat(server): reject transcripts longer than 500 chars with HTTP 400"
```

---

### Task 7: Fix test_pipeline.py duplicate fixtures

**Files:**
- Modify: `tests/test_pipeline.py`

`tests/test_pipeline.py` redefines module-scoped `ha()` and `ollama()` fixtures that already exist as session-scoped in `tests/conftest.py`. This shadows the session fixtures, meaning the live tests create extra client instances and don't share the session. Remove the local definitions.

- [ ] **Step 7.1: Write a failing test**

Add to `tests/test_server_contract.py`:

```python
def test_test_pipeline_has_no_local_fixtures():
    """test_pipeline.py must not redefine ha() or ollama() — conftest.py owns them."""
    import ast, pathlib
    src = pathlib.Path("tests/test_pipeline.py").read_text()
    tree = ast.parse(src)
    fixture_names = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            (isinstance(d, ast.Attribute) and d.attr == "fixture")
            or (isinstance(d, ast.Name) and d.id == "fixture")
            for d in node.decorator_list
        )
    ]
    assert "ha" not in fixture_names, "ha() fixture redefined in test_pipeline.py — remove it"
    assert "ollama" not in fixture_names, "ollama() fixture redefined in test_pipeline.py — remove it"
```

- [ ] **Step 7.2: Run to confirm it fails**

```bash
python3 -m pytest tests/test_server_contract.py::test_test_pipeline_has_no_local_fixtures -v
```

Expected: FAILED.

- [ ] **Step 7.3: Edit `tests/test_pipeline.py`**

Remove these lines from `tests/test_pipeline.py` (the four lines that define `ha()` and `ollama()` fixtures at module level, along with the `HA_URL`/`HA_TOKEN`/`OLLAMA_URL`/`MODEL` constants that duplicate conftest):

```python
# DELETE these lines:
HA_URL     = os.getenv("HA_URL",     "http://homeassistant.local:8123")
HA_TOKEN   = os.getenv("HA_TOKEN",   "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://homeassistant.local:11434")
MODEL      = os.getenv("MODEL",      "default")

@pytest.fixture(scope="module")
def ha():
    return HAClient(HA_URL, HA_TOKEN)

@pytest.fixture(scope="module")
def ollama():
    return OllamaClient(OLLAMA_URL, MODEL)
```

Also remove the now-unused imports at the top of `test_pipeline.py`:

```python
# KEEP only:
import os, pytest
from pipeline.runner import run_pipeline
# REMOVE: from pipeline.ha_client import HAClient
# REMOVE: from pipeline.ollama_client import OllamaClient
```

- [ ] **Step 7.4: Run test**

```bash
python3 -m pytest tests/test_server_contract.py::test_test_pipeline_has_no_local_fixtures -v
python3 -m pytest tests/test_pipeline.py -v
```

Expected: both pass.

- [ ] **Step 7.5: Commit**

```bash
git add tests/test_pipeline.py tests/test_server_contract.py
git commit -m "fix(tests): remove duplicate ha/ollama fixtures from test_pipeline.py"
```

---

### Task 8: Fix hardcoded entity IDs in test_agents.py

**Files:**
- Modify: `tests/test_agents.py`

`light.kitchen_ceiling` appears on lines 65 and 78. This entity is specific to one home setup and fails on any other HA instance. Replace with a dynamically-resolved entity from the live `ha_context` fixture.

- [ ] **Step 8.1: Write a failing test (self-referential)**

Add to `tests/test_server_contract.py`:

```python
def test_test_agents_has_no_hardcoded_entity_ids():
    """test_agents.py must not reference specific entity IDs like light.kitchen_ceiling."""
    import pathlib
    src = pathlib.Path("tests/test_agents.py").read_text()
    assert "light.kitchen_ceiling" not in src, (
        "Hardcoded entity_id 'light.kitchen_ceiling' found in test_agents.py — use ha_context instead"
    )
```

- [ ] **Step 8.2: Run to confirm it fails**

```bash
python3 -m pytest tests/test_server_contract.py::test_test_agents_has_no_hardcoded_entity_ids -v
```

Expected: FAILED.

- [ ] **Step 8.3: Edit `tests/test_agents.py`**

Replace the two hardcoded-entity tests. The original `test_execute_query_returns_state` and `test_execute_action_turn_on` use `"light.kitchen_ceiling"`. Rewrite them to resolve a light entity dynamically from `ha_context`:

```python
@pytest.mark.asyncio(loop_scope="session")
async def test_execute_query_returns_state(ha, ollama, ha_context):
    lights = [e for e in ha_context["entities"] if e["entity_id"].startswith("light.")]
    if not lights:
        pytest.skip("No light entities available in this HA instance")
    eid = lights[0]["entity_id"]
    steps = [{"domain": "light", "service": "get_state", "entity_id": eid}]
    result = await execute(intent="query", steps=steps, ha=ha, ollama=ollama,
                           ok_response=f"The {lights[0]['name']} state was checked.",
                           fail_response="Could not read state.")
    assert isinstance(result, str) and len(result) > 0


@pytest.mark.asyncio(loop_scope="session")
async def test_execute_action_turn_on(ha, ollama, ha_context):
    lights = [e for e in ha_context["entities"] if e["entity_id"].startswith("light.")]
    if not lights:
        pytest.skip("No light entities available in this HA instance")
    eid = lights[0]["entity_id"]
    try:
        initial = (await ha.get_state(eid))["state"]
    except Exception:
        initial = None

    steps = [{"domain": "light", "service": "turn_on", "entity_id": eid}]
    try:
        result = await execute(intent="action", steps=steps, ha=ha, ollama=ollama,
                               ok_response=f"The {lights[0]['name']} is now on.",
                               fail_response=f"Sorry, I couldn't reach the {lights[0]['name']}.")
        assert isinstance(result, str) and len(result) > 0
    finally:
        if initial is not None:
            svc = "turn_on" if initial == "on" else "turn_off"
            try:
                await ha.call_service("light", svc, entity_id=eid)
            except Exception:
                pass
```

- [ ] **Step 8.4: Run tests**

```bash
python3 -m pytest tests/test_server_contract.py::test_test_agents_has_no_hardcoded_entity_ids -v
python3 -m pytest tests/test_agents.py -v
```

Expected: both pass.

- [ ] **Step 8.5: Commit**

```bash
git add tests/test_agents.py tests/test_server_contract.py
git commit -m "fix(tests): replace hardcoded light.kitchen_ceiling with dynamic entity resolution"
```

---

### Task 9: Deep health endpoint — GET /health/deep

**Files:**
- Modify: `pipeline/server.py`
- Modify: `tests/test_server_contract.py`

The current `/health` returns OK regardless of HA/Ollama reachability. Add `/health/deep` that probes both services with a 4-second timeout and reports partial degradation. The shallow `/health` is unchanged (used by Docker healthcheck).

- [ ] **Step 9.1: Write a failing contract test**

Add to `tests/test_server_contract.py`:

```python
def test_health_deep_returns_expected_shape():
    """GET /health/deep must return status and per-component health."""
    client = TestClient(app)
    with patch("pipeline.server._ha") as mock_ha, \
         patch("pipeline.server._ollama") as mock_ollama:
        mock_ha._url = "http://test"
        mock_ha._hdrs = {}
        mock_ollama.url = "http://test"
        # Simulate both services unreachable (connection error)
        with patch("httpx.AsyncClient") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.get = AsyncMock(side_effect=Exception("unreachable"))
            mock_cls.return_value = mock_instance
            r = client.get("/health/deep")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert body["status"] in ("ok", "degraded", "down")
    assert "components" in body
    assert "ha" in body["components"]
    assert "ollama" in body["components"]
    for comp in body["components"].values():
        assert "status" in comp
        assert comp["status"] in ("ok", "error")
```

- [ ] **Step 9.2: Run to confirm it fails**

```bash
python3 -m pytest tests/test_server_contract.py::test_health_deep_returns_expected_shape -v
```

Expected: FAILED — 404 (route doesn't exist yet).

- [ ] **Step 9.3: Add `GET /health/deep` to server.py**

Add after the existing `/health` route in `pipeline/server.py`:

```python
@app.get("/health/deep")
async def health_deep():
    """Probe HA and Ollama reachability. Always returns HTTP 200; check body status."""
    import httpx as _httpx
    results: dict = {}

    async def _probe(name: str, url: str, headers: dict = {}) -> None:
        try:
            async with _httpx.AsyncClient(timeout=4) as c:
                r = await c.get(url, headers=headers)
                r.raise_for_status()
            results[name] = {"status": "ok"}
        except Exception as exc:
            results[name] = {"status": "error", "detail": str(exc)[:120]}

    import asyncio
    await asyncio.gather(
        _probe("ha",     f"{_ha._url}/api/",     _ha._hdrs),
        _probe("ollama", f"{_ollama.url}/api/version"),
    )

    overall = "ok" if all(v["status"] == "ok" for v in results.values()) else (
        "down" if all(v["status"] == "error" for v in results.values()) else "degraded"
    )
    return {"status": overall, "components": results}
```

- [ ] **Step 9.4: Run tests**

```bash
python3 -m pytest tests/test_server_contract.py::test_health_deep_returns_expected_shape -v
```

Expected: PASS.

- [ ] **Step 9.5: Commit**

```bash
git add pipeline/server.py tests/test_server_contract.py
git commit -m "feat(server): add GET /health/deep — probes HA and Ollama reachability"
```

---

### Task 10: Status endpoint — GET /status

**Files:**
- Modify: `pipeline/server.py`
- Modify: `tests/test_server_contract.py`

Expose internal runtime state without reading logs.

- [ ] **Step 10.1: Write a failing contract test**

Add to `tests/test_server_contract.py`:

```python
def test_status_endpoint_returns_expected_shape():
    """GET /status must expose model, satellite_map, spotify_sync_enabled, vram_manager_url."""
    client = TestClient(app)
    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert "model" in body
    assert "satellite_map" in body
    assert isinstance(body["satellite_map"], dict)
    assert "spotify_sync_enabled" in body
    assert isinstance(body["spotify_sync_enabled"], bool)
    assert "vram_manager_url" in body
```

- [ ] **Step 10.2: Run to confirm it fails**

```bash
python3 -m pytest tests/test_server_contract.py::test_status_endpoint_returns_expected_shape -v
```

Expected: FAILED — 404.

- [ ] **Step 10.3: Add `GET /status` to server.py**

Add after the `/health/deep` route:

```python
@app.get("/status")
async def status():
    """Internal runtime state: model, satellite map, feature flags."""
    return {
        "model":                os.getenv("MODEL", "default"),
        "satellite_map":        _ma._satellite_map,
        "spotify_sync_enabled": _spotify_sync is not None,
        "vram_manager_url":     _VRAM_MANAGER_URL or None,
    }
```

- [ ] **Step 10.4: Run test**

```bash
python3 -m pytest tests/test_server_contract.py::test_status_endpoint_returns_expected_shape -v
```

Expected: PASS.

- [ ] **Step 10.5: Commit**

```bash
git add pipeline/server.py tests/test_server_contract.py
git commit -m "feat(server): add GET /status — exposes model, satellite map, feature flags"
```

---

### Task 11: Reload endpoint — POST /reload

**Files:**
- Modify: `pipeline/server.py`
- Modify: `tests/test_server_contract.py`

When a new satellite is added to HA, the container currently requires a restart. `POST /reload` triggers MA rediscovery without restart.

- [ ] **Step 11.1: Write a failing contract test**

Add to `tests/test_server_contract.py`:

```python
def test_reload_endpoint_returns_satellite_map():
    """POST /reload must trigger MA discovery and return updated satellite_map."""
    client = TestClient(app)
    with patch("pipeline.server._ma") as mock_ma:
        mock_ma.discover = AsyncMock()
        mock_ma._satellite_map = {"respeaker_lite": "media_player.respeaker_lite_media_player_2"}
        r = client.post("/reload")
    assert r.status_code == 200
    body = r.json()
    assert "satellite_map" in body
    mock_ma.discover.assert_called_once()
```

- [ ] **Step 11.2: Run to confirm it fails**

```bash
python3 -m pytest tests/test_server_contract.py::test_reload_endpoint_returns_satellite_map -v
```

Expected: FAILED — 405 (method not allowed) or 404.

- [ ] **Step 11.3: Add `POST /reload` to server.py**

Add after the `/status` route:

```python
@app.post("/reload")
async def reload():
    """Re-run Music Assistant satellite discovery without container restart."""
    await _ma.discover()
    log.info("RELOAD | satellite_map refreshed: %s", _ma._satellite_map)
    return {"satellite_map": _ma._satellite_map}
```

- [ ] **Step 11.4: Run test**

```bash
python3 -m pytest tests/test_server_contract.py::test_reload_endpoint_returns_satellite_map -v
```

Expected: PASS.

- [ ] **Step 11.5: Commit**

```bash
git add pipeline/server.py tests/test_server_contract.py
git commit -m "feat(server): add POST /reload — triggers MA satellite rediscovery"
```

---

### Task 12: Optional API key authentication

**Files:**
- Modify: `pipeline/server.py`
- Modify: `config.env.example`
- Modify: `tests/test_server_contract.py`

When `API_KEY` env var is set, `POST /v1/chat/completions` and `POST /reload` require `X-API-Key: <value>`. If `API_KEY` is unset or empty, auth is bypassed (existing behavior). Health, models, and status endpoints are always public.

- [ ] **Step 12.1: Write failing tests**

Add to `tests/test_server_contract.py`:

```python
def test_api_key_auth_rejects_missing_key(monkeypatch):
    """When API_KEY is set, missing X-API-Key header returns 401."""
    monkeypatch.setenv("API_KEY", "secret-key-123")
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "test"}]
    })
    assert r.status_code == 401


def test_api_key_auth_rejects_wrong_key(monkeypatch):
    """When API_KEY is set, wrong X-API-Key returns 401."""
    monkeypatch.setenv("API_KEY", "secret-key-123")
    client = TestClient(app)
    r = client.post("/v1/chat/completions",
                    headers={"X-API-Key": "wrong"},
                    json={"messages": [{"role": "user", "content": "test"}]})
    assert r.status_code == 401


def test_api_key_auth_accepts_correct_key(monkeypatch):
    """When API_KEY is set, correct X-API-Key proceeds past auth."""
    monkeypatch.setenv("API_KEY", "secret-key-123")
    client = TestClient(app)
    with patch("pipeline.server.run_pipeline", new_callable=AsyncMock, return_value="done"):
        r = client.post("/v1/chat/completions",
                        headers={"X-API-Key": "secret-key-123"},
                        json={"messages": [{"role": "user", "content": "test"}]})
    # Should not be 401 — auth passed, pipeline ran
    assert r.status_code != 401


def test_api_key_auth_bypassed_when_not_set(monkeypatch):
    """When API_KEY is empty/unset, no auth is required."""
    monkeypatch.delenv("API_KEY", raising=False)
    client = TestClient(app)
    with patch("pipeline.server.run_pipeline", new_callable=AsyncMock, return_value="done"):
        r = client.post("/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": "test"}]})
    assert r.status_code != 401


def test_health_never_requires_api_key(monkeypatch):
    """GET /health must be reachable without auth even when API_KEY is set."""
    monkeypatch.setenv("API_KEY", "secret-key-123")
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
```

- [ ] **Step 12.2: Run to confirm they fail**

```bash
python3 -m pytest tests/test_server_contract.py::test_api_key_auth_rejects_missing_key \
  tests/test_server_contract.py::test_api_key_auth_rejects_wrong_key -v
```

Expected: FAILED — gets 200/400 instead of 401.

- [ ] **Step 12.3: Add auth middleware to server.py**

Add after the existing imports in `pipeline/server.py`, before the route definitions:

```python
from fastapi import Depends, HTTPException


def _require_api_key(request: Request):
    """FastAPI dependency: enforce X-API-Key when API_KEY env var is set."""
    api_key = os.getenv("API_KEY", "").strip()
    if not api_key:
        return  # auth disabled
    provided = request.headers.get("X-API-Key", "")
    if provided != api_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
```

Apply the dependency to the two protected routes:

```python
@app.post("/v1/chat/completions", dependencies=[Depends(_require_api_key)])
async def chat_completions(request: Request):
    ...

@app.post("/reload", dependencies=[Depends(_require_api_key)])
async def reload():
    ...
```

- [ ] **Step 12.4: Add `API_KEY` to config.env.example**

Add to `config.env.example` (after the PORT section):

```
# ── API authentication (optional) ─────────────────────────────────────────────
# When set, POST /v1/chat/completions requires X-API-Key: <value> header.
# Leave empty to disable authentication (default for private LAN use).
API_KEY=
```

- [ ] **Step 12.5: Run all auth tests**

```bash
python3 -m pytest tests/test_server_contract.py -k "api_key" -v
```

Expected: all pass.

- [ ] **Step 12.6: Commit**

```bash
git add pipeline/server.py config.env.example tests/test_server_contract.py
git commit -m "feat(server): optional X-API-Key auth — enabled by setting API_KEY env var"
```

---

### Task 13: VRAM proxy URL routing — extract and test

**Files:**
- Modify: `pipeline/server.py`
- Modify: `tests/test_server_contract.py`

The VRAM proxy URL construction in `server.py` is inline and untested. Extract it into `_resolve_ollama_url()` and add unit tests.

- [ ] **Step 13.1: Write failing tests**

Add to `tests/test_server_contract.py`:

```python
def test_resolve_ollama_url_via_vram_proxy():
    from pipeline.server import _resolve_ollama_url
    assert _resolve_ollama_url("http://vram:8890", "http://ollama:11434") == "http://vram:8890/ollama"


def test_resolve_ollama_url_direct():
    from pipeline.server import _resolve_ollama_url
    assert _resolve_ollama_url("", "http://ollama:11434") == "http://ollama:11434"


def test_resolve_ollama_url_vram_with_trailing_slash():
    from pipeline.server import _resolve_ollama_url
    assert _resolve_ollama_url("http://vram:8890/", "http://ollama:11434") == "http://vram:8890/ollama"
```

- [ ] **Step 13.2: Run to confirm they fail**

```bash
python3 -m pytest tests/test_server_contract.py::test_resolve_ollama_url_via_vram_proxy -v
```

Expected: FAILED — `ImportError: cannot import name '_resolve_ollama_url'`.

- [ ] **Step 13.3: Extract `_resolve_ollama_url` in server.py**

Replace the inline URL construction in `pipeline/server.py`:

```python
# BEFORE (inline):
_VRAM_MANAGER_URL = os.environ.get("VRAM_MANAGER_URL", "").rstrip("/")
_EFFECTIVE_OLLAMA_URL = (
    f"{_VRAM_MANAGER_URL}/ollama"
    if _VRAM_MANAGER_URL
    else os.environ.get("OLLAMA_URL", "")
)

# AFTER (extracted):
def _resolve_ollama_url(vram_manager_url: str, ollama_url: str) -> str:
    """Return the effective Ollama URL: proxy path when VRAM_MANAGER_URL is set."""
    if vram_manager_url:
        return f"{vram_manager_url.rstrip('/')}/ollama"
    return ollama_url

_VRAM_MANAGER_URL = os.environ.get("VRAM_MANAGER_URL", "").rstrip("/")
_EFFECTIVE_OLLAMA_URL = _resolve_ollama_url(
    _VRAM_MANAGER_URL, os.environ.get("OLLAMA_URL", "")
)
```

- [ ] **Step 13.4: Run tests**

```bash
python3 -m pytest tests/test_server_contract.py -k "resolve_ollama" -v
```

Expected: all pass.

- [ ] **Step 13.5: Commit**

```bash
git add pipeline/server.py tests/test_server_contract.py
git commit -m "test(server): extract _resolve_ollama_url and add VRAM proxy routing tests"
```

---

### Task 14: HTTP contract and error-path tests

**Files:**
- Modify: `tests/test_server_contract.py`

Complete the contract coverage: response shapes for all endpoints, and error paths the server should handle.

- [ ] **Step 14.1: Write the contract tests**

Add to `tests/test_server_contract.py`:

```python
# ── Response shape contracts ──────────────────────────────────────────────────

def test_models_endpoint_shape():
    """GET /v1/models must return OpenAI-compatible model list shape."""
    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body.get("object") == "list"
    assert isinstance(body.get("data"), list)
    assert len(body["data"]) >= 1
    model = body["data"][0]
    assert "id" in model
    assert "object" in model
    assert model["object"] == "model"


def test_chat_completions_response_shape():
    """POST /v1/chat/completions must return OpenAI-compatible completion shape."""
    client = TestClient(app)
    with patch("pipeline.server.run_pipeline", new_callable=AsyncMock, return_value="The light is on."):
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "is the light on"}]
        })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert isinstance(body["id"], str) and body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)
    assert isinstance(body["choices"], list) and len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "The light is on."
    assert choice["finish_reason"] == "stop"
    assert "usage" in body


def test_chat_completions_streaming_response_shape():
    """POST /v1/chat/completions with stream=true must return SSE event-stream."""
    client = TestClient(app)
    with patch("pipeline.server.run_pipeline", new_callable=AsyncMock, return_value="Done."):
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "turn on the light"}],
            "stream": True,
        })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")
    # Response body must contain data: lines and [DONE]
    text = r.text
    assert "data:" in text
    assert "[DONE]" in text


# ── Error paths ───────────────────────────────────────────────────────────────

def test_no_user_message_returns_400():
    """Missing user role in messages returns HTTP 400."""
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "system", "content": "hello"}]
    })
    assert r.status_code == 400
    assert "error" in r.json()


def test_empty_messages_returns_400():
    """Empty messages list returns HTTP 400."""
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 400
```

- [ ] **Step 14.2: Run to confirm new tests pass**

```bash
python3 -m pytest tests/test_server_contract.py -v
```

Expected: all pass (the streaming test may need verification that TestClient handles SSE responses).

- [ ] **Step 14.3: Commit**

```bash
git add tests/test_server_contract.py
git commit -m "test(server): complete HTTP contract and error-path test suite"
```

---

### Task 15: Dockerfile healthcheck — use curl

**Files:**
- Modify: `Dockerfile`

Replace the Python-based healthcheck (loads interpreter + httpx, ~200ms) with `curl` (~5ms).

- [ ] **Step 15.1: Write a failing test (structural check)**

Add to `tests/test_server_contract.py`:

```python
def test_dockerfile_uses_curl_healthcheck():
    """Dockerfile healthcheck must use curl, not python -c."""
    import pathlib
    dockerfile = pathlib.Path("Dockerfile").read_text()
    assert "curl" in dockerfile, "Dockerfile healthcheck must use curl"
    assert 'python -c "import httpx' not in dockerfile, \
        "Dockerfile healthcheck must not use 'python -c import httpx'"
```

- [ ] **Step 15.2: Run to confirm it fails**

```bash
python3 -m pytest tests/test_server_contract.py::test_dockerfile_uses_curl_healthcheck -v
```

Expected: FAILED.

- [ ] **Step 15.3: Edit `Dockerfile`**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pipeline/ ./pipeline/
CMD ["sh", "-c", "uvicorn pipeline.server:app --host 0.0.0.0 --port ${PORT:-18795}"]
```

And update `docker-compose.yml` healthcheck:

```yaml
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:18795/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 15s
```

- [ ] **Step 15.4: Run test**

```bash
python3 -m pytest tests/test_server_contract.py::test_dockerfile_uses_curl_healthcheck -v
```

Expected: PASS.

- [ ] **Step 15.5: Commit**

```bash
git add Dockerfile docker-compose.yml tests/test_server_contract.py
git commit -m "fix(docker): replace python healthcheck with curl — 200ms → 5ms per probe"
```

---

### Task 16: Full e2e validation and push

- [ ] **Step 16.1: Run all unit tests**

```bash
python3 -m pytest tests/ --ignore=tests/test_planner_accuracy.py --ignore=tests/test_latency.py -v 2>&1 | tail -20
```

Expected: all pass (was 100, now higher from new contract tests).

- [ ] **Step 16.2: Run live planner accuracy suite**

```bash
python3 -m pytest tests/test_planner_accuracy.py -v 2>&1 | tail -15
```

Expected: 29/29 pass (≥50% per case).

- [ ] **Step 16.3: Run latency suite**

```bash
PIPELINE_URL=http://localhost:18795 python3 -m pytest tests/test_latency.py -v
```

Expected: all 9 pass within thresholds.

- [ ] **Step 16.4: Push to GitHub**

```bash
git push origin master
```

Expected: clean push.

---

## Self-Review

**Spec coverage check:**
- T1.1 httpx pooling → Tasks 2, 3, 4 ✓
- T1.2 python-dotenv → Task 1 ✓
- T1.3 input validation → Task 6 ✓
- T1.4 test_agents.py hardcoded → Task 8 ✓
- T1.5 test_pipeline.py duplicates → Task 7 ✓
- T2.1 deep health → Task 9 ✓
- T2.2 /status → Task 10 ✓
- T2.3 /reload → Task 11 ✓
- T2.4 Dockerfile healthcheck → Task 15 ✓
- T3.1 API key auth → Task 12 ✓
- T3.2 Contract tests → Task 14 ✓
- T3.3 Error path tests → Task 14 ✓
- T3.4 VRAM proxy test → Task 13 ✓

**Placeholder scan:** No TBDs, TODOs, or vague requirements. All steps contain exact code.

**Type consistency:**
- `HAClient.close()` → `async def close(self) -> None` — referenced in Task 5 lifespan ✓
- `OllamaClient.close()` → same signature ✓
- `MusicAssistantClient.close()` → same signature ✓
- `_discover_satellite_players(url, hdrs, client)` — 3-arg signature used consistently in Tasks 4 and 4.4 ✓
- `_resolve_ollama_url(vram_manager_url, ollama_url)` — defined Task 13, tested Task 13 ✓
- `_require_api_key(request)` — defined and applied in Task 12 ✓
