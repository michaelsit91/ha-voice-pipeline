# Latency Optimization & Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut fixed non-LLM latency from the request hot path and close a robustness gap, all behind env-gated defaults that preserve current behavior.

**Architecture:** Three independent changes — (1) a TTL cache in `HAClient` for `get_entities`/`get_areas` with a `clear_cache()` hook wired to `POST /reload`; (2) an env-gated skip of the executor's settle-sleep + state readback when `ZIGBEE_PROPAGATION_MS=0`; (3) a try/except around request-body JSON parsing in the chat endpoint.

**Tech Stack:** Python 3.12, FastAPI, httpx, pytest (asyncio_mode=auto), unittest.mock.

## Global Constraints

- Default behavior MUST be unchanged: `HA_CACHE_TTL_S` defaults to `10`, `ZIGBEE_PROPAGATION_MS` defaults to `400`.
- `get_state()` MUST NEVER be cached — query fast-path and executor readback require fresh state.
- All three clients already pool httpx via `_get_client()`; do not reintroduce per-call clients.
- Full unit suite must stay green: `pytest --ignore=tests/test_planner_accuracy.py --ignore=tests/test_latency.py` (currently 125 passing).
- Test style: `MagicMock`/`AsyncMock` for clients, `@pytest.mark.asyncio` on async tests, FastAPI `TestClient` for endpoints.

---

### Task 1: HAClient TTL cache + clear_cache, wired to /reload

**Files:**
- Modify: `pipeline/ha_client.py` (add cache to `HAClient`, rename fetch bodies)
- Modify: `pipeline/server.py:161-166` (call `_ha.clear_cache()` in `/reload`)
- Modify: `config.env.example` (document `HA_CACHE_TTL_S`)
- Test: `tests/test_ha_cache.py` (new)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `HAClient.clear_cache() -> None`; `HAClient.get_entities()`/`get_areas()` signatures unchanged but now TTL-cached; private `HAClient._fetch_entities()` / `_fetch_areas()` holding the original fetch logic.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ha_cache.py`:

```python
import pytest
from unittest.mock import AsyncMock
from pipeline.ha_client import HAClient


def _client(ttl: float) -> HAClient:
    ha = HAClient("http://test", "token")
    ha._cache_ttl = ttl
    ha._cache.clear()
    return ha


@pytest.mark.asyncio
async def test_get_entities_cached_within_ttl():
    ha = _client(ttl=10)
    ha._fetch_entities = AsyncMock(return_value=[{"entity_id": "light.a", "name": "A", "state": "on"}])
    a = await ha.get_entities()
    b = await ha.get_entities()
    assert a == b
    ha._fetch_entities.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_areas_cached_within_ttl():
    ha = _client(ttl=10)
    ha._fetch_areas = AsyncMock(return_value=[{"area_id": "kitchen", "name": "Kitchen"}])
    await ha.get_areas()
    await ha.get_areas()
    ha._fetch_areas.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_refetches_after_expiry():
    ha = _client(ttl=10)
    ha._fetch_entities = AsyncMock(return_value=[{"entity_id": "light.a", "name": "A", "state": "on"}])
    await ha.get_entities()
    # Force the cached entry to look stale
    ts, val = ha._cache["entities"]
    ha._cache["entities"] = (ts - 1000, val)
    await ha.get_entities()
    assert ha._fetch_entities.await_count == 2


@pytest.mark.asyncio
async def test_clear_cache_forces_refetch():
    ha = _client(ttl=10)
    ha._fetch_entities = AsyncMock(return_value=[{"entity_id": "light.a", "name": "A", "state": "on"}])
    await ha.get_entities()
    ha.clear_cache()
    await ha.get_entities()
    assert ha._fetch_entities.await_count == 2


@pytest.mark.asyncio
async def test_ttl_zero_disables_cache():
    ha = _client(ttl=0)
    ha._fetch_entities = AsyncMock(return_value=[{"entity_id": "light.a", "name": "A", "state": "on"}])
    await ha.get_entities()
    await ha.get_entities()
    assert ha._fetch_entities.await_count == 2


@pytest.mark.asyncio
async def test_get_state_is_never_cached():
    ha = _client(ttl=10)
    sent = []

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"entity_id": "light.a", "state": "on", "attributes": {}}

    class _FakeClient:
        is_closed = False
        async def get(self, *a, **k):
            sent.append(a)
            return _Resp()

    ha._client = _FakeClient()
    ha._loop = None
    await ha.get_state("light.a")
    await ha.get_state("light.a")
    assert len(sent) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_ha_cache.py -q`
Expected: FAIL — `AttributeError: 'HAClient' object has no attribute '_cache'` / `clear_cache` / `_fetch_entities`.

- [ ] **Step 3: Implement the cache in `pipeline/ha_client.py`**

Add `import os, time` to the top import line (currently `import asyncio, httpx`):

```python
import asyncio, os, time, httpx
```

In `HAClient.__init__`, after `self._loop = None`, add:

```python
        self._cache_ttl = float(os.getenv("HA_CACHE_TTL_S", "10"))
        self._cache: dict[str, tuple[float, object]] = {}
```

Add these methods to `HAClient` (place directly after `close`):

```python
    def clear_cache(self) -> None:
        """Invalidate cached entities/areas so the next call refetches."""
        self._cache.clear()

    async def _cached(self, key: str, fetch) -> object:
        """Return a TTL-cached fetch result. TTL<=0 disables caching."""
        if self._cache_ttl > 0:
            hit = self._cache.get(key)
            if hit is not None and (time.monotonic() - hit[0]) < self._cache_ttl:
                return hit[1]
        value = await fetch()
        if self._cache_ttl > 0:
            self._cache[key] = (time.monotonic(), value)
        return value
```

Rename the existing `get_entities` method to `_fetch_entities` (rename `async def get_entities` → `async def _fetch_entities`; body unchanged). Rename the existing `get_areas` method to `_fetch_areas` (same). Then add the public wrappers:

```python
    async def get_entities(self) -> list[dict]:
        return await self._cached("entities", self._fetch_entities)

    async def get_areas(self) -> list[dict]:
        return await self._cached("areas", self._fetch_areas)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_ha_cache.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Wire `/reload` to clear the cache**

In `pipeline/server.py`, the `reload()` function currently is:

```python
@app.post("/reload", dependencies=[Depends(_require_api_key)])
async def reload():
    """Re-run Music Assistant satellite discovery without container restart."""
    await _ma.discover()
    log.info("RELOAD | satellite_map refreshed: %s", _ma._satellite_map)
    return {"satellite_map": _ma._satellite_map}
```

Add `_ha.clear_cache()` as the first line of the body (after the docstring):

```python
@app.post("/reload", dependencies=[Depends(_require_api_key)])
async def reload():
    """Re-run Music Assistant satellite discovery without container restart."""
    _ha.clear_cache()
    await _ma.discover()
    log.info("RELOAD | satellite_map refreshed: %s", _ma._satellite_map)
    return {"satellite_map": _ma._satellite_map}
```

- [ ] **Step 6: Document `HA_CACHE_TTL_S` in `config.env.example`**

In `config.env.example`, under the `# ── Zigbee/Z-Wave propagation delay ──` block, append a new block before the Music Assistant section:

```bash
# ── HA state cache ────────────────────────────────────────────────────────────
# Seconds to cache the device/area list used as LLM planning context. Device
# STATE answers ("is the light on?") always read fresh, so this is safe to raise.
# Set to 0 to disable caching (fetch fresh every request).
HA_CACHE_TTL_S=10
```

- [ ] **Step 7: Run the full unit suite to confirm no regressions**

Run: `python3 -m pytest -q --ignore=tests/test_planner_accuracy.py --ignore=tests/test_latency.py`
Expected: PASS (131 passed — 125 existing + 6 new).

- [ ] **Step 8: Commit**

```bash
git add pipeline/ha_client.py pipeline/server.py config.env.example tests/test_ha_cache.py
git commit -m "perf(ha): TTL cache for entities/areas, cleared on /reload"
```

---

### Task 2: Configurable action readback (skip settle + readback when ZIGBEE_PROPAGATION_MS=0)

**Files:**
- Modify: `pipeline/agents/executor.py` (`_run_step`)
- Modify: `config.env.example` (clarify `ZIGBEE_PROPAGATION_MS=0`)
- Test: `tests/test_executor_readback.py` (new)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: no signature change; `_run_step` behavior now gated on `_ZIGBEE_SETTLE_S > 0`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_executor_readback.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pipeline.agents import executor
from pipeline.agents.executor import _run_step

_STEP = {"domain": "light", "service": "turn_on", "entity_id": "light.a"}


def _make_ha(states):
    """states: list of return values for successive get_state calls."""
    ha = MagicMock()
    ha.call_service = AsyncMock(return_value={})
    ha.get_state = AsyncMock(side_effect=[
        {"entity_id": "light.a", "state": s, "attributes": {}} for s in states
    ])
    return ha


@pytest.mark.asyncio
async def test_settle_zero_skips_readback():
    ha = _make_ha([])  # get_state must not be called
    with patch.object(executor, "_ZIGBEE_SETTLE_S", 0), \
         patch("asyncio.sleep", new=AsyncMock()) as sleep:
        result = await _run_step(_STEP, ha)
    assert result["outcome"] == "success"
    ha.get_state.assert_not_awaited()
    sleep.assert_not_awaited()
    ha.call_service.assert_awaited_once()


@pytest.mark.asyncio
async def test_settle_positive_detects_already():
    ha = _make_ha(["on", "on"])  # before == after → already
    with patch.object(executor, "_ZIGBEE_SETTLE_S", 0.4), \
         patch("asyncio.sleep", new=AsyncMock()):
        result = await _run_step(_STEP, ha)
    assert result["outcome"] == "already"


@pytest.mark.asyncio
async def test_settle_positive_detects_success():
    ha = _make_ha(["off", "on"])  # before != after → success
    with patch.object(executor, "_ZIGBEE_SETTLE_S", 0.4), \
         patch("asyncio.sleep", new=AsyncMock()):
        result = await _run_step(_STEP, ha)
    assert result["outcome"] == "success"


@pytest.mark.asyncio
async def test_failure_caught_in_fast_mode():
    ha = MagicMock()
    ha.call_service = AsyncMock(side_effect=Exception("HA down"))
    ha.get_state = AsyncMock()
    with patch.object(executor, "_ZIGBEE_SETTLE_S", 0):
        result = await _run_step(_STEP, ha)
    assert result["outcome"] == "failed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_executor_readback.py -q`
Expected: FAIL — `test_settle_zero_skips_readback` fails because current code still captures `states_before` via `get_state` and runs the readback.

- [ ] **Step 3: Implement the gate in `pipeline/agents/executor.py`**

In `_run_step`, the block that currently captures `states_before` (the section starting `# Capture state before execution`) must only run when readback is enabled. Replace the `states_before` capture block:

```python
    # Capture state before execution (string or list entity_id; area_id excluded —
    # HA exposes no per-area state endpoint).
    states_before: dict[str, str] = {}
    if isinstance(entity_id, str) and entity_id:
        try:
            states_before[entity_id] = (await ha.get_state(entity_id))["state"]
        except Exception:
            pass
    elif isinstance(entity_id, list) and entity_id:
        try:
            pre = await asyncio.gather(
                *[ha.get_state(eid) for eid in entity_id], return_exceptions=True
            )
            for eid, r in zip(entity_id, pre):
                if not isinstance(r, Exception):
                    states_before[eid] = r["state"]
        except Exception:
            pass
```

with this (wrap the same logic in a `readback` guard):

```python
    # Readback (state diff for "already" detection) is gated on the settle delay.
    # ZIGBEE_PROPAGATION_MS=0 → skip both the pre-read and the post-read entirely
    # for lowest action latency; the optimistic ok_response is spoken instead.
    readback = _ZIGBEE_SETTLE_S > 0

    # Capture state before execution (string or list entity_id; area_id excluded —
    # HA exposes no per-area state endpoint).
    states_before: dict[str, str] = {}
    if readback and isinstance(entity_id, str) and entity_id:
        try:
            states_before[entity_id] = (await ha.get_state(entity_id))["state"]
        except Exception:
            pass
    elif readback and isinstance(entity_id, list) and entity_id:
        try:
            pre = await asyncio.gather(
                *[ha.get_state(eid) for eid in entity_id], return_exceptions=True
            )
            for eid, r in zip(entity_id, pre):
                if not isinstance(r, Exception):
                    states_before[eid] = r["state"]
        except Exception:
            pass
```

Then, immediately after the `if area_id:` early-return block and before `# Determine which entity IDs to read back`, add a fast-mode early return:

```python
    # Area calls: HA has no per-area state endpoint, nothing to diff
    if area_id:
        return {"entity_id": area_id, "outcome": "success"}

    # Fast mode: readback disabled — return optimistic success without re-reading
    if not readback:
        return {"entity_id": entity_id, "outcome": "success"}
```

(The existing `target_ids` computation, `await asyncio.sleep(_ZIGBEE_SETTLE_S)`, and the post-read diff remain unchanged below this point — they are now only reached when `readback` is True.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_executor_readback.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Clarify `ZIGBEE_PROPAGATION_MS=0` in `config.env.example`**

In `config.env.example`, the current block is:

```bash
# ── Zigbee/Z-Wave propagation delay ───────────────────────────────────────────
# Time (ms) to wait after a service call before re-reading device state.
# Increase if state reads return stale values after commands.
ZIGBEE_PROPAGATION_MS=400
```

Append one line to the comment (before the `ZIGBEE_PROPAGATION_MS=400` line):

```bash
# ── Zigbee/Z-Wave propagation delay ───────────────────────────────────────────
# Time (ms) to wait after a service call before re-reading device state.
# Increase if state reads return stale values after commands.
# Set to 0 to skip the readback entirely (~400ms faster actions; loses
# "already on/off" phrasing — the optimistic confirmation is spoken instead).
ZIGBEE_PROPAGATION_MS=400
```

- [ ] **Step 6: Run the full unit suite to confirm no regressions**

Run: `python3 -m pytest -q --ignore=tests/test_planner_accuracy.py --ignore=tests/test_latency.py`
Expected: PASS (135 passed — 131 + 4 new).

- [ ] **Step 7: Commit**

```bash
git add pipeline/agents/executor.py config.env.example tests/test_executor_readback.py
git commit -m "perf(executor): skip settle+readback when ZIGBEE_PROPAGATION_MS=0"
```

---

### Task 3: Malformed request body → HTTP 400

**Files:**
- Modify: `pipeline/server.py` (`chat_completions`)
- Test: `tests/test_server_contract.py` (append one test)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `POST /v1/chat/completions` returns 400 (not 500) on non-JSON body.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_server_contract.py`:

```python
def test_chat_completions_malformed_body_returns_400():
    """Non-JSON request body must return 400, not 500."""
    r = _client().post(
        "/v1/chat/completions",
        content=b"this is not json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400
    assert "error" in r.json()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_server_contract.py::test_chat_completions_malformed_body_returns_400 -q`
Expected: FAIL — raises `json.JSONDecodeError` (surfaces as 500 / server exception).

- [ ] **Step 3: Wrap the JSON parse in `pipeline/server.py`**

In `chat_completions`, replace the first line of the body:

```python
async def chat_completions(request: Request):
    body     = await request.json()
    messages = body.get("messages", [])
```

with:

```python
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    messages = body.get("messages", [])
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_server_contract.py::test_chat_completions_malformed_body_returns_400 -q`
Expected: PASS.

- [ ] **Step 5: Run the full unit suite to confirm no regressions**

Run: `python3 -m pytest -q --ignore=tests/test_planner_accuracy.py --ignore=tests/test_latency.py`
Expected: PASS (136 passed — 135 + 1 new).

- [ ] **Step 6: Commit**

```bash
git add pipeline/server.py tests/test_server_contract.py
git commit -m "fix(server): return 400 on malformed request body"
```

---

## Final Verification

- [ ] Run full unit suite: `python3 -m pytest -q --ignore=tests/test_planner_accuracy.py --ignore=tests/test_latency.py` → 136 passed.
- [ ] Confirm defaults unchanged: `HA_CACHE_TTL_S=10`, `ZIGBEE_PROPAGATION_MS=400` in `config.env.example`.
- [ ] (Optional, requires live container) Re-measure latency suite against the container with `ZIGBEE_PROPAGATION_MS=0` to confirm action-path improvement.
