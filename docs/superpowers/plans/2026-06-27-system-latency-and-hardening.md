# System Latency & Hardening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut real latency from the voice hot path and close two confirmed correctness/security gaps, tuned to the measured RTX 3060 / i5-11500 server, verified by TDD unit tests.

**Architecture:** Three independent repos. ha-voice-pipeline: pin Ollama keep_alive, root-caused HA cache, executor service allowlist, dead-code + DRY. vram-manager: make the 2 s Ollama yield gap conditional on generation activity, stop leaking admin tokens into the dashboard. wyoming-voice: extract one shared audio-normalize util.

**Tech Stack:** Python 3.12, FastAPI, httpx, aiohttp, pytest (asyncio_mode=auto), unittest.mock, Jinja2.

## Global Constraints
- ha-voice-pipeline fast suite stays green: `python3 -m pytest -q --ignore=tests/test_planner_accuracy.py --ignore=tests/test_latency.py` (130 passing).
- vram-manager suite stays green: `python3 -m pytest -q` (142 passing).
- wyoming-voice suite stays green: `python3 -m pytest -q`.
- `get_state()` is NEVER cached.
- No lossy default flipped: `ZIGBEE_PROPAGATION_MS` stays `400`.
- New env: `HA_CACHE_TTL_S=10` (roster), `HA_AREAS_TTL_S=300` (areas), `OLLAMA_NUM_CTX=8192`.
- Each repo on its own feature branch; one combined diff per repo for review at the end.

## File structure
- `ha-voice-pipeline/pipeline/ollama_client.py` — keep_alive/num_ctx pin; lose `chat_with_tools`; compose `PooledClient`.
- `ha-voice-pipeline/pipeline/ha_client.py` — TTL cache; compose `PooledClient`.
- `ha-voice-pipeline/pipeline/music_assistant_client.py` — compose `PooledClient`.
- `ha-voice-pipeline/pipeline/_http.py` — NEW shared pooled-client mixin.
- `ha-voice-pipeline/pipeline/server.py` — `/reload` clears HA cache.
- `ha-voice-pipeline/pipeline/agents/planner.py` — drop state column from context + examples.
- `ha-voice-pipeline/pipeline/agents/executor.py` — service allowlist; dedupe early-returns.
- `vram-manager/proxy_lock.py` — conditional yield.
- `vram-manager/monitor.py` — expose `generation_busy`; wire predicate.
- `vram-manager/main.py` + `templates/dashboard.html` — stop token injection.
- `wyoming-voice/wyoming_faster_whisper/audio_utils.py` — NEW shared normalize.

---

## REPO A — ha-voice-pipeline
Branch first: `git checkout -b perf/system-latency-hardening`

### Task 1 — OPT-2: pin keep_alive + num_ctx in OllamaClient.chat

**Files:**
- Modify: `pipeline/ollama_client.py`
- Test: `tests/test_ollama_client.py` (append)

**Interfaces:**
- Produces: `OllamaClient.chat()` POSTs body containing `keep_alive: -1` and `options.num_ctx: int`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_ollama_client.py`:

```python
@pytest.mark.asyncio
async def test_chat_pins_keep_alive_and_num_ctx():
    """chat() must pin keep_alive=-1 and num_ctx so the resident model isn't unpinned/reloaded."""
    sent = {}

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": "ok"}}

    class _FakeClient:
        is_closed = False
        async def post(self, url, json=None):
            sent["json"] = json
            return _Resp()

    o = OllamaClient("http://test:11434", "m")
    o._client = _FakeClient()
    o._loop = None
    await o.chat(system="s", user="u")
    assert sent["json"]["keep_alive"] == -1
    assert sent["json"]["options"]["num_ctx"] == 8192
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_ollama_client.py::test_chat_pins_keep_alive_and_num_ctx -q`
Expected: FAIL (`KeyError: 'keep_alive'`).

- [ ] **Step 3: Implement** — in `pipeline/ollama_client.py`, change the import line `import asyncio, json, httpx` to `import asyncio, json, os, httpx`. In `__init__`, after `self.model = model`, add:

```python
        self._num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
```

In `chat()`, replace the `body` assignment:

```python
        body: dict = {"model": self.model, "messages": messages,
                      "stream": False, "think": False}
```

with:

```python
        body: dict = {"model": self.model, "messages": messages,
                      "stream": False, "think": False,
                      "keep_alive": -1, "options": {"num_ctx": self._num_ctx}}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_ollama_client.py::test_chat_pins_keep_alive_and_num_ctx -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/ollama_client.py tests/test_ollama_client.py
git commit -m "perf(ollama): pin keep_alive=-1 + num_ctx so planner call never unpins/reloads model"
```

### Task 2 — OPT-4: delete dead chat_with_tools; dedupe executor early-returns

**Files:**
- Modify: `pipeline/ollama_client.py` (remove `chat_with_tools`)
- Modify: `tests/test_ollama_client.py` (remove the tool-calling test)

**Interfaces:**
- Produces: `OllamaClient` no longer exposes `chat_with_tools`.

- [ ] **Step 1: Remove the dead method.** In `pipeline/ollama_client.py` delete the entire `async def chat_with_tools(...)` method (the block from `async def chat_with_tools(` through its final `return messages[-1].get("content", "").strip()`). Then remove the now-unused import: change `from typing import Callable` line — delete it (no other use). If `json` is still used elsewhere in the file, keep it; otherwise the import stays (it is used by `chat_with_tools` only — after deletion, remove `json` from the import line so it reads `import asyncio, os, httpx`).

- [ ] **Step 2: Remove its test.** In `tests/test_ollama_client.py` delete `test_tool_calling_executes_and_returns` (the whole function).

- [ ] **Step 3: Executor early-returns — confirm, no change.** The `executor.py:261-264` block (`n_fail == 0` → already/ok) is distinct logic, not duplicated. The several `return {"entity_id": entity_id, "outcome": "success"}` statements in `_run_step` each guard a *different* condition (readback disabled / no target_ids / empty states_after) and merging them would entangle independent guards. Confirmed during audit: **no safe dedupe exists — leave executor.py unchanged in this task.** (This task's only executor change is none; the deletion is in ollama_client.py.)

- [ ] **Step 4: Run the fast suite**

Run: `python3 -m pytest -q --ignore=tests/test_planner_accuracy.py --ignore=tests/test_latency.py`
Expected: PASS (129 — one test removed).

- [ ] **Step 5: Commit**

```bash
git add pipeline/ollama_client.py tests/test_ollama_client.py
git commit -m "chore: remove dead OllamaClient.chat_with_tools and its test"
```

### Task 3 — OPT-3: HA TTL cache (areas + roster), drop stale state from planner context

**Files:**
- Modify: `pipeline/ha_client.py`
- Modify: `pipeline/server.py` (`/reload`)
- Modify: `pipeline/agents/planner.py` (`_build_context` + `_SYSTEM` examples)
- Modify: `config.env.example`
- Test: `tests/test_ha_cache.py` (new), `tests/test_planner.py` or append to an existing planner test for context shape

**Interfaces:**
- Produces: `HAClient.clear_cache() -> None`; `get_entities`/`get_areas` TTL-cached; `_build_context` emits `entity_id,name` rows (no state).

- [ ] **Step 1: Write failing cache tests** — create `tests/test_ha_cache.py`:

```python
import pytest
from unittest.mock import AsyncMock
from pipeline.ha_client import HAClient


def _client(ttl: float, areas_ttl: float = 300) -> HAClient:
    ha = HAClient("http://test", "token")
    ha._cache_ttl = ttl
    ha._areas_ttl = areas_ttl
    ha._cache.clear()
    return ha


@pytest.mark.asyncio
async def test_entities_cached_within_ttl():
    ha = _client(ttl=10)
    ha._fetch_entities = AsyncMock(return_value=[{"entity_id": "light.a", "name": "A", "state": "on"}])
    await ha.get_entities()
    await ha.get_entities()
    ha._fetch_entities.assert_awaited_once()


@pytest.mark.asyncio
async def test_areas_cached_within_ttl():
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

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_ha_cache.py -q`
Expected: FAIL (`AttributeError: '_cache'` / `_fetch_entities`).

- [ ] **Step 3: Implement the cache in `pipeline/ha_client.py`.** Change `import asyncio, httpx` → `import asyncio, os, time, httpx`. In `__init__`, after `self._loop = None`, add:

```python
        self._cache_ttl = float(os.getenv("HA_CACHE_TTL_S", "10"))
        self._areas_ttl = float(os.getenv("HA_AREAS_TTL_S", "300"))
        self._cache: dict[str, tuple[float, object]] = {}
```

Add, directly after `close`:

```python
    def clear_cache(self) -> None:
        """Invalidate cached entities/areas so the next call refetches."""
        self._cache.clear()

    async def _cached(self, key: str, fetch, ttl: float) -> object:
        """Return a TTL-cached fetch result. ttl<=0 disables caching."""
        if ttl > 0:
            hit = self._cache.get(key)
            if hit is not None and (time.monotonic() - hit[0]) < ttl:
                return hit[1]
        value = await fetch()
        if ttl > 0:
            self._cache[key] = (time.monotonic(), value)
        return value
```

Rename `async def get_entities` → `async def _fetch_entities` and `async def get_areas` → `async def _fetch_areas` (bodies unchanged). Add the public wrappers after `_fetch_areas`:

```python
    async def get_entities(self) -> list[dict]:
        return await self._cached("entities", self._fetch_entities, self._cache_ttl)

    async def get_areas(self) -> list[dict]:
        return await self._cached("areas", self._fetch_areas, self._areas_ttl)
```

- [ ] **Step 4: Run cache tests** — `python3 -m pytest tests/test_ha_cache.py -q` → PASS (6).

- [ ] **Step 5: Drop the stale state column from the planner context.** In `pipeline/agents/planner.py`, replace `_build_context`:

```python
def _build_context(entities: list[dict], areas: list[dict]) -> str:
    """Build the prompt context from the (already-filtered) entity list.
    States are intentionally omitted: they can be stale under the HA roster cache
    and must not bias entity/intent selection — the executor's readback is the
    authority on current state. Caps at 30 entities for the context budget."""
    area_rows = ["area_id,name"] + [f"{a['area_id']},{a['name']}" for a in areas]
    rows = ["entity_id,name"]
    for e in entities[:30]:
        rows.append(f"{e['entity_id']},{e['name']}")
    return "Areas:\n" + "\n".join(area_rows) + "\n\nDevices:\n" + "\n".join(rows)
```

In the `_SYSTEM` string, update the three `Devices:` example lines to drop the trailing state token so the examples match the new two-column shape:
- `Devices: light.office_light,Office Light,on | fan.living_room_fan,Living Room Fan,off` → `Devices: light.office_light,Office Light | fan.living_room_fan,Living Room Fan`
- `Devices: fan.living_room_fan,Living Room Fan,on | fan.master_bedroom_fan,Master Bedroom Fan,off | fan.office_fan,Office Fan,on | fan.guest_room_fan,Guest Room Fan,off` → drop each `,on`/`,off` so each entry is `entity_id,Name`
- `Devices: media_player.respeaker_lite_media_player_2,Spotify,playing` → `Devices: media_player.respeaker_lite_media_player_2,Spotify`

- [ ] **Step 6: Write a context-shape test** — create `tests/test_planner_context.py`:

```python
from pipeline.agents.planner import _build_context


def test_context_omits_state_column():
    entities = [{"entity_id": "light.a", "name": "A", "state": "on"}]
    areas = [{"area_id": "kitchen", "name": "Kitchen"}]
    ctx = _build_context(entities, areas)
    assert "entity_id,name\n" in ctx
    assert "light.a,A" in ctx
    # the live state must NOT leak into the planner context
    assert "light.a,A,on" not in ctx
    assert ctx.count(",on") == 0
```

Run: `python3 -m pytest tests/test_planner_context.py -q` → PASS.

- [ ] **Step 7: Wire `/reload` to clear the cache.** In `pipeline/server.py`, in `reload()` add `_ha.clear_cache()` as the first body line after the docstring (before `await _ma.discover()`).

- [ ] **Step 8: Document env.** In `config.env.example`, before the Music Assistant section, add:

```bash
# ── HA state cache ────────────────────────────────────────────────────────────
# Roster (device id/name) cache TTL in seconds. Device STATE is always read fresh,
# so this only affects how fast a newly added/renamed device appears. /reload and
# restart clear it. Set 0 to disable.
HA_CACHE_TTL_S=10
# Area-list cache TTL (seconds). Areas are static HA config — safe to cache long.
HA_AREAS_TTL_S=300
```

- [ ] **Step 9: Full fast suite** — `python3 -m pytest -q --ignore=tests/test_planner_accuracy.py --ignore=tests/test_latency.py` → PASS (129 + 7 new = 136).

- [ ] **Step 10: Commit**

```bash
git add pipeline/ha_client.py pipeline/server.py pipeline/agents/planner.py config.env.example tests/test_ha_cache.py tests/test_planner_context.py
git commit -m "perf(ha): TTL cache for roster/areas; drop stale state from planner context"
```

> **Live-accuracy note (operator):** after deploy, run `tests/test_planner_accuracy.py` against the live Ollama. If accuracy regresses from the dropped state column, fall back by restoring the state column in `_build_context` and the examples while keeping the cache (the cache itself is independent of the column change).

### Task 4 — SEC-2: executor domain/service allowlist

**Files:**
- Modify: `pipeline/agents/executor.py`
- Test: `tests/test_executor_allowlist.py` (new)

**Interfaces:**
- Produces: `_run_step` refuses any `(domain, service)` not in `_ALLOWED_SERVICES`, returning `{"outcome": "failed", ...}` without calling HA.

- [ ] **Step 1: Write failing tests** — create `tests/test_executor_allowlist.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pipeline.agents import executor
from pipeline.agents.executor import _run_step


@pytest.mark.asyncio
async def test_disallowed_service_refused_without_ha_call():
    ha = MagicMock()
    ha.call_service = AsyncMock()
    ha.get_state = AsyncMock()
    step = {"domain": "hassio", "service": "host_reboot", "entity_id": "x"}
    result = await _run_step(step, ha)
    assert result["outcome"] == "failed"
    ha.call_service.assert_not_awaited()


@pytest.mark.asyncio
async def test_allowed_service_executes():
    ha = MagicMock()
    ha.call_service = AsyncMock(return_value={})
    ha.get_state = AsyncMock()
    step = {"domain": "light", "service": "turn_on", "entity_id": "light.a"}
    with patch.object(executor, "_ZIGBEE_SETTLE_S", 0):
        result = await _run_step(step, ha)
    assert result["outcome"] == "success"
    ha.call_service.assert_awaited_once()
```

- [ ] **Step 2: Run to verify the first fails**

Run: `python3 -m pytest tests/test_executor_allowlist.py -q`
Expected: FAIL — `host_reboot` currently reaches `call_service`.

- [ ] **Step 3: Implement.** In `pipeline/agents/executor.py`, after the `_VOLUME_STEP` constant, add:

```python
# Services the planner is permitted to emit. The LLM chooses domain+service, so
# this is the safety boundary preventing a prompt-injected transcript from
# reaching a destructive service. music_assistant is handled before _run_step.
_ALLOWED_SERVICES: dict[str, frozenset[str]] = {
    "light":         frozenset({"turn_on", "turn_off", "toggle", "get_state"}),
    "switch":        frozenset({"turn_on", "turn_off", "toggle", "get_state"}),
    "fan":           frozenset({"turn_on", "turn_off", "toggle", "get_state"}),
    "input_boolean": frozenset({"turn_on", "turn_off", "toggle", "get_state"}),
    "cover":         frozenset({"open_cover", "close_cover", "stop_cover", "toggle", "get_state"}),
    "climate":       frozenset({"set_temperature", "set_hvac_mode", "turn_on", "turn_off", "get_state"}),
    "media_player":  frozenset({"media_play", "media_pause", "media_stop",
                                "volume_up", "volume_down", "volume_set", "get_state"}),
}
```

In `_run_step`, immediately after `area_id = step.get("area_id")` (before the `extra` line), add:

```python
    if service not in _ALLOWED_SERVICES.get(domain, frozenset()):
        log.warning("EXEC | refused disallowed service %s.%s", domain, service)
        return {"entity_id": entity_id or area_id, "outcome": "failed",
                "error": f"service {domain}.{service} not allowed"}
```

- [ ] **Step 4: Run to verify both pass** — `python3 -m pytest tests/test_executor_allowlist.py -q` → PASS (2).

- [ ] **Step 5: Full fast suite** — `python3 -m pytest -q --ignore=tests/test_planner_accuracy.py --ignore=tests/test_latency.py` → PASS (138).

- [ ] **Step 6: Commit**

```bash
git add pipeline/agents/executor.py tests/test_executor_allowlist.py
git commit -m "feat(executor): allowlist domain/service so LLM can't trigger arbitrary HA services"
```

### Task 5 — OPT-5a: DRY shared pooled httpx client

**Files:**
- Create: `pipeline/_http.py`
- Modify: `pipeline/ha_client.py`, `pipeline/ollama_client.py`, `pipeline/music_assistant_client.py`
- Test: `tests/test_http_pool.py` (new)

**Interfaces:**
- Produces: `pipeline._http.PooledClient` with `_get_client()` / `close()`; subclasses set `self._client_timeout` (None = no default timeout) and `self._client`/`self._loop` in `__init__`.

- [ ] **Step 1: Write failing test** — create `tests/test_http_pool.py`:

```python
import asyncio, httpx, pytest
from pipeline._http import PooledClient


class _C(PooledClient):
    def __init__(self, timeout=None):
        self._client = None
        self._loop = None
        self._client_timeout = timeout


def test_reuses_same_client():
    c = _C()
    assert c._get_client() is c._get_client()


@pytest.mark.asyncio
async def test_recreates_after_close():
    c = _C()
    first = c._get_client()
    await c.close()
    assert c._client is None
    assert c._get_client() is not first


def test_timeout_applied():
    c = _C(timeout=60)
    assert c._get_client().timeout.read == 60
```

- [ ] **Step 2: Run to verify it fails** — `python3 -m pytest tests/test_http_pool.py -q` → FAIL (`No module named 'pipeline._http'`).

- [ ] **Step 3: Create `pipeline/_http.py`:**

```python
import asyncio, httpx


class PooledClient:
    """Mixin: a lazily-created, event-loop-aware pooled httpx.AsyncClient.

    Subclasses must set in __init__: self._client = None, self._loop = None,
    and self._client_timeout (float seconds, or None for httpx default).
    In production there is one loop per process so the client is pooled for
    life; under per-function test loops it is transparently recreated.
    """

    _client: httpx.AsyncClient | None
    _loop: object | None
    _client_timeout: float | None

    def _get_client(self) -> httpx.AsyncClient:
        try:
            current_loop: object | None = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        loop_changed = self._loop is not None and self._loop is not current_loop
        if self._client is None or self._client.is_closed or loop_changed:
            kwargs = {} if self._client_timeout is None else {"timeout": self._client_timeout}
            self._client = httpx.AsyncClient(**kwargs)
            self._loop = current_loop
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._loop = None
```

- [ ] **Step 4: Refactor the three clients.** In each of `ha_client.py`, `ollama_client.py`, `music_assistant_client.py`: (a) import `from pipeline._http import PooledClient`; (b) make the class inherit it (`class HAClient(PooledClient):` etc.); (c) delete that class's own `_get_client` and `close` methods; (d) in `__init__`, keep `self._client = None` and `self._loop = None`, and add `self._client_timeout = None` for HAClient and MusicAssistantClient, `self._client_timeout = 60` for OllamaClient (replacing the old `httpx.AsyncClient(timeout=60)` behaviour).

Exact import fixes after the method deletions (asyncio is now unused in all three; httpx type-hints remain in `ha_client`/`ollama_client`/`music_assistant_client` on `self._client`):
- `ha_client.py`: change `import asyncio, os, time, httpx` → `import os, time, httpx`.
- `ollama_client.py`: change `import asyncio, os, httpx` → `import os, httpx`.
- `music_assistant_client.py`: change `import asyncio, logging, re` → `import logging, re` (keep the separate `import httpx` line).

- [ ] **Step 5: Run pool test + full suite**

Run: `python3 -m pytest tests/test_http_pool.py -q` → PASS (3).
Run: `python3 -m pytest -q --ignore=tests/test_planner_accuracy.py --ignore=tests/test_latency.py` → PASS (141).

- [ ] **Step 6: Commit**

```bash
git add pipeline/_http.py pipeline/ha_client.py pipeline/ollama_client.py pipeline/music_assistant_client.py tests/test_http_pool.py
git commit -m "refactor: extract shared PooledClient mixin (DRY _get_client across 3 clients)"
```

---

## REPO B — vram-manager
Branch first: `git checkout -b perf/conditional-yield-and-token-fix`

### Task 6 — OPT-1: conditional Ollama yield gap

**Files:**
- Modify: `proxy_lock.py`, `monitor.py`
- Test: `tests/test_proxy_lock.py` (new)

**Interfaces:**
- Produces: `OllamaProxyLock(yield_gap, should_yield=None)`; skips the gap when `should_yield()` is False. `GPUMonitor.generation_busy: bool` updated each tick.

- [ ] **Step 1: Write failing tests** — create `tests/test_proxy_lock.py`:

```python
from unittest.mock import AsyncMock, patch
import pytest
from proxy_lock import OllamaProxyLock


async def _arm(lock):
    await lock.acquire()
    lock.release()  # sets _yield_until = now + gap


async def test_no_yield_when_predicate_false():
    lock = OllamaProxyLock(yield_gap=2.0, should_yield=lambda: False)
    await _arm(lock)
    with patch("proxy_lock.asyncio.sleep", new=AsyncMock()) as sleep:
        await lock.acquire()
        sleep.assert_not_awaited()
    lock.release()


async def test_yields_when_predicate_true():
    lock = OllamaProxyLock(yield_gap=2.0, should_yield=lambda: True)
    await _arm(lock)
    with patch("proxy_lock.asyncio.sleep", new=AsyncMock()) as sleep:
        await lock.acquire()
        sleep.assert_awaited()
    lock.release()


async def test_legacy_always_yields_without_predicate():
    lock = OllamaProxyLock(yield_gap=2.0)
    await _arm(lock)
    with patch("proxy_lock.asyncio.sleep", new=AsyncMock()) as sleep:
        await lock.acquire()
        sleep.assert_awaited()
    lock.release()


def test_release_on_unlocked_does_not_raise():
    OllamaProxyLock().release()
```

- [ ] **Step 2: Run to verify they fail** — `python3 -m pytest tests/test_proxy_lock.py -q` → FAIL (`unexpected keyword 'should_yield'`).

- [ ] **Step 3: Implement in `proxy_lock.py`.** Replace `__init__` and `_wait_yield_gap`:

```python
    def __init__(self, yield_gap: float = 2.0, should_yield=None):
        self.lock = asyncio.Lock()
        self._yield_until: float = 0
        self._yield_gap = yield_gap
        self._should_yield = should_yield
        self.active = False

    async def _wait_yield_gap(self):
        # Skip the gap entirely when no generation is contending for the GPU —
        # the gap exists only to let image/video gen reclaim VRAM.
        if self._should_yield is not None and not self._should_yield():
            return
        wait = max(0.0, self._yield_until - time.time())
        if wait > 0:
            await asyncio.sleep(wait)
```

- [ ] **Step 4: Run to verify they pass** — `python3 -m pytest tests/test_proxy_lock.py -q` → PASS (4).

- [ ] **Step 5: Wire the predicate in `monitor.py`.** In `GPUMonitor.__init__`, before the line `self.proxy = OllamaProxyLock(ollama_yield_gap)`, add:

```python
        self.generation_busy = False
```

and change that line to:

```python
        self.proxy = OllamaProxyLock(ollama_yield_gap, should_yield=lambda: self.generation_busy)
```

In `tick`, after the line `invokeai_busy = invokeai_in_progress > 0`, add:

```python
        self.generation_busy = invokeai_busy or invokeai_pending > 0
```

- [ ] **Step 6: Run full suite** — `python3 -m pytest -q` → PASS (146 — 142 + 4).

- [ ] **Step 7: Commit**

```bash
git add proxy_lock.py monitor.py tests/test_proxy_lock.py
git commit -m "perf(proxy): skip 2s Ollama yield gap when no generation is active"
```

### Task 7 — SEC-1: stop leaking admin tokens into the dashboard

**Files:**
- Modify: `templates/dashboard.html:338`, `main.py:509-517`
- Test: `tests/test_dashboard_no_token_leak.py` (new)

**Interfaces:**
- Produces: `GET /` HTML contains no token; dashboard JS reads tokens from `localStorage`.

- [ ] **Step 1: Write failing test** — create `tests/test_dashboard_no_token_leak.py`:

```python
from pathlib import Path


def test_template_has_no_server_side_token_injection():
    """The dashboard template must not interpolate the admin tokens server-side."""
    src = (Path(__file__).parent.parent / "templates" / "dashboard.html").read_text()
    assert "{{ api_token" not in src
    assert "{{ services_token" not in src
```

- [ ] **Step 2: Run to verify it fails** — `python3 -m pytest tests/test_dashboard_no_token_leak.py -q` → FAIL.

- [ ] **Step 3: Fix the template.** In `templates/dashboard.html`, replace line 338:

```html
<script>window._apiToken={{ api_token | tojson }};window._servicesToken={{ services_token | tojson }};</script>
```

with:

```html
<script>
(function () {
  function readToken(key, label) {
    var v = localStorage.getItem(key);
    if (v === null) { v = window.prompt('Enter ' + label + ' (blank if none):') || ''; localStorage.setItem(key, v); }
    return v;
  }
  window._apiToken = readToken('gm_api_token', 'API token');
  window._servicesToken = readToken('gm_services_token', 'services token');
})();
</script>
```

- [ ] **Step 4: Stop passing the tokens from the route.** In `main.py`, in the `dashboard` route's `TemplateResponse`, delete the two lines `"api_token": API_TOKEN,` and `"services_token": SERVICES_TOKEN,` from the context dict.

- [ ] **Step 5: Run to verify it passes** — `python3 -m pytest tests/test_dashboard_no_token_leak.py -q` → PASS.

- [ ] **Step 6: Run full suite** — `python3 -m pytest -q` → PASS (147).

- [ ] **Step 7: Commit**

```bash
git add templates/dashboard.html main.py tests/test_dashboard_no_token_leak.py
git commit -m "fix(security): stop rendering admin tokens into unauthenticated dashboard"
```

---

## REPO C — wyoming-voice
Branch first: `git checkout -b refactor/shared-audio-normalize`

### Task 8 — OPT-5b: extract shared normalize_audio

**Files:**
- Create: `wyoming_faster_whisper/audio_utils.py`
- Modify: `wyoming_faster_whisper/faster_whisper_engine.py`, `wyoming_faster_whisper/parakeet_engine.py`
- Test: `tests/test_audio_utils.py` (new)

**Interfaces:**
- Produces: `audio_utils.normalize_audio(np.ndarray) -> np.ndarray` (float32 in [-1, 1]).

- [ ] **Step 1: Write failing test** — create `tests/test_audio_utils.py`:

```python
import numpy as np
from wyoming_faster_whisper.audio_utils import normalize_audio


def test_int16_range_scaled_to_unit():
    audio = np.array([32768, -32768, 0], dtype=np.float32)
    out = normalize_audio(audio)
    assert out.dtype == np.float32
    assert out.max() <= 1.0 and out.min() >= -1.0


def test_already_normalized_passthrough():
    audio = np.array([0.5, -0.5, 0.0], dtype=np.float32)
    out = normalize_audio(audio)
    assert np.allclose(out, audio)


def test_empty_input():
    out = normalize_audio(np.array([], dtype=np.float32))
    assert out.size == 0
```

- [ ] **Step 2: Run to verify it fails** — `python3 -m pytest tests/test_audio_utils.py -q` → FAIL (`No module named ...audio_utils`).

- [ ] **Step 3: Create `wyoming_faster_whisper/audio_utils.py`:**

```python
"""Shared audio helpers for the ASR engines."""
import numpy as np


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to float32 in range [-1, 1] (int16 PCM → unit scale)."""
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.size == 0:
        return audio
    if audio.max() > 1.0 or audio.min() < -1.0:
        audio = audio / 32768.0
    return audio
```

- [ ] **Step 4: Use it in both engines.** In `faster_whisper_engine.py` and `parakeet_engine.py`: add `from .audio_utils import normalize_audio`; delete the `_normalize_audio` method from each class; replace `self._normalize_audio(audio)` calls with `normalize_audio(audio)`.

- [ ] **Step 5: Run audio test + full suite**

Run: `python3 -m pytest tests/test_audio_utils.py -q` → PASS (3).
Run: `python3 -m pytest -q` → PASS (existing + 3).

- [ ] **Step 6: Commit**

```bash
git add wyoming_faster_whisper/audio_utils.py wyoming_faster_whisper/faster_whisper_engine.py wyoming_faster_whisper/parakeet_engine.py tests/test_audio_utils.py
git commit -m "refactor: extract shared normalize_audio used by both ASR engines"
```

---

## Final verification
- [ ] ha-voice-pipeline: `python3 -m pytest -q --ignore=tests/test_planner_accuracy.py --ignore=tests/test_latency.py` → green (~141).
- [ ] vram-manager: `python3 -m pytest -q` → green (~147).
- [ ] wyoming-voice: `python3 -m pytest -q` → green.
- [ ] One combined diff per repo for review.
- [ ] Defaults confirmed: `ZIGBEE_PROPAGATION_MS=400`, `HA_CACHE_TTL_S=10`, `HA_AREAS_TTL_S=300`, `OLLAMA_NUM_CTX=8192`.
- [ ] Operator (live): run `test_planner_accuracy.py` post-deploy for the OPT-3 context change; measure action latency with generation idle (expect yield-gap savings) and on the partial-failure path.

## Expected latency impact
- OPT-1: up to −2000 ms on partial-failure / back-to-back commands when generation idle (the common case); 0 ms change while generating.
- OPT-2: avoids −1-3 s cold reloads (model stays pinned with matching num_ctx).
- OPT-3: −50-150 ms/command (areas template render + roster fetch removed within TTL).
- OPT-4/5, SEC-1/2: 0 ms; correctness/security/maintainability.
