# Test Infrastructure Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 66 test failures/errors by loading `config.env` in conftest.py and removing deprecated pytest-asyncio patterns.

**Architecture:** Three targeted file edits. `config.env` is loaded once at session start via `python-dotenv` so all test modules pick up the real HA IP (`192.168.68.250`). The deprecated `event_loop` session fixture and `event_loop.run_until_complete()` calls are replaced with `asyncio.run()` in sync fixtures — safe because `HAClient` creates a fresh `httpx.AsyncClient` per call. `test_planner_accuracy.py`'s duplicate `ha`/`ha_context` fixtures are removed so it uses conftest's session-scoped ones.

**Tech Stack:** pytest 9.0.3, pytest-asyncio 1.3.0 (AUTO mode), python-dotenv (already installed), httpx (already installed)

---

## File Map

| File | Action | What changes |
|---|---|---|
| `tests/conftest.py` | Modify | Add dotenv load; remove `event_loop` fixture; convert `ha_context` to sync+asyncio.run() |
| `tests/test_planner_accuracy.py` | Modify | Remove local `ha`, `ha_context` fixtures; fix `_populate_known_ids` to not use `event_loop` |
| `pytest.ini` | No change | `asyncio_mode = auto` is already correct |

---

## Task 1: Confirm baseline — document what is broken

**Files:**
- Read: `tests/conftest.py`
- Read: `tests/test_planner_accuracy.py` (fixtures block only)

- [ ] **Step 1.1: Run the unit-only tests to confirm 67 pass (baseline)**

```bash
cd /home/vertiq/ha-voice-pipeline
python3 -m pytest tests/test_music_control.py tests/test_music_executor.py \
  tests/test_music_runner.py tests/test_music_server.py \
  tests/test_spotify_connect_sync.py -q --tb=no
```

Expected output contains: `67 passed` (or close — these are the mocked tests that never touch live services).

- [ ] **Step 1.2: Run full suite to confirm baseline failure count**

```bash
python3 -m pytest --tb=no -q 2>&1 | tail -3
```

Expected: something like `11 failed, 67 passed, 1 skipped, 55 errors`

---

## Task 2: conftest.py — load config.env before any `os.getenv` call

**Files:**
- Modify: `tests/conftest.py` (lines 1–4, imports block)

TDD framing: the integration tests are the failing "tests" we want to turn green. After this step, the `HA_URL`/`OLLAMA_URL` env vars will resolve to `192.168.68.250` instead of `homeassistant.local`.

- [ ] **Step 2.1: Confirm the env var is NOT loaded before the fix**

```bash
python3 -c "import os; print(os.getenv('HA_URL', 'NOT SET'))"
```

Expected: `NOT SET` (or `homeassistant.local` if partially set)

- [ ] **Step 2.2: Implement — add dotenv load at top of conftest.py**

Replace the entire `tests/conftest.py` with:

```python
import asyncio, os, pytest
from dotenv import load_dotenv
from pipeline.ha_client import HAClient
from pipeline.ollama_client import OllamaClient

# Load config.env (project root) so HA_URL/OLLAMA_URL/HA_TOKEN are available
# to integration tests. override=False means explicit env vars (e.g. CI) win.
load_dotenv("config.env", override=False)

HA_URL     = os.getenv("HA_URL",     "http://homeassistant.local:8123")
HA_TOKEN   = os.getenv("HA_TOKEN",   "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://homeassistant.local:11434")
MODEL      = os.getenv("MODEL",      "default")


@pytest.fixture(scope="session")
def ha():
    return HAClient(HA_URL, HA_TOKEN)


@pytest.fixture(scope="session")
def ollama():
    return OllamaClient(OLLAMA_URL, MODEL)


@pytest.fixture(scope="session")
def ha_context(ha):
    async def _fetch():
        entities = await ha.get_entities()
        areas    = await ha.get_areas()
        return {"entities": entities, "areas": areas}
    return asyncio.run(_fetch())
```

Key changes from old conftest:
- Added `load_dotenv("config.env", override=False)` before `os.getenv` calls
- Removed the `event_loop` fixture entirely (deprecated since pytest-asyncio 0.21)
- `ha_context` is now a sync fixture that calls `asyncio.run()` — safe because `HAClient` creates a fresh `httpx.AsyncClient` per request (no shared connection pool to corrupt)
- Removed `import asyncio` from the old `event_loop` fixture — moved to top-level since we use it in `ha_context`

- [ ] **Step 2.3: Verify env var is now loaded**

```bash
python3 -c "
from dotenv import load_dotenv; import os
load_dotenv('config.env'); print(os.getenv('HA_URL'))
"
```

Expected: `http://192.168.68.250:8123`

- [ ] **Step 2.4: Run the unit tests to confirm no regressions**

```bash
python3 -m pytest tests/test_music_control.py tests/test_music_executor.py \
  tests/test_music_runner.py tests/test_music_server.py \
  tests/test_spotify_connect_sync.py -q --tb=short
```

Expected: all pass (same count as Step 1.1). These tests use mocks and don't care about env vars.

- [ ] **Step 2.5: Commit**

```bash
git add tests/conftest.py
git commit -m "fix(tests): load config.env via dotenv, drop deprecated event_loop fixture"
```

---

## Task 3: test_planner_accuracy.py — remove duplicate fixtures and fix _populate_known_ids

**Files:**
- Modify: `tests/test_planner_accuracy.py`

The file currently defines its own `ha` (session-scoped) and `ha_context` (module-scoped) fixtures that shadow conftest's. This causes HA to be queried twice per session and prevents conftest's `ha_context` (now fixed) from being used. `_populate_known_ids` uses the deprecated `event_loop.run_until_complete()` pattern.

- [ ] **Step 3.1: Locate the fixture block in test_planner_accuracy.py**

The block to remove/replace is between the `CASES` list and the `_score_run` function. It looks like this (present in the file now):

```python
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
```

And the `_populate_known_ids` autouse fixture that also uses `event_loop`:

```python
@pytest.fixture(scope="session", autouse=True)
def _populate_known_ids(ha, event_loop):
    global _known_ids
    entities = event_loop.run_until_complete(ha.get_entities())
    _known_ids = {e["entity_id"] for e in entities}
    print(f"\n  [benchmark] loaded {len(_known_ids)} entity IDs from HA")
```

- [ ] **Step 3.2: Remove the local ha/ha_context fixtures and fix _populate_known_ids**

In `tests/test_planner_accuracy.py`, replace the `# ── Fixtures ──` block with:

```python
# ── Fixtures ──────────────────────────────────────────────────────────────────
# ha and ha_context come from conftest.py (session-scoped).

@pytest.fixture(scope="module")
def ollama():
    return OllamaClient(OLLAMA_URL, MODEL)
```

And replace `_populate_known_ids` with:

```python
@pytest.fixture(scope="session", autouse=True)
def _populate_known_ids(ha):
    """Fetch live entity list from HA and populate the known entity ID set."""
    global _known_ids
    async def _fetch():
        return await ha.get_entities()
    entities = asyncio.run(_fetch())
    _known_ids = {e["entity_id"] for e in entities}
    print(f"\n  [benchmark] loaded {len(_known_ids)} entity IDs from HA")
```

Also add `import asyncio` to the imports at the top of the file if not present. The current imports are:
```python
import asyncio, json, os, re, time, pytest
from pipeline.ha_client import HAClient
from pipeline.ollama_client import OllamaClient
from pipeline.agents.planner import plan
```

`asyncio` is already imported — no change needed.

Also remove the `HA_URL`, `HA_TOKEN` module-level constants that fed the now-removed `ha` fixture (they are no longer used):

```python
# Remove these three lines:
HA_URL     = os.getenv("HA_URL",     "http://localhost:8123")
HA_TOKEN   = os.getenv("HA_TOKEN",   "")
```

Keep `OLLAMA_URL` and `MODEL` since `ollama` fixture still uses them.

- [ ] **Step 3.3: Verify collection — test_planner_accuracy.py collects cleanly**

```bash
python3 -m pytest tests/test_planner_accuracy.py --collect-only -q 2>&1 | tail -10
```

Expected: 29 items collected, no errors.

- [ ] **Step 3.4: Run unit tests again — confirm still passing**

```bash
python3 -m pytest tests/test_music_control.py tests/test_music_executor.py \
  tests/test_music_runner.py tests/test_music_server.py \
  tests/test_spotify_connect_sync.py -q --tb=short
```

Expected: all pass.

- [ ] **Step 3.5: Commit**

```bash
git add tests/test_planner_accuracy.py
git commit -m "fix(tests): remove duplicate fixtures from test_planner_accuracy, drop event_loop dep"
```

---

## Task 4: Verify full suite — confirm 0 errors

**Files:**
- Read: none (verification only)

- [ ] **Step 4.1: Run full suite and check error count is 0**

```bash
python3 -m pytest --tb=no -q 2>&1 | tail -5
```

Expected: `0 errors` in the summary line. The 55 errors from `test_planner_accuracy.py` should be gone.

The integration tests (ha_client, ollama_client, pipeline, planner_accuracy) will either:
- **PASS** if `192.168.68.250` is reachable (HA + Ollama up)
- **FAIL** with a clean `httpx.ConnectError` if not reachable — no more DNS errors, no errors-vs-failures confusion

- [ ] **Step 4.2: Run only the mocked unit tests for a clean baseline**

```bash
python3 -m pytest \
  tests/test_music_control.py \
  tests/test_music_executor.py \
  tests/test_music_runner.py \
  tests/test_music_server.py \
  tests/test_spotify_connect_sync.py \
  -v --tb=short 2>&1 | tail -20
```

Expected: all green, 0 failures, 0 errors.

- [ ] **Step 4.3: Final commit (if any stray changes)**

```bash
git status
# If clean, nothing to do.
```

---

## Self-Review

**Spec coverage check:**
- ✅ Load `config.env` via `load_dotenv()` in conftest → Task 2
- ✅ Remove deprecated `event_loop` fixture → Task 2
- ✅ Convert `ha_context` to non-deprecated pattern → Task 2 (asyncio.run in sync fixture)
- ✅ Remove duplicate `ha` fixture from test_planner_accuracy → Task 3
- ✅ Remove duplicate `ha_context` from test_planner_accuracy → Task 3
- ✅ Fix `_populate_known_ids` event_loop dep → Task 3
- ✅ Verify 0 errors after fix → Task 4

**Placeholder scan:** None found. All code blocks are complete and exact.

**Type consistency:** `ha_context` returns `{"entities": list[dict], "areas": list[dict]}` in conftest Task 2 — matches the usage in `test_planner_accuracy.py` (`ha_context["entities"]`, `ha_context["areas"]`) and in `test_agents.py`.
