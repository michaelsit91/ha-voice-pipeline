# Production Hardening Design

**Date:** 2026-05-31  
**Status:** Approved — proceeding to implementation

## Problem

The pipeline works correctly for all 138 tests (100 unit + 29 accuracy + 9 latency) but has several gaps that prevent it from being called production-ready:

- Performance: every HA/Ollama call creates a new TCP connection (no pooling)
- Correctness: `python-dotenv` missing from dev dependencies; test fixtures duplicated/hardcoded
- Observability: `/health` is shallow; no `/status` or `/reload` endpoints
- Security: no authentication (opt-in only)
- Test coverage: missing contract tests, HTTP error paths, VRAM proxy routing

## Architecture (unchanged)

```
HTTP /v1/chat/completions
    → server.py (FastAPI)
        → runner.py (pipeline orchestrator)
            → ha_client.py       (fetch entities + areas)
            → _filter_entities() (keyword scoring)
            → planner.py         (Ollama LLM → JSON steps)
            → executor.py        (HA service calls, state diff)
                → music_assistant_client.py  (search + play)
                → spotify_connect_sync.py    (Spotify Connect bridge)
```

The architecture is sound. No structural changes — only hardening within existing components.

## Changes

### T1.1 — httpx connection pooling

**Files:** `pipeline/ha_client.py`, `pipeline/ollama_client.py`, `pipeline/music_assistant_client.py`

Each client stores a single `httpx.AsyncClient` instance created in `__init__`. All methods use this shared client instead of `async with httpx.AsyncClient() as c:`.

Rationale: A new `AsyncClient` per call performs a full TCP handshake + TLS negotiation per request. Each pipeline command makes 4–6 HA calls. Connection reuse eliminates this overhead (10–50ms per call).

Interface contract: caller-facing method signatures are unchanged.

### T1.2 — python-dotenv in requirements-dev.txt

**File:** `requirements-dev.txt`

Add `python-dotenv>=1.0.0`. This package is already used in `conftest.py` (`from dotenv import load_dotenv`) but was missing from the dev install manifest.

### T1.3 — Input validation

**File:** `pipeline/server.py`

Transcript length capped at 500 characters. Requests with longer transcripts return HTTP 400 with `{"error": "transcript too long"}`. Rationale: prevents accidental LLM calls from multi-minute ambient speech captures.

### T1.4 — Fix test_agents.py hardcoded entity IDs

**File:** `tests/test_agents.py`

`light.kitchen_ceiling` (lines 65, 78) is replaced with a dynamically resolved entity from `ha_context["entities"]`. The test skips gracefully if no light entities are available.

### T1.5 — Fix test_pipeline.py duplicate fixtures

**File:** `tests/test_pipeline.py`

Module-level `ha()` and `ollama()` fixture definitions removed. The session-scoped fixtures from `conftest.py` are used instead, ensuring test isolation and correct scoping.

### T2.1 — Deep health endpoint

**File:** `pipeline/server.py`

- `GET /health` — unchanged: returns `{"status": "ok"}` immediately (used by Docker healthcheck)
- `GET /health/deep` — probes HA and Ollama with a 4-second timeout each; returns:
  ```json
  {
    "status": "ok" | "degraded" | "down",
    "components": {
      "ha":     {"status": "ok" | "error", "detail": "..."},
      "ollama": {"status": "ok" | "error", "detail": "..."}
    }
  }
  ```
  HTTP 200 always (health endpoints should not return 5xx — callers parse body).

### T2.2 — /status endpoint

**File:** `pipeline/server.py`

`GET /status` returns internal state:
```json
{
  "model": "qwen3:9b",
  "satellite_map": {"respeaker_lite": "media_player.respeaker_lite_media_player_2"},
  "spotify_sync_enabled": true,
  "vram_manager_url": null
}
```

### T2.3 — MA rediscovery endpoint

**File:** `pipeline/server.py`

`POST /reload` calls `await _ma.discover()` and returns updated satellite map. Allows adding a new satellite without container restart.

### T2.4 — Dockerfile healthcheck

**File:** `Dockerfile`

Replace `python -c "import httpx; ..."` with:
```
CMD curl -sf http://localhost:${PORT:-18795}/health
```
Reduces per-probe overhead from ~200ms (Python startup + import) to ~5ms.

### T3.1 — Optional API key auth

**File:** `pipeline/server.py`

When `API_KEY` env var is set, all non-`/health` endpoints require `X-API-Key: <value>` header. Returns HTTP 401 if missing or wrong. When `API_KEY` is not set, authentication is bypassed (existing behavior unchanged).

Config: add `API_KEY=` (empty = disabled) to `config.env.example`.

### T3.2 — Contract tests for HTTP API shape

**File:** `tests/test_server_contract.py` (new)

Uses FastAPI `TestClient`. Tests the exact JSON shape of:
- `/health` response
- `/v1/models` response  
- `/v1/chat/completions` response (both streaming and non-streaming)
- `/status` response
- `/health/deep` response

### T3.3 — HTTP error path tests

**File:** `tests/test_server_contract.py` (same file)

Tests: malformed JSON body → 422, missing `messages` field → returns gracefully, no user message → 400, transcript > 500 chars → 400, wrong `X-API-Key` → 401.

### T3.4 — VRAM proxy routing test

**File:** `tests/test_ollama_client.py`

Unit test: when `OllamaClient` is constructed with a VRAM-proxy URL, the `/api/chat` request is sent to the proxy path, not the raw Ollama URL. Uses `respx` or `httpx` mock.

## TDD Sequence

For each change:
1. Write failing test that captures the desired contract
2. Implement the minimal change to make it pass
3. Run the full test suite to verify no regressions
4. Commit

## SDD (contract-first) Approach

New endpoints (`/health/deep`, `/status`, `/reload`) are defined as response schemas in tests before the handler is written. The test defines the contract; the implementation satisfies it.

## Non-changes

- Planner prompt, schema, examples — unchanged (passing 29/29 accuracy tests)
- Entity filter logic — unchanged  
- Executor state-diff logic — unchanged
- HTTP API compatibility — all existing callers continue to work
- Streaming behavior — single-chunk SSE is correct for wyoming TTS

## Success criteria

- `python3 -m pytest tests/ --ignore=tests/test_planner_accuracy.py --ignore=tests/test_latency.py` — all pass (currently 100, will grow)
- `PIPELINE_URL=http://localhost:18795 python3 -m pytest tests/test_latency.py` — all pass within thresholds
- `python3 -m pytest tests/test_planner_accuracy.py` — all pass (≥50% per case)
- No regressions: zero previously-passing tests broken
