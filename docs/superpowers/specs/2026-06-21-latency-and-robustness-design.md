# Latency Optimization & Robustness — Design Spec

**Date:** 2026-06-21
**Status:** Approved for implementation
**Author:** Pipeline audit (end-to-end)

## Background

A full end-to-end audit of the voice pipeline (`server → runner → planner → executor → ha/ollama/ma/spotify` clients) was performed. All 125 unit tests pass. Measured latency is LLM-dominated (~0.85–1.1s). The audit identified two fixed non-LLM costs on the request hot path and one robustness gap. A suspected correctness bug (dropped light service parameters) was investigated against the live model and found to be real, but is **out of scope** because the deployment's lights are strictly on/off (no dimming, no color) and the only parametric device is the media player (volume), which already works.

## Goals

1. Remove fixed latency from the request hot path where it is safe to do so.
2. Make the largest fixed action-path cost (state readback) tunable to zero.
3. Close a robustness gap (malformed request body → wrong HTTP status).

## Non-Goals

- Adding light brightness/color/temperature service parameters (hardware does not support them).
- Adding climate/cover parameters (deployment has no such parametric devices beyond media volume).
- Authentication on read-only endpoints (`/status`, `/health/deep`, `/v1/models`) — LAN-only, low risk.
- Any change to the planner prompt, model, or LLM call path.

## Changes

### 1. Entity/area cache with short TTL

**Problem:** `HAClient.get_entities()` and `HAClient.get_areas()` are called fresh on every request (`runner.py` `run_pipeline`), blocking before the LLM call. `/api/states` returns all entities; `get_areas()` runs a server-side Jinja template render. Areas are effectively static; entities change slowly.

**Design:**
- Add an in-memory TTL cache inside `HAClient` for the results of `get_entities()` and `get_areas()`.
- TTL is read from env `HA_CACHE_TTL_S` (default `10`). A value of `0` disables caching (always fetch fresh).
- Add `HAClient.clear_cache()` which invalidates both cached entries; called by the `POST /reload` endpoint so an operator can force a refresh without a container restart.
- Cache is keyed only by method (no arguments); `get_state(entity_id)` is **never** cached — the query fast-path and executor readback must always see fresh device state.

**Why safe:** The planner only uses entity name/state as *context* for intent parsing; a few seconds of staleness does not change which entity a command targets. Actual state answers ("is the light on?") go through `get_state()` fresh.

**Interface:**
- `get_entities()` / `get_areas()` signatures unchanged.
- New: `clear_cache() -> None`.
- New env var: `HA_CACHE_TTL_S` (int seconds, default 10).

**Tests (unit):**
- Second call within TTL returns cached result without a second httpx request.
- Call after TTL expiry refetches.
- `clear_cache()` forces the next call to refetch.
- `HA_CACHE_TTL_S=0` disables caching (every call fetches).
- `get_state()` is never cached (each call hits httpx).

### 2. Configurable action readback

**Problem:** `executor.py` `_run_step` sleeps `_ZIGBEE_SETTLE_S` (from `ZIGBEE_PROPAGATION_MS`, default 400ms) then re-reads entity state on every action, solely to distinguish "already on" from "now on". This adds ~400ms plus extra round-trips to every action command.

**Design:**
- When `ZIGBEE_PROPAGATION_MS=0`, `_run_step` skips both the settle sleep and the post-execution readback, returning the optimistic `success` outcome immediately after `call_service` succeeds.
- Default remains `400` — existing behavior (including "already" detection) is unchanged unless explicitly opted out.
- Failures are still caught: `call_service` raises on HTTP error → `failed` outcome, independent of readback.

**Why safe:** Readback only drives cosmetic "already on/off" phrasing. With it disabled, the optimistic `ok_response` is spoken; failures are still surfaced via the HTTP error path.

**Interface:** No signature change. Behavior gated on existing `ZIGBEE_PROPAGATION_MS` env var (now meaningful at `0`).

**Tests (unit):**
- With settle=0: no `get_state` readback call occurs; outcome is `success`; the 400ms sleep is not awaited.
- With settle>0: readback still runs; an unchanged entity yields `already`; a changed entity yields `success`.
- Failure path (`call_service` raises) yields `failed` in both modes.

### 3. Malformed request body → HTTP 400

**Problem:** `server.py` `chat_completions` calls `await request.json()`; a non-JSON body raises and surfaces as HTTP 500.

**Design:** Wrap the parse; on failure return `JSONResponse({"error": "invalid JSON body"}, status_code=400)`, consistent with the existing 400 responses for empty/oversized transcripts.

**Tests (unit, FastAPI TestClient):**
- Non-JSON body → 400 with error payload.
- Valid body path unchanged (existing contract tests stay green).

## Config / Documentation

- `config.env.example`: document `HA_CACHE_TTL_S` (new) and clarify `ZIGBEE_PROPAGATION_MS=0` skips readback for lowest action latency.

## Testing Strategy

- TDD: write the failing unit test for each change first, then implement.
- Full unit suite (`pytest --ignore=tests/test_planner_accuracy.py --ignore=tests/test_latency.py`) must stay green (currently 125 passing).
- Latency suite thresholds in `baseline.json` should only improve; not lowered as part of this work unless re-measured against the container.

## Rollback

Each change is independently revertable. Cache and readback are env-gated to their prior behavior (`HA_CACHE_TTL_S=0`, `ZIGBEE_PROPAGATION_MS=400`), so production behavior can be restored by config alone without a code revert.
