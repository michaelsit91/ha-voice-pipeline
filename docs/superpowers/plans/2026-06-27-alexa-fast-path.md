# Alexa-like Fast-Path Implementation Plan

> **For agentic workers:** Implement task-by-task with TDD. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Sub-second, Alexa-like responses for common commands by routing them through a deterministic matcher (no LLM) with optimistic execution and a terse cached ack; the LLM planner remains the fallback for everything ambiguous or complex.

**Architecture:** A pure `match_fast_intent()` parses intent + resolves a single confident entity target via fuzzy matching. `runner.run_pipeline` tries it before the LLM planner; on a hit it executes optimistically (no readback) and returns a fixed ack. Misses fall through to the existing planner→executor path unchanged. Kokoro pre-caches the ack phrases so their TTS is instant.

**Tech Stack:** Python 3.12, rapidfuzz, pytest (asyncio_mode=auto), unittest.mock.

## Global Constraints
- ha-voice-pipeline fast suite stays green: `pytest --ignore=tests/test_planner_accuracy.py --ignore=tests/test_latency.py`.
- Fast-path is **entity-targeted only** (areas/broadcasts → LLM) and **single-clause only** (`" and "` → LLM).
- Fast-path is **optimistic**: no readback settle; loses "already on/off" phrasing by design.
- Defaults on, behind `FAST_PATH_ENABLED` (default `"true"`); `play <song>` always routes to the LLM (title correction).
- `get_state` never cached; existing behavior for misses unchanged.

## Scope
- FP1 `pipeline/agents/fast_intent.py` (new) — matcher.
- FP2 `pipeline/runner.py` — routing + optimistic execution + ack.
- FP3 config/env + flag.
- FP4 `wyoming-voice/wyoming_kokoro_tts` — ack-phrase synthesis cache.

---

### FP1: deterministic intent matcher
`match_fast_intent(transcript, entities, areas) -> dict | None` returns a plan dict
(`{"kind":"action"|"query", "domain","service","entity_id", optional "volume_level"/"brightness_pct", "ack"}`)
or `None` to fall back to the LLM. Confidence-gated: best fuzzy score ≥ `min_score`
and margin over runner-up ≥ `margin`, else `None`. Intent must be valid for the
matched entity's domain. `" and "`, area-only, and `play …` → `None`.

- [ ] Write `tests/test_fast_intent.py`: on/off/toggle/dim/volume-set/stop/query happy paths return a plan; ambiguous (two similar names), unknown device, compound "and", and "play despacito" return `None`.
- [ ] Run → fail (no module).
- [ ] Implement `fast_intent.py`.
- [ ] Run → pass.
- [ ] Commit.

### FP2: runner routing + optimistic ack
- [ ] Write `tests/test_fast_path_routing.py`: a fast-path hit executes via `ha.call_service` and returns the ack **without calling `ollama.chat`**; a miss calls the planner (`ollama.chat` invoked).
- [ ] Run → fail.
- [ ] In `run_pipeline`, after entities/areas fetch and the hesitation check, call `match_fast_intent`; on hit, `_execute_fast()` (optimistic `call_service`, or `get_state` for query) and return the ack/state sentence; else existing path.
- [ ] Run → pass; full fast suite green.
- [ ] Commit.

### FP3: config + flag
- [ ] Gate routing on `FAST_PATH_ENABLED` (default true); thresholds `FAST_PATH_MIN_SCORE` (default 82), `FAST_PATH_MARGIN` (default 8). Document in `config.env.example`.
- [ ] Test: `FAST_PATH_ENABLED=false` forces the LLM path.
- [ ] Commit.

### FP4: Kokoro ack cache (wyoming-voice)
- [ ] Branch `feat/kokoro-ack-cache`. Write `tests/test_ack_cache.py`: synthesizing a phrase in the ack set twice synthesizes once (cached WAV reused).
- [ ] Implement an LRU/dict cache in `KokoroTTSWrapper.synthesize` keyed by (text, voice, speed), warmed with the ack set at load.
- [ ] Run → pass; full wyoming suite green.
- [ ] Commit.

## Final verification
- [ ] ha-voice fast suite green; wyoming suite green.
- [ ] Confirm a fast-path command issues zero `ollama.chat` calls (analysis + unit assertion).
- [ ] Operator (live): measure e2e for "turn on the kitchen light" (expect sub-second, no LLM).
