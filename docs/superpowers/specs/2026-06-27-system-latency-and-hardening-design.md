# System Latency & Hardening — Design Spec

_Date: 2026-06-27 · Scope: ha-voice-pipeline, vram-manager, wyoming-voice · Mode: **Balanced** (faster by default; nothing that loses information silently — lossy changes stay behind a flag)._

## Goal
Cut real latency from the voice hot path and close two confirmed correctness/security gaps, verified by TDD unit tests + code analysis (the live HA/Ollama/GPU stack is not reachable from the dev environment; operator measures wall-clock after deploy).

## Inputs
- Multi-persona audit (architect / performance / developer / end-user+security), saved as `persona-*.md` in scratch.
- Personas **corrected** the original audit: rapidfuzz-hoist and prompt-trim are ~0 ms (dropped); readback-skip-as-default violates Balanced (dropped — flag stays); the HA cache needs a redesign (below); and surfaced a new win (keep_alive pin).

## Global constraints
- ha-voice-pipeline fast suite stays green: `pytest --ignore=tests/test_planner_accuracy.py --ignore=tests/test_latency.py` (130 passing).
- vram-manager suite stays green (142 passing). wyoming-voice tests stay green.
- No **lossy** default is flipped: `ZIGBEE_PROPAGATION_MS` stays `400`. (Enabling the HA cache by default — OPT-3 — is the intended Balanced speedup: states stay fresh, so it is not a lossy flip; the only effect is ≤TTL roster lag, cleared by `/reload`/restart.)
- `get_state()` is NEVER cached (authoritative for query answers, executor readback, volume).
- Test style unchanged: `pytest` asyncio_mode=auto, `MagicMock`/`AsyncMock`, FastAPI `TestClient`.
- Each change is one TDD cycle; one combined review at the end (batch).

---

## Server capabilities (measured 2026-06-27, this host)
- GPU: RTX 3060 **12 GB**, ~6.2 GB free in steady state (Whisper 2.2 GB + Ollama 3.3 GB + ComfyUI 0.2 GB co-resident).
- Ollama voice model: qwen3 **4B Q4_K_M, 3.3 GB, num_ctx 8192**, already `keep_alive:-1` resident.
- CPU i5-11500 (6c/12t); RAM 31 GB (17 GB free).
- **Implications baked in:** (1) ~6 GB headroom with no generation running ⇒ the yield gap is waste in the common case (OPT-1 wins often) and the keep_alive pin is safe (OPT-2, no eviction pressure); (2) match `num_ctx=8192` on pinned calls so a param mismatch can't trigger a reload; (3) Parakeet CPU `num_threads=4` leaves 8 threads — unchanged.

## OPT-2 — Pin `keep_alive:-1` (+ matching `num_ctx`) in `OllamaClient.chat`  _(do first: trivial, high payoff)_
**Problem:** `pipeline/ollama_client.py:33-34` builds the chat body without `keep_alive`, so every planner call resets Ollama's model timer to the 5-min default → risk of a 1-3 s cold reload, masked today only by the AudioStart warmup side-channel.
**Change:** add `"keep_alive": -1` and `"options": {"num_ctx": OLLAMA_NUM_CTX}` (default `8192`, matching the warmup) to the `chat()` request body, so the call neither un-pins nor reloads the resident instance. (`chat_with_tools` also omits it but is deleted in OPT-4.)
**Behaviour:** none user-visible; strictly removes a cold-reload risk; capability-validated (3.3 GB model fits the 6 GB headroom).
**Test intent:** capture the POSTed JSON; assert `keep_alive == -1` and `options.num_ctx == 8192` on `chat()`.

## OPT-1 — Conditional Ollama yield gap  _(biggest real win)_
**Problem:** `vram-manager/proxy_lock.py:17-32` arms a 2 s gap on every `release()` and waits it on the next `acquire()`, unconditionally. This taxes the pipeline's own back-to-back Ollama calls (partial-failure 2nd LLM call `executor.py:278`, and rapid commands) even when no image/video generation is contending for the GPU. Yield only matters when SwarmUI generation is actually busy/pending.
**Change:** make the wait conditional on generation activity.
- `OllamaProxyLock.__init__` gains an optional `should_yield: Callable[[], bool] | None = None`.
- `_wait_yield_gap()` returns immediately (no sleep) when `should_yield` is provided and returns `False`.
- `monitor.py`: expose `generation_busy` (updated each tick = `invokeai_busy or invokeai_pending > 0`; default `False` before first tick → no yield, safe-for-latency). Construct the lock with `should_yield=lambda: self.generation_busy`.
**Why `busy OR pending`, not `busy`:** gating on busy-alone would let queued/loading SwarmUI gens (`monitor.py:288` counts `waiting_gens`+`loading_models` into pending) starve — voice would skip the yield while a gen is about to start. (developer persona)
**Behaviour:** identical when generation is active; 0 ms gap when idle (the normal case). No spoken-output change.
**Test intent (new `tests/test_proxy_lock.py` — zero coverage today):** (a) `should_yield→False`: a release then immediate acquire does **not** sleep; (b) `should_yield→True`: the gap is waited; (c) `should_yield=None`: legacy always-wait preserved; (d) release-on-unlocked still warns, not raises.

## OPT-3 — HA caching, root-caused redesign
**Root-cause of the revert (76ea14d):** no recorded reason; timeline shows it was pulled 15 min after landing, after the other two tasks, while `progress.md` still marked it "review clean" → a live-testing discovery, not a review finding. Personas identified the two real failure vectors the original "fresh-read" safety claim missed:
1. Stale **state column** in the planner LLM context (`planner.py:183`) can bias which entity/intent the model selects — executor readback cannot fix a *wrong step chosen*.
2. Stale **roster membership** — `ha_client.py:50` filters `unavailable`, so a just-online device is absent for up to one TTL → dropped step / "didn't understand".

**Design (eliminates both vectors):**
- **Areas** (`get_areas`): TTL-cached at `HA_AREAS_TTL_S` (default `300`). Areas are static HA config — the safest, highest-value cache target (performance persona N5: removes a Jinja template render every command, ~zero staleness).
- **Entity roster** (`get_entities`): TTL-cached at `HA_CACHE_TTL_S` (default `10`), but the cached entry stores **id + name only**. The live `state` column is **dropped from the planner context** — `planner._build_context` no longer prints a per-entity state. Authoritative state still comes from `get_state()` (query fast-path, executor readback) which is never cached. This removes vector #1 entirely; vector #2 is bounded to ≤ `HA_CACHE_TTL_S` (10 s) and cleared by `POST /reload` and restart.
- `clear_cache()` wired into `POST /reload` (alongside `_ma.discover()`).
- Fix the shared-object-ref the team's own review flagged: `_cached` returns the stored value; since the planner context is read-only over it, document immutability OR return a shallow copy. (Chosen: document; no caller mutates.)
- Update the planner prompt's context-format note + examples to drop the `,state` column so examples match the new context shape.

**Behaviour / Balanced check:** entity selection is name/room-based, not state-based, so dropping the state column does not change which device gets actuated; speculative `already_response`/`ok_response` strings are still generated and the executor's readback remains the authority on which one is spoken. No user-perceptible regression. **Residual risk (documented):** a device added/renamed/just-online lags ≤10 s until TTL or `/reload`.
**Accuracy caveat:** the prompt-example change should be sanity-checked against `tests/test_planner_accuracy.py` on the live Ollama stack post-merge (operator-run; not in CI).
**Test intent:** roster cached within TTL (one `_fetch_entities`); areas cached within `HA_AREAS_TTL_S`; refetch after expiry; `clear_cache()`/`/reload` forces refetch; `TTL=0` disables; `get_state` never cached; `_build_context` output contains no per-entity state token.

## OPT-4 — Dead code
- Delete `OllamaClient.chat_with_tools` (`ollama_client.py:41-80`) — confirmed unused in production; delete its test in `tests/test_ollama_client.py` in the same change. Drop now-unused imports (`Callable`, and `json` if unused).
- Collapse the two duplicate success early-returns in `executor.py:261-264`.
**Behaviour:** none. **Test intent:** suite stays green; no test references the removed method.

## OPT-5 — DRY extraction
- **HTTP client:** extract the triplicated lazy/loop-tracking `_get_client` + `close` into a shared `pipeline/_http.py::PooledClient` with a parametrized default timeout; `HAClient`, `OllamaClient`, `MusicAssistantClient` compose/subclass it (preserving each one's current timeout: HA per-call, Ollama 60 s, MA default).
- **Audio:** extract the byte-identical `_normalize_audio` (`faster_whisper_engine.py:53`, `parakeet_engine.py:132`) into `wyoming_faster_whisper/audio_utils.py`; both engines call it. (Parakeet's copy is currently untested — one shared, tested implementation covers both.)
**Behaviour:** none. **Test intent:** existing client/engine tests stay green; add a focused unit test for `PooledClient` (loop-change recreation, is_closed recreation) and for `normalize_audio` (int16→[-1,1], empty input, already-normalized passthrough).

## SEC-1 — Stop leaking admin tokens into the unauthenticated dashboard  _(HIGH/CRITICAL, confirmed)_
**Problem:** `vram-manager/main.py:514-515` renders `API_TOKEN` + `SERVICES_TOKEN` into `templates/dashboard.html` (`window._apiToken/_servicesToken`) on the **unauthenticated** `GET /`. Middleware only gates POST. Anyone who can load the page harvests both tokens → full admin incl. `/api/upgrade-gpu` (chroot driver install + reboot).
**Change:** remove server-side token injection from the template render. The dashboard JS obtains the token from `localStorage`, which the operator enters once via a prompt; POSTs attach it from there. `GET /` HTML no longer contains any secret.
**Behaviour:** dashboard still works after the operator enters the token once; programmatic POSTs unaffected.
**Test intent:** `GET /` response body does NOT contain the configured `API_TOKEN`/`SERVICES_TOKEN` values (contract test with tokens set).

## SEC-2 — Executor domain/service allowlist
**Problem:** `executor.py:130` passes the LLM-chosen `domain`+`service` to `ha.call_service` verbatim. Entities are validated against the known roster; **services are not** — a prompt-injected transcript could steer the model to a destructive no-entity service.
**Change:** add a module-level allowlist of permitted `(domain, service)` pairs (covering the services the planner is actually instructed to emit: turn_on/off/toggle, get_state, media_* , volume_*, light brightness, etc.). In `_run_step`, reject a step whose `(domain, service)` is not allowed → return `{"outcome": "failed", ...}` (so the existing fail-response path speaks a safe apology) and log a warning. No HA call is made for a disallowed step.
**Behaviour:** legitimate commands unaffected; out-of-policy services are refused with the normal failure speech.
**Test intent:** allowed step (`light.turn_on`) executes; disallowed step (`hassio.host_reboot`) is refused without calling `ha.call_service`; mixed batch refuses only the bad step.

---

## Config additions (documented in `config.env.example`)
- `HA_CACHE_TTL_S=10` — entity-roster cache TTL (ids+names only; states always fresh; 0 disables).
- `HA_AREAS_TTL_S=300` — area-list cache TTL (static config).
- No new vram-manager env (OPT-1 reuses existing `OLLAMA_YIELD_GAP` + generation state).

## Build order & verification
1. OPT-2 → 2. OPT-1 → 3. OPT-3 → 4. SEC-1 → 5. SEC-2 → 6. OPT-4 → 7. OPT-5.
Each: write failing test → implement → green → full suite. Final: both suites green; one combined diff for review. Expected latency: −up-to-2000 ms (partial-failure/back-to-back when gen idle), −1-3 s avoided cold reloads, −50-150 ms/command (areas+roster cache). Operator confirms with the live latency suite.

## Out of scope (documented findings, not touched)
- `ZIGBEE_PROPAGATION_MS=0` default flip (lossy → flag only).
- `/ollama` proxy auth gating; spotify settings.json write race (reliability) — recorded, deferred.
- `monitor.py` size / `invokeai_*` rename (debt) — recorded, deferred.
