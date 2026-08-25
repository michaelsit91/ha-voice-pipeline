# Test Infrastructure Fix — Multi-Persona Audit

## Problem

All 11 test failures and 55 test errors are integration tests that fall back to
`homeassistant.local` (unresolvable DNS) when `HA_URL`/`OLLAMA_URL` env vars are
not loaded. `config.env` has the correct IP (`192.168.68.250`) but pytest never
reads it. `python-dotenv` is already installed.

Secondary: `conftest.py` uses a deprecated `scope="session"` `event_loop` fixture
and synchronous `event_loop.run_until_complete()` in `ha_context` — both deprecated
since pytest-asyncio 0.21.

No pipeline logic bugs were found. All 67 mocked unit tests pass.

## Audit Findings by Persona

| Persona | Finding | Severity |
|---|---|---|
| Validation Engineer | `conftest.py` never loads `config.env` | Critical (all integration tests fail) |
| Validation Engineer | `test_planner_accuracy.py` duplicates `ha` + `ha_context` fixtures at module scope | Minor (double HA query per run) |
| Developer | Custom `event_loop` fixture (session-scoped) is deprecated in pytest-asyncio ≥ 0.21 | Medium (future breakage) |
| Developer | `ha_context` uses `event_loop.run_until_complete()` directly — deprecated pattern | Medium (future breakage) |
| User | No pipeline logic bugs | N/A |
| Admin | `config.env` has correct IPs and token; `config.env.example` uses DNS examples correctly | N/A |

## Changes

### 1. `tests/conftest.py`

- Add `load_dotenv("config.env", override=False)` at top — loads real IPs before any
  test module reads env vars. `override=False` means explicit env vars (e.g. from CI)
  still win.
- Remove the custom `event_loop` fixture entirely. pytest-asyncio in AUTO mode manages
  the loop.
- Convert `ha_context` from a sync fixture using `event_loop.run_until_complete()` to
  a proper `async` fixture (pytest-asyncio AUTO mode handles this automatically).

### 2. `tests/test_planner_accuracy.py`

- Remove the local `ha` fixture override (session-scoped duplicate of conftest's).
- Remove the local `ha_context` fixture override (module-scoped duplicate of conftest's
  session-scoped one).
- Remove the local `event_loop` dependency from `_populate_known_ids` fixture.
- Convert `_populate_known_ids` to an async fixture.

### 3. No changes to pipeline code, pytest.ini, Dockerfile, docker-compose.yml

## Success Criteria

```
pytest                     # 0 failures, 0 errors (with config.env present)
pytest -x --ignore=tests/test_latency.py --ignore=tests/test_pipeline_advanced.py
                           # fast unit test subset still passes
```

Integration tests (ha_client, ollama_client, pipeline, planner_accuracy) pass when
`192.168.68.250` is reachable. When it's not, they fail with a clear `ConnectError`
rather than a DNS error — same behaviour, just more honest.

## Out of Scope

- Changing pipeline logic
- Adding `@pytest.mark.integration` markers (not requested)
- Changing `config.env.example` defaults
