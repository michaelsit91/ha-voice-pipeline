# ha-voice-pipeline

A local voice assistant pipeline for [Home Assistant](https://www.home-assistant.io/) that replaces the default conversation agent with a fast, LLM-driven intent planner backed by [Ollama](https://ollama.ai/).

Say *"turn on the kitchen light"* and the pipeline:
1. Queries HA for relevant devices and areas
2. Sends the transcript + device list to a local Ollama model
3. Parses the structured JSON response (intent, target entities, confirmation phrase)
4. Executes the HA service calls in parallel
5. Returns spoken text → Piper TTS → speaker — **or returns nothing at all**, which is
   the normal outcome for a successful action (see [Response behaviour](#response-behaviour))

Common commands never reach steps 2–3: a deterministic fast path answers them
without touching the LLM.

Optional integration:
- **Music Assistant** — voice-controlled music playback with satellite player routing,
  including fuzzy catalog search re-ranked against the heard transcript

---

## Requirements

| Dependency | Purpose |
|---|---|
| Home Assistant | Smart home controller |
| Ollama (local) | LLM for intent parsing — any model that follows JSON schema |
| Docker / Docker Compose | Container runtime |
| Music Assistant *(optional)* | Music playback integration |

---

## Quick start

### 1. Configure

```bash
cp config.env.example config.env
```

Edit `config.env` with your values:

```env
HA_URL=http://homeassistant.local:8123
HA_TOKEN=your_ha_long_lived_access_token
OLLAMA_URL=http://homeassistant.local:11434
MODEL=your_ollama_model_name
```

See `config.env.example` for all available options.

### 2. Set up the Docker network

The container runs on an external Docker network named `llm-voice_default`. Create it if it doesn't exist:

```bash
docker network create llm-voice_default
```

### 3. Build and run

```bash
docker compose up -d --build
```

The container exposes port `18795` (configurable via `PORT` in `config.env`).

### 4. Configure Home Assistant

1. In HA go to **Settings → Voice assistants → Add assistant**
2. Set the conversation agent to **OpenAI Conversation** (or compatible)
3. Set the API endpoint to `http://<your-docker-host>:18795/v1`
4. Set the model name to match your `MODEL` env var
5. Assign the assistant to your voice satellite(s)

---

## Response behaviour

This is the part that surprises people. The service does **not** speak a confirmation
after every command, and that is deliberate.

### Silent acknowledgement

A successful fast-path action returns an **empty response body**. The satellite plays its
own success chime (firmware `on_end`), so speaking "OK" on top of it is slower and
redundant. The empty string is the ack.

`pipeline/agents/fast_intent.py` encodes this as a single constant:

```python
_ACK = ""   # fast-path successes are acknowledged by the satellite chime, not by TTS
```

Speech is produced only when the words carry information the chime cannot:

| Outcome | Spoken response |
|---|---|
| Fast-path action succeeded | `""` — silence, chime only |
| Hesitation / cancellation | `""` — silence |
| `intent="ignore"` (non-directed speech) | `""` — silence |
| Planner-path action, single step, succeeded | `"Done."` |
| Planner-path action, multiple steps, succeeded | the planner's `ok_response`, with a random prefix |
| Device already in the requested state | the planner's `already_response` |
| State query | e.g. `"The office light is on."` |
| Partial failure | one sentence composed by a micro-LLM call |
| Total failure | the planner's `fail_response`, else `"Sorry, I couldn't complete that."` |

**If you are writing or reading tests: an empty string is a pass, not a failure.**
Stale assertions expecting spoken text after a successful action have already bitten
this repo once. `tests/test_fast_path_routing.py` documents the contract.

### `intent="ignore"` — non-directed speech

The planner's JSON schema allows three intents: `action`, `query`, and `ignore`. `ignore`
is chosen when the transcript is not addressed to the assistant at all — TV or movie
dialogue, background conversation between people, song lyrics, or a meaningless fragment
picked up by a false wake.

When the planner returns `ignore`, `run_pipeline` logs it and returns `""` rather than
speaking "Sorry, I didn't understand that command." Talking over the room is worse than
staying quiet. The planner is also instructed to choose `ignore` only on *positive*
evidence of non-direction: brevity alone is not evidence, so `"make it dark"` and
`"louder"` remain commands.

`ignore` is a legitimate empty plan, so it does **not** trigger the thinking retry
described below.

### Hesitation / cancellation

Before any Home Assistant or Ollama call, `run_pipeline` tests the transcript against
`_HESITATION_PATTERNS` in `pipeline/agents/planner.py` — `hold on`, `wait`, `never mind`,
`forget it`, `cancel`, `abort`, `actually`, `scratch that`, `hang on`, plus Chinese
equivalents (`等一下`, `算了`, `取消`, …). A match short-circuits immediately and returns
`""`. Zero network calls, zero latency.

`stop` is included, but with a negative lookahead so `"stop the music"`, `"stop playing"`,
`"stop the song"` and `"stop the track"` are *not* treated as hesitation — those fall
through to the media-stop fast path.

### The deterministic fast path

Enabled by default (`FAST_PATH_ENABLED=true`). Before the LLM is consulted,
`pipeline/agents/fast_intent.py` tries to map the transcript straight onto a Home
Assistant call:

- on / off / toggle for `light`, `switch`, `fan`, `input_boolean`
- whole-room commands (`"kitchen lights off"`) resolved to an `area_id`
- volume up / down / set to N percent, media stop, media pause
- `"play <query>"` / `"put on <query>"`, with `by <artist>`, and `album` / `playlist` / `artist` prefixes
- state queries (`"is the office light on"`, `"what's …"`)

It only fires on a **confident, unambiguous single target**: the best fuzzy score must
reach `FAST_PATH_MIN_SCORE` (default 82) *and* beat the runner-up by `FAST_PATH_MARGIN`
(default 8). Anything multi-clause (contains `" and "`), vague (`"play some music"`,
`"play something relaxing"`), or unrecognised returns `None` and falls through to the
planner. So do fast-path plans that cannot resolve a Music Assistant player.

Fast-path actions are **optimistic** — no state readback — so they cannot say
"it's already on". They return silence instead.

### Adaptive thinking retry

`plan()` runs at most two passes over Ollama:

| Pass | `think` | Token budget | When it runs |
|---|---|---|---|
| 1 | off | `OLLAMA_NUM_PREDICT` (1024) | always |
| 2 | on | `OLLAMA_NUM_PREDICT_THINK` (8192) | only if pass 1 failed |

Pass 1 is the low-latency default. Pass 2 runs only if pass 1 either failed to parse as
JSON, or parsed but produced no actionable steps after entity validation — so the
thinking cost is paid only when the cheap pass was not good enough. The retry also
re-sends the prompt with an explicit "Reply with ONLY the JSON object" instruction.

The two budgets are separate on purpose: reasoning tokens are drawn from the same budget
as the answer, so running a thinking pass at 1024 makes the model spend its entire budget
reasoning and return empty content, which the planner then sees as unparseable.

Both passes run the response through `_validate_steps`, which fuzzy-repairs or drops
hallucinated `entity_id`s and `area_id`s. Fuzzy repair is confined to the candidate's own
domain, so a hallucinated `fan.*` id can never resolve to a `light.*` entity.

---

## Music Assistant (optional)

Set `MA_CONFIG_ENTRY_ID` in `config.env` (HA → **Settings → Devices & Services → Music
Assistant → ⚙ → Copy entry ID**). At startup, and on every `POST /reload`, the pipeline
discovers the satellite → MA player map so a `"play …"` command lands on the speaker that
heard it. Pass the satellite as a query parameter: `/v1/chat/completions?satellite=<name>`.

Music search goes through Music Assistant's own search service, and results are re-ranked
with rapidfuzz against the heard query so the best acoustic match wins even when STT
misheard the words. **The pipeline never handles Spotify tokens itself** — MA owns that
auth, and two clients refreshing the same PKCE refresh-token chain is what revoked it
previously. This is also why `cryptography` is no longer a dependency.

> `MA_DATA_PATH`, `MA_SETTINGS_JSON` and `LIBRESPOT_DEVICE_NAME` still appear in
> `config.env.example`, but no code reads them and `docker-compose.yml` no longer mounts
> the MA data directory. They are leftovers from the removed Spotify Connect sync feature.

---

## Running tests

Install dev dependencies:

```bash
pip install -r requirements-dev.txt
```

Unit tests (no HA required):

```bash
pytest tests/test_music_executor.py tests/test_music_control.py tests/test_spotify_connect_sync.py -v
```

Integration tests (require live HA + Ollama — set env vars first):

```bash
export $(grep -v '^#' config.env | xargs)
pytest tests/ -v -s
```

Piper TTS end-to-end tests (require `HA_PIPELINE_ID`):

```bash
export HA_PIPELINE_ID=your_pipeline_id
pytest tests/test_piper_tts.py -v -s
```

Find your pipeline ID in **HA Settings → Voice assistants** → copy the ID from the URL.

---

## Architecture

```
Voice satellite
    │  transcript
    ▼
Home Assistant  ──▶  /v1/chat/completions
                           │
                     runner.py
                      ├─ planner (Ollama LLM) ──▶ JSON intent
                      ├─ executor ──▶ HA service calls
                      └─ [music] ──▶ Music Assistant search + play
                           │
                     spoken text
                    ▼
              Piper TTS ──▶ speaker
```

### Key files

| File | Responsibility |
|---|---|
| `pipeline/server.py` | FastAPI app, OpenAI-compatible `/v1/chat/completions` |
| `pipeline/runner.py` | Orchestration: entity filtering, plan → execute |
| `pipeline/agents/planner.py` | LLM prompt, JSON schema, entity validation |
| `pipeline/agents/executor.py` | HA service execution, volume steps, response variation |
| `pipeline/ha_client.py` | HA REST API client |
| `pipeline/music_assistant_client.py` | Music Assistant search + satellite discovery |
| `pipeline/spotify_connect_sync.py` | Spotify Connect token management + playback transfer |

---

## Security note

This service has **no authentication** on the `/v1/chat/completions` endpoint. It is designed to run on a private home network and should not be exposed to the internet. Restrict access with your router/firewall or Docker network rules.
