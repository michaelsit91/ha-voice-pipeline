# ha-voice-pipeline

A local voice assistant pipeline for [Home Assistant](https://www.home-assistant.io/) that replaces the default conversation agent with a fast, LLM-driven intent planner backed by [Ollama](https://ollama.ai/).

Say *"turn on the kitchen light"* and the pipeline:
1. Queries HA for relevant devices and areas
2. Sends the transcript + device list to a local Ollama model
3. Parses the structured JSON response (intent, target entities, confirmation phrase)
4. Executes the HA service calls in parallel
5. Returns a natural spoken confirmation → Piper TTS → speaker

Optional integrations:
- **Music Assistant** — voice-controlled music playback with satellite player routing
- **Spotify Connect sync** — phone Spotify app shows what's playing after a voice command

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

## Spotify Connect sync (optional)

When enabled, this feature transfers Spotify playback to the Librespot device on your satellite after a *"play …"* command, so the phone's Spotify app shows what's playing.

Requirements:
- Music Assistant with the Spotify provider configured
- A Librespot device (e.g. ReSpeaker Lite running Wyoming satellite)

Set the env vars in `config.env`:

```env
MA_DATA_PATH=/path/to/music-assistant/data
MA_SETTINGS_JSON=/ma-data/settings.json
LIBRESPOT_DEVICE_NAME=ReSpeaker Lite
```

`MA_DATA_PATH` is the host-side path to MA's data directory. It is mounted into the container at `/ma-data` automatically via `docker-compose.yml`.

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
