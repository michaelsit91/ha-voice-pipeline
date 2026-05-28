import json, logging, os, time, uuid
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pipeline.ha_client import HAClient
from pipeline.ollama_client import OllamaClient
from pipeline.music_assistant_client import MusicAssistantClient
from pipeline.runner import run_pipeline
from pipeline.spotify_connect_sync import SpotifyConnectSync

# ── Logging with local timezone ──────────────────────────────────────────────
_TZ = timezone(timedelta(hours=int(os.getenv("TZ_OFFSET_HOURS", "0"))))

class _LocalFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return datetime.fromtimestamp(record.created, tz=_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")

_handler = logging.StreamHandler()
_handler.setFormatter(_LocalFormatter("%(asctime)s | %(levelname)s | %(message)s"))
log = logging.getLogger("pipeline")
log.setLevel(logging.INFO)
log.addHandler(_handler)
log.propagate = False

app = FastAPI()


@app.on_event("startup")
async def _startup_warmup():
    """One-time warmup on startup. GPU clock is locked by systemd nvidia-clocks.service,
    so no periodic keepalive needed — wyoming-voice fires AudioStart warmup per command."""
    import httpx
    url   = _ollama.url
    model = _ollama.model
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            await c.post(f"{url}/api/chat",
                         json={"model": model, "messages": [], "keep_alive": -1,
                               "options": {"num_ctx": 8192}})
    except Exception:
        pass
    # Discover satellite → MA player map
    try:
        await _ma.discover()
        log.info("MUSIC | satellite_map: %s", _ma._satellite_map)
    except Exception as e:
        log.warning("MUSIC | discovery failed at startup: %s", e)

_ha = HAClient(
    os.getenv("HA_URL", ""),
    os.getenv("HA_TOKEN", ""),
)
_ollama = OllamaClient(
    os.getenv("OLLAMA_URL", ""),
    os.getenv("MODEL",      "default"),
)
_ma = MusicAssistantClient(
    os.getenv("HA_URL",             ""),
    os.getenv("HA_TOKEN",           ""),
    os.getenv("MA_CONFIG_ENTRY_ID", ""),
)

# Optional: Spotify Connect sync — bridges MA queue playback to the phone's Spotify app.
# Reads MA's encrypted Spotify refresh token from settings.json; disabled if path not set.
_spotify_sync: SpotifyConnectSync | None = None
_ma_settings = os.getenv("MA_SETTINGS_JSON", "")
_librespot_name = os.getenv("LIBRESPOT_DEVICE_NAME", "ReSpeaker Lite")
if _ma_settings:
    try:
        _spotify_sync = SpotifyConnectSync(_ma_settings, _librespot_name)
        log.info("SPOTIFY_SYNC | enabled (device=%r, settings=%s)", _librespot_name, _ma_settings)
    except Exception as e:
        log.warning("SPOTIFY_SYNC | disabled — could not init: %s", e)

# Validate required environment variables at startup
if not os.getenv("HA_URL"):
    raise RuntimeError("HA_URL environment variable is required")
if not os.getenv("OLLAMA_URL"):
    raise RuntimeError("OLLAMA_URL environment variable is required")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/v1/models")
async def list_models():
    model_id = os.getenv("MODEL", "default")
    return {
        "object": "list",
        "data": [{"id": model_id, "object": "model", "owned_by": "local"}],
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body     = await request.json()
    messages = body.get("messages", [])
    stream   = body.get("stream", False)
    def _extract_text(content):
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
            return " ".join(parts).strip()
        return str(content).strip()

    transcript = next(
        (_extract_text(m["content"]) for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    if not transcript:
        return JSONResponse({"error": "no user message"}, status_code=400)

    t0 = time.perf_counter()
    log.info("IN  | %r", transcript)
    satellite = request.query_params.get("satellite")
    text = await run_pipeline(
        transcript, _ha, _ollama,
        ma=_ma, satellite=satellite,
        spotify_sync=_spotify_sync,
    )
    log.info("OUT | %.2fs | %r", time.perf_counter() - t0, text)
    cid  = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    model_id = os.getenv("MODEL", "default")
    ts   = int(time.time())

    if stream:
        async def event_stream():
            chunk = {
                "id": cid, "object": "chat.completion.chunk", "created": ts,
                "model": model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": text},
                             "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            done_chunk = {
                "id": cid, "object": "chat.completion.chunk", "created": ts,
                "model": model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return {
        "id":      cid,
        "object":  "chat.completion",
        "created": ts,
        "model":   model_id,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage":   {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
