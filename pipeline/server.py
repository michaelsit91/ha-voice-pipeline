import asyncio, json, logging, os, time, uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from fastapi import Depends, FastAPI, HTTPException, Request
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


def _resolve_ollama_url(vram_manager_url: str, ollama_url: str) -> str:
    """Return the effective Ollama URL: proxy path when VRAM_MANAGER_URL is set."""
    if vram_manager_url:
        return f"{vram_manager_url.rstrip('/')}/ollama"
    return ollama_url


_ha = HAClient(
    os.getenv("HA_URL", ""),
    os.getenv("HA_TOKEN", ""),
)
_VRAM_MANAGER_URL = os.environ.get("VRAM_MANAGER_URL", "").rstrip("/")
_EFFECTIVE_OLLAMA_URL = _resolve_ollama_url(
    _VRAM_MANAGER_URL, os.environ.get("OLLAMA_URL", "")
)
_ollama = OllamaClient(
    _EFFECTIVE_OLLAMA_URL,
    os.getenv("MODEL", "default"),
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

_MAX_TRANSCRIPT_CHARS = 500


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """One-time warmup on startup. GPU clock is locked by systemd nvidia-clocks.service,
    so no periodic keepalive needed — wyoming-voice fires AudioStart warmup per command."""
    if not os.getenv("HA_URL"):
        raise RuntimeError("HA_URL environment variable is required")
    if not os.getenv("OLLAMA_URL"):
        raise RuntimeError("OLLAMA_URL environment variable is required")
    import httpx
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            await c.post(f"{_ollama.url}/api/chat",
                         json={"model": _ollama.model, "messages": [], "keep_alive": -1,
                               "options": {"num_ctx": 8192}})
    except Exception:
        pass
    # Discover satellite → MA player map
    try:
        await _ma.discover()
        log.info("MUSIC | satellite_map: %s", _ma._satellite_map)
    except Exception as e:
        log.warning("MUSIC | discovery failed at startup: %s", e)
    yield
    # Graceful shutdown: close pooled HTTP clients
    await _ha.close()
    await _ollama.close()
    await _ma.close()
    if _spotify_sync is not None and _spotify_sync._client is not None:
        await _spotify_sync._client.aclose()


app = FastAPI(lifespan=_lifespan)


def _require_api_key(request: Request) -> None:
    """FastAPI dependency: enforce X-API-Key when API_KEY env var is set."""
    api_key = os.getenv("API_KEY", "").strip()
    if not api_key:
        return
    if request.headers.get("X-API-Key", "") != api_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/deep")
async def health_deep():
    """Probe HA and Ollama reachability. Always HTTP 200; check body status."""
    import httpx as _httpx
    results: dict = {}

    async def _probe(name: str, url: str, headers: dict = {}) -> None:
        try:
            async with _httpx.AsyncClient(timeout=4) as c:
                r = await c.get(url, headers=headers)
                r.raise_for_status()
            results[name] = {"status": "ok"}
        except Exception as exc:
            results[name] = {"status": "error", "detail": str(exc)[:120]}

    await asyncio.gather(
        _probe("ha",     f"{_ha._url}/api/",  _ha._hdrs),
        _probe("ollama", f"{_ollama.url}/api/version"),
    )

    all_ok   = all(v["status"] == "ok"    for v in results.values())
    all_down = all(v["status"] == "error" for v in results.values())
    overall  = "ok" if all_ok else ("down" if all_down else "degraded")
    return {"status": overall, "components": results}


@app.get("/v1/models")
async def list_models():
    model_id = os.getenv("MODEL", "default")
    return {
        "object": "list",
        "data": [{"id": model_id, "object": "model", "owned_by": "local"}],
    }


@app.get("/status")
async def status():
    """Internal runtime state: model, satellite map, feature flags."""
    return {
        "model":                os.getenv("MODEL", "default"),
        "satellite_map":        _ma._satellite_map,
        "spotify_sync_enabled": _spotify_sync is not None,
        "vram_manager_url":     _VRAM_MANAGER_URL or None,
    }


@app.post("/reload", dependencies=[Depends(_require_api_key)])
async def reload():
    """Re-run Music Assistant satellite discovery without container restart."""
    await _ma.discover()
    log.info("RELOAD | satellite_map refreshed: %s", _ma._satellite_map)
    return {"satellite_map": _ma._satellite_map}


@app.post("/v1/chat/completions", dependencies=[Depends(_require_api_key)])
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

    if len(transcript) > _MAX_TRANSCRIPT_CHARS:
        return JSONResponse({"error": "transcript too long"}, status_code=400)

    t0 = time.perf_counter()
    log.info("IN  | %r", transcript)
    satellite = request.query_params.get("satellite")
    text = await run_pipeline(
        transcript, _ha, _ollama,
        ma=_ma, satellite=satellite,
        spotify_sync=_spotify_sync,
    )
    log.info("OUT | %.2fs | %r", time.perf_counter() - t0, text)
    cid      = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    model_id = os.getenv("MODEL", "default")
    ts       = int(time.time())

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
