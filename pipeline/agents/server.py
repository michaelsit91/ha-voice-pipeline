import json, os, time, uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pipeline.ha_client import HAClient
from pipeline.ollama_client import OllamaClient
from pipeline.runner import run_pipeline
from pipeline.music_assistant_client import MusicAssistantClient

app = FastAPI()

import logging
log = logging.getLogger(__name__)

_ha = HAClient(
    os.getenv("HA_URL",   "http://192.168.68.250:8123"),
    os.getenv("HA_TOKEN", ""),
)
_ollama = OllamaClient(
    os.getenv("OLLAMA_URL", "http://192.168.68.250:11434"),
    os.getenv("MODEL",      "default"),
)
_ma = MusicAssistantClient(
    os.getenv("HA_URL",             ""),
    os.getenv("HA_TOKEN",           ""),
    os.getenv("MA_CONFIG_ENTRY_ID", ""),
)


@app.on_event("startup")
async def _startup_warmup():
    try:
        await _ma.discover()
    except Exception as e:
        log.warning("MUSIC | discovery failed at startup: %s", e)


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

    satellite = request.query_params.get("satellite")
    text = await run_pipeline(transcript, _ha, _ollama, ma=_ma, satellite=satellite)
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
