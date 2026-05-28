"""End-to-end Piper TTS validation.

Drives the *full* HA assist pipeline (intent → our pipeline container → Piper)
via the WebSocket `assist_pipeline/run` command so we can inspect exactly what
text Piper receives and confirm it varies across repeated identical commands.

Prerequisites
-------------
- Pipeline container must be running (healthy)
- HA env vars must be set: HA_URL, HA_TOKEN
- The 'Local Home Assistant' pipeline must be configured with
  conversation.extended_openai_conversation + tts.wyoming_voice

Run:
    export $(cat config.env | xargs)
    pytest tests/test_piper_tts.py -v -s
"""

import asyncio
import json
import os

import pytest
import websockets

HA_URL   = os.getenv("HA_URL",   "http://192.168.68.250:8123")
HA_TOKEN = os.getenv("HA_TOKEN", "")

_WS_URL  = HA_URL.replace("http://", "ws://").replace("https://", "wss://") + "/api/websocket"

# The pipeline that wires Wyoming STT → extended_openai_conversation → Wyoming TTS (Piper)
_PIPELINE_ID = "01kac7jws1frz6s2x3v9mvnfz4"


async def _run_assist(text: str) -> dict:
    """Run the assist pipeline with a text input and return all events keyed by type."""
    async with websockets.connect(_WS_URL) as ws:
        await ws.recv()  # auth_required
        await ws.send(json.dumps({"type": "auth", "access_token": HA_TOKEN}))
        auth = json.loads(await ws.recv())
        assert auth["type"] == "auth_ok", f"Auth failed: {auth}"

        await ws.send(json.dumps({
            "id": 1,
            "type": "assist_pipeline/run",
            "pipeline": _PIPELINE_ID,
            "start_stage": "intent",   # skip STT — provide text directly
            "end_stage": "tts",
            "input": {"text": text},
        }))

        events: dict[str, dict] = {}
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            if msg.get("type") == "event":
                ev = msg["event"]
                events[ev["type"]] = ev.get("data", {})
                if ev["type"] in ("run-end", "error"):
                    break
        return events


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def event_loop_module():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_piper_receives_tts_input():
    """Full pipeline run produces a tts-start event with non-empty tts_input."""
    events = await _run_assist("is the kitchen light on")

    assert "tts-start" in events, f"No tts-start event — got: {list(events)}"
    tts_input = events["tts-start"].get("tts_input", "")
    assert tts_input.strip(), "tts_input sent to Piper is empty"
    print(f"\n  Piper received: {tts_input!r}")


@pytest.mark.asyncio
async def test_piper_uses_wyoming_voice_engine():
    """tts-start event confirms Wyoming Voice (Piper) is the TTS engine."""
    events = await _run_assist("is the office light on")

    tts_start = events.get("tts-start", {})
    assert "wyoming" in tts_start.get("engine", "").lower(), (
        f"Expected Wyoming engine, got: {tts_start.get('engine')}"
    )


@pytest.mark.asyncio
async def test_piper_generates_audio_url():
    """tts-end event contains a media_id URL, confirming Piper synthesised audio."""
    events = await _run_assist("turn on the office light")

    assert "tts-end" in events, f"No tts-end event — got: {list(events)}"
    media_id = events["tts-end"].get("tts_output", {}).get("media_id", "")
    assert media_id.startswith("media-source://tts/"), (
        f"Unexpected media_id: {media_id!r}"
    )
    print(f"\n  Piper audio URL: {media_id}")


@pytest.mark.asyncio
async def test_reply_phrases_vary_across_runs():
    """Running the same command 5 times produces at least 2 distinct Piper inputs.

    The _vary() function in executor.py randomly prepends acknowledgment phrases
    (\"Sure,\", \"Done!\", \"Got it,\", etc.).  With 5 runs and ~70% variation rate
    we statistically expect several distinct phrases.
    """
    responses = []
    for _ in range(5):
        events = await _run_assist("turn on the kitchen light")
        tts_input = events.get("tts-start", {}).get("tts_input", "")
        responses.append(tts_input)
        print(f"  Piper: {tts_input!r}")

    unique = set(responses)
    assert len(unique) >= 2, (
        f"Expected varied Piper phrases across 5 runs, but got only: {unique}"
    )
    print(f"\n  {len(unique)} distinct phrases across 5 runs ✓")


@pytest.mark.asyncio
async def test_piper_direct_tts_speak():
    """tts.speak via HA service API confirms Piper can synthesise on demand."""
    import httpx
    url   = f"{HA_URL}/api/services/tts/speak"
    hdrs  = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "entity_id": "tts.wyoming_voice",
        "media_player_entity_id": "media_player.respeaker_lite_media_player",
        "message": "Pipeline test successful.",
    }
    async with httpx.AsyncClient() as c:
        r = await c.post(url, headers=hdrs, json=payload, timeout=10)
    assert r.status_code == 200, f"tts.speak failed: {r.status_code} {r.text[:200]}"
