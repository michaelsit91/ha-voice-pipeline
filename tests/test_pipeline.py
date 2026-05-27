import os, pytest
from pipeline.runner import run_pipeline
from pipeline.ha_client import HAClient
from pipeline.ollama_client import OllamaClient

HA_URL     = os.getenv("HA_URL",     "http://192.168.68.250:8123")
HA_TOKEN   = os.getenv("HA_TOKEN",   "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.68.250:11434")
MODEL      = os.getenv("MODEL",      "default")

@pytest.fixture(scope="module")
def ha():
    return HAClient(HA_URL, HA_TOKEN)

@pytest.fixture(scope="module")
def ollama():
    return OllamaClient(OLLAMA_URL, MODEL)

async def test_status_query_mentions_state(ha, ollama):
    r = await run_pipeline("is the living room light on", ha, ollama)
    assert any(w in r.lower() for w in ("on", "off", "living room", "light"))

async def test_single_action_responds(ha, ollama):
    r = await run_pipeline("turn on the kitchen light", ha, ollama)
    assert isinstance(r, str) and len(r) > 0

async def test_multi_device_responds(ha, ollama):
    r = await run_pipeline("turn off all living room lights", ha, ollama)
    assert isinstance(r, str) and len(r) > 0

async def test_nonsense_responds_gracefully(ha, ollama):
    r = await run_pipeline("xkqzfwm blarg", ha, ollama)
    assert isinstance(r, str) and len(r) > 0
