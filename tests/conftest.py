import os, asyncio, pytest
from pipeline.ha_client import HAClient
from pipeline.ollama_client import OllamaClient

HA_URL     = os.getenv("HA_URL",     "http://192.168.68.250:8123")
HA_TOKEN   = os.getenv("HA_TOKEN",   "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.68.250:11434")
MODEL      = os.getenv("MODEL",      "default")

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def ha():
    return HAClient(HA_URL, HA_TOKEN)

@pytest.fixture(scope="session")
def ollama():
    return OllamaClient(OLLAMA_URL, MODEL)

@pytest.fixture(scope="session")
def ha_context(ha, event_loop):
    entities = event_loop.run_until_complete(ha.get_entities())
    areas    = event_loop.run_until_complete(ha.get_areas())
    return {"entities": entities, "areas": areas}
