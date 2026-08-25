import asyncio, os, pytest
from pathlib import Path
from dotenv import load_dotenv
from pipeline.ha_client import HAClient
from pipeline.ollama_client import OllamaClient
from tests._cleanup import turn_off_all_switches

# Load config.env (project root) so HA_URL/OLLAMA_URL/HA_TOKEN are available
# to integration tests. override=False means explicit env vars (e.g. CI) win.
load_dotenv(Path(__file__).parent.parent / "config.env", override=False)

HA_URL     = os.getenv("HA_URL",     "http://homeassistant.local:8123")
HA_TOKEN   = os.getenv("HA_TOKEN",   "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://homeassistant.local:11434")
MODEL      = os.getenv("MODEL",      "default")


@pytest.fixture(scope="session")
def ha():
    return HAClient(HA_URL, HA_TOKEN)


@pytest.fixture(scope="session")
def ollama():
    return OllamaClient(OLLAMA_URL, MODEL)


@pytest.fixture(scope="session", autouse=True)
def ensure_switches_off_after_e2e():
    """After the live/e2e session, turn every switch off so tests don't leave
    real devices powered on. Uses its own client (independent of any module-scoped
    `ha` override) so it never causes a fixture scope mismatch. Best-effort."""
    yield

    async def _run() -> None:
        cleanup_ha = HAClient(HA_URL, HA_TOKEN)
        try:
            await turn_off_all_switches(cleanup_ha)
        finally:
            await cleanup_ha.close()

    try:
        asyncio.run(_run())
    except Exception:
        pass


@pytest.fixture(scope="session")
def ha_context(ha):
    async def _fetch():
        entities = await ha.get_entities()
        areas    = await ha.get_areas()
        return {"entities": entities, "areas": areas}
    return asyncio.run(_fetch())
