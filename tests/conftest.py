import asyncio, os, pytest
from pathlib import Path
from dotenv import load_dotenv
from pipeline.ha_client import HAClient
from pipeline.ollama_client import OllamaClient
from tests._cleanup import snapshot_states, restore_states

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
def restore_devices_after_session():
    """Snapshot device state before the e2e session and restore anything that
    drifted afterwards, so tests leave the home exactly as they found it. Uses
    its own client (independent of any module-scoped `ha` override). Best-effort."""
    async def _snapshot() -> dict:
        ha = HAClient(HA_URL, HA_TOKEN)
        try:
            return await snapshot_states(ha)
        finally:
            await ha.close()

    try:
        snap = asyncio.run(_snapshot())
    except Exception:
        snap = {}

    yield

    async def _restore() -> None:
        ha = HAClient(HA_URL, HA_TOKEN)
        try:
            await restore_states(ha, snap)
        finally:
            await ha.close()

    try:
        asyncio.run(_restore())
    except Exception:
        pass


@pytest.fixture(scope="session")
def ha_context(ha):
    async def _fetch():
        entities = await ha.get_entities()
        areas    = await ha.get_areas()
        return {"entities": entities, "areas": areas}
    return asyncio.run(_fetch())
