import os
import pytest
import httpx
from pipeline.ha_client import HAClient

HA_URL = os.getenv("HA_URL", "http://homeassistant.local:8123")
HA_TOKEN = os.getenv("HA_TOKEN", "")
CONTROLLABLE_DOMAINS = {"light", "switch", "fan", "media_player", "climate", "cover", "input_boolean"}


def test_ha_client_reuses_http_client():
    """HAClient._get_client() must return the same instance on repeated calls."""
    ha = HAClient("http://test", "token")
    # _client is None before first use (lazy init)
    assert ha._client is None
    # After calling _get_client twice, same object returned (connection pooling)
    c1 = ha._get_client()
    c2 = ha._get_client()
    assert c1 is c2
    assert isinstance(c1, httpx.AsyncClient)


@pytest.fixture(scope="module")
def client():
    return HAClient(HA_URL, HA_TOKEN)


@pytest.mark.asyncio
async def test_get_entities_returns_controllable_only(client):
    entities = await client.get_entities()
    assert len(entities) > 0
    for e in entities:
        assert e["entity_id"].split(".")[0] in CONTROLLABLE_DOMAINS
        assert "name" in e and "state" in e


@pytest.mark.asyncio
async def test_get_areas_returns_list(client):
    areas = await client.get_areas()
    assert isinstance(areas, list)
    assert all("name" in a for a in areas)


@pytest.mark.asyncio
async def test_get_state_returns_entity(client):
    entities = await client.get_entities()
    eid = entities[0]["entity_id"]
    state = await client.get_state(eid)
    assert state["entity_id"] == eid
    assert "state" in state


@pytest.mark.asyncio
async def test_call_service_light(client):
    await client.call_service("light", "turn_on", "light.kitchen_ceiling")
