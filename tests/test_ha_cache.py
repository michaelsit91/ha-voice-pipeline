import pytest
from unittest.mock import AsyncMock
from pipeline.ha_client import HAClient


def _client(ttl: float, areas_ttl: float = 300) -> HAClient:
    ha = HAClient("http://test", "token")
    ha._cache_ttl = ttl
    ha._areas_ttl = areas_ttl
    ha._cache.clear()
    return ha


@pytest.mark.asyncio
async def test_entities_cached_within_ttl():
    ha = _client(ttl=10)
    ha._fetch_entities = AsyncMock(return_value=[{"entity_id": "light.a", "name": "A", "state": "on"}])
    await ha.get_entities()
    await ha.get_entities()
    ha._fetch_entities.assert_awaited_once()


@pytest.mark.asyncio
async def test_areas_cached_within_ttl():
    ha = _client(ttl=10)
    ha._fetch_areas = AsyncMock(return_value=[{"area_id": "kitchen", "name": "Kitchen"}])
    await ha.get_areas()
    await ha.get_areas()
    ha._fetch_areas.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_refetches_after_expiry():
    ha = _client(ttl=10)
    ha._fetch_entities = AsyncMock(return_value=[{"entity_id": "light.a", "name": "A", "state": "on"}])
    await ha.get_entities()
    ts, val = ha._cache["entities"]
    ha._cache["entities"] = (ts - 1000, val)
    await ha.get_entities()
    assert ha._fetch_entities.await_count == 2


@pytest.mark.asyncio
async def test_clear_cache_forces_refetch():
    ha = _client(ttl=10)
    ha._fetch_entities = AsyncMock(return_value=[{"entity_id": "light.a", "name": "A", "state": "on"}])
    await ha.get_entities()
    ha.clear_cache()
    await ha.get_entities()
    assert ha._fetch_entities.await_count == 2


@pytest.mark.asyncio
async def test_ttl_zero_disables_cache():
    ha = _client(ttl=0)
    ha._fetch_entities = AsyncMock(return_value=[{"entity_id": "light.a", "name": "A", "state": "on"}])
    await ha.get_entities()
    await ha.get_entities()
    assert ha._fetch_entities.await_count == 2


@pytest.mark.asyncio
async def test_get_state_is_never_cached():
    ha = _client(ttl=10)
    sent = []

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"entity_id": "light.a", "state": "on", "attributes": {}}

    class _FakeClient:
        is_closed = False
        async def get(self, *a, **k):
            sent.append(a)
            return _Resp()

    ha._client = _FakeClient()
    ha._loop = None
    await ha.get_state("light.a")
    await ha.get_state("light.a")
    assert len(sent) == 2
