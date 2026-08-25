import httpx
import pytest
from unittest.mock import AsyncMock
from pipeline.ha_client import HAClient


class _Resp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload if payload is not None else []
        self.content = b"x"
    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)
    def json(self):
        return self._payload
    @property
    def text(self):
        import json
        return json.dumps(self._payload)


def _client(ttl=10):
    ha = HAClient("http://test", "tok")
    ha._cache_ttl = ttl
    ha._areas_ttl = ttl
    ha._cache.clear()
    return ha


@pytest.mark.asyncio
async def test_read_retries_once_on_transient_then_succeeds():
    ha = _client()
    states = [{"entity_id": "light.a", "state": "on", "attributes": {"friendly_name": "A"}}]
    calls = {"n": 0}

    class _FakeClient:
        is_closed = False
        async def get(self, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("boom")
            return _Resp(200, states)
    ha._client = _FakeClient(); ha._loop = None
    out = await ha.get_entities()
    assert calls["n"] == 2          # retried once
    assert out and out[0]["entity_id"] == "light.a"


@pytest.mark.asyncio
async def test_read_retries_on_503_status():
    ha = _client()
    calls = {"n": 0}

    class _FakeClient:
        is_closed = False
        async def get(self, *a, **k):
            calls["n"] += 1
            return _Resp(503) if calls["n"] == 1 else _Resp(200, {"entity_id": "light.a", "state": "on", "attributes": {}})
    ha._client = _FakeClient(); ha._loop = None
    st = await ha.get_state("light.a")
    assert calls["n"] == 2 and st["state"] == "on"


@pytest.mark.asyncio
async def test_call_service_is_NOT_retried():
    """Re-firing a service can double-execute a toggle — must never retry."""
    ha = _client()
    calls = {"n": 0}

    class _FakeClient:
        is_closed = False
        async def post(self, *a, **k):
            calls["n"] += 1
            raise httpx.ConnectError("boom")
    ha._client = _FakeClient(); ha._loop = None
    with pytest.raises(httpx.ConnectError):
        await ha.call_service("light", "toggle", entity_id="light.a")
    assert calls["n"] == 1          # exactly one attempt, no retry
