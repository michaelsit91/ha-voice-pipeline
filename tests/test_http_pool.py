import httpx, pytest
from pipeline._http import PooledClient


class _C(PooledClient):
    def __init__(self, timeout=None):
        self._client = None
        self._loop = None
        self._client_timeout = timeout


def test_reuses_same_client():
    c = _C()
    assert c._get_client() is c._get_client()


@pytest.mark.asyncio
async def test_recreates_after_close():
    c = _C()
    first = c._get_client()
    await c.close()
    assert c._client is None
    assert c._get_client() is not first


def test_timeout_applied():
    c = _C(timeout=60)
    assert c._get_client().timeout.read == 60
