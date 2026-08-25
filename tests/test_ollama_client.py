import os, pytest
import httpx
from pipeline.ollama_client import OllamaClient


def test_ollama_client_reuses_http_client():
    """OllamaClient._get_client() must return the same instance on repeated calls."""
    ollama = OllamaClient("http://test:11434", "model")
    assert ollama._client is None
    c1 = ollama._get_client()
    c2 = ollama._get_client()
    assert c1 is c2
    assert isinstance(c1, httpx.AsyncClient)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://homeassistant.local:11434")
MODEL      = os.getenv("MODEL", "default")

@pytest.fixture(scope="module")
def client():
    return OllamaClient(OLLAMA_URL, MODEL)

@pytest.mark.asyncio
async def test_chat_returns_string(client):
    result = await client.chat(system="Reply with exactly: PONG", user="PING")
    assert isinstance(result, str) and len(result) > 0

@pytest.mark.asyncio
async def test_no_think_blocks_absent(client):
    result = await client.chat(system="Count to 3, comma-separated.", user="go")
    assert "<think>" not in result

@pytest.mark.asyncio
async def test_chat_pins_keep_alive_and_num_ctx():
    """chat() must pin keep_alive=-1 and num_ctx so the resident model isn't unpinned/reloaded."""
    sent = {}

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": "ok"}}

    class _FakeClient:
        is_closed = False
        async def post(self, url, json=None, headers=None):
            sent["headers"] = headers
            sent["json"] = json
            return _Resp()

    o = OllamaClient("http://test:11434", "m")
    o._client = _FakeClient()
    o._loop = None
    await o.chat(system="s", user="u")
    assert sent["json"]["keep_alive"] == -1
    assert sent["json"]["options"]["num_ctx"] == o._num_ctx


@pytest.mark.asyncio
async def test_chat_think_param_default_false_and_overridable():
    sent = {}

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": "ok"}}

    class _FakeClient:
        is_closed = False
        async def post(self, url, json=None, headers=None):
            sent["headers"] = headers
            sent["json"] = json
            return _Resp()

    o = OllamaClient("http://test:11434", "m")
    o._client = _FakeClient()
    o._loop = None
    await o.chat(system="s", user="u")
    assert sent["json"]["think"] is False
    assert sent["headers"] == {"X-Consumer": "ha-voice-pipeline"}
    await o.chat(system="s", user="u", think=True)
    assert sent["json"]["think"] is True


@pytest.mark.asyncio
async def test_thinking_pass_gets_a_larger_num_predict_budget():
    """Reasoning tokens come out of num_predict, so the thinking pass needs its own
    budget — at the fast path's cap the model spends it all reasoning and returns
    empty content, which the planner cannot parse."""
    sent = {}

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": "ok"}}

    class _FakeClient:
        is_closed = False
        async def post(self, url, json=None, headers=None):
            sent["json"] = json
            return _Resp()

    o = OllamaClient("http://test:11434", "m")
    o._client = _FakeClient()
    o._loop = None

    await o.chat(system="s", user="u")
    assert sent["json"]["options"]["num_predict"] == o._num_predict

    await o.chat(system="s", user="u", think=True)
    assert sent["json"]["options"]["num_predict"] == o._thinking_num_predict
    assert OllamaClient._THINKING_NUM_PREDICT > OllamaClient._DEFAULT_NUM_PREDICT
