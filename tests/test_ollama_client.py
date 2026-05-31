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
async def test_tool_calling_executes_and_returns(client):
    tools = [{
        "type": "function",
        "function": {
            "name": "get_temp",
            "description": "Get temperature for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    calls = []
    def handler(name, args):
        calls.append((name, args))
        return '{"temperature": 28, "unit": "C"}'

    result = await client.chat_with_tools(
        system="Use get_temp to answer.",
        user="What is the temperature in Bangkok?",
        tools=tools,
        tool_handler=handler,
    )
    assert len(calls) >= 1
    assert calls[0][0] == "get_temp"
    assert "bangkok" in calls[0][1].get("city", "").lower()
    assert isinstance(result, str) and len(result) > 0
