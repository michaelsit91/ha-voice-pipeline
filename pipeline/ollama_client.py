import os, httpx
from pipeline._http import PooledClient

# Identity for the vram-manager proxy: attributes traffic on the dashboard mesh.
# ha-voice-pipeline keeps voice priority (priority_class=voice in consumers.json).
_CONSUMER_NAME = os.getenv("OLLAMA_CONSUMER", "ha-voice-pipeline")

class OllamaClient(PooledClient):
    def __init__(self, ollama_url: str, model: str):
        self.url   = ollama_url.rstrip("/")
        self.model = model
        self._num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
        self._client: httpx.AsyncClient | None = None
        self._loop: object | None = None
        self._client_timeout = 60

    async def chat(self, system: str, user: str, format: dict | str | None = None,
                   think: bool = False) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        body: dict = {"model": self.model, "messages": messages,
                      "stream": False, "think": think,
                      "keep_alive": -1, "options": {"num_ctx": self._num_ctx}}
        if format is not None:
            body["format"] = format
        r = await self._get_client().post(f"{self.url}/api/chat", json=body,
                                          headers={"X-Consumer": _CONSUMER_NAME})
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
