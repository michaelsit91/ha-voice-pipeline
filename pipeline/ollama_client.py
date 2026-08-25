import os, httpx
from pipeline._http import PooledClient

# Identity for the vram-manager proxy: attributes traffic on the dashboard mesh.
# ha-voice-pipeline keeps voice priority (priority_class=voice in consumers.json).
_CONSUMER_NAME = os.getenv("OLLAMA_CONSUMER", "ha-voice-pipeline")

class OllamaClient(PooledClient):
    # A planned response is a small JSON object — a few hundred tokens at most.
    # Without a cap the model occasionally runs away and blows past the client
    # timeout, which surfaces as a failed voice command rather than a slow one.
    # Capping turns that worst case into a bounded, recoverable response.
    _DEFAULT_NUM_PREDICT = 1024

    # Reasoning tokens are drawn from the same budget as the answer, so the fast
    # path's cap starves a thinking pass: at 1024 the model spends the whole
    # budget reasoning and returns done_reason="length" with empty content, which
    # the planner sees as an unparseable response. The thinking pass gets its own,
    # much larger budget — it is already the slow path, so latency is not the
    # constraint there.
    _THINKING_NUM_PREDICT = 8192

    def __init__(self, ollama_url: str, model: str):
        self.url   = ollama_url.rstrip("/")
        self.model = model
        self._num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
        self._num_predict = int(os.getenv("OLLAMA_NUM_PREDICT",
                                          str(self._DEFAULT_NUM_PREDICT)))
        self._thinking_num_predict = int(os.getenv("OLLAMA_NUM_PREDICT_THINK",
                                                   str(self._THINKING_NUM_PREDICT)))
        self._client: httpx.AsyncClient | None = None
        self._loop: object | None = None
        self._client_timeout = 60

    async def chat(self, system: str, user: str, format: dict | str | None = None,
                   think: bool = False) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        num_predict = self._thinking_num_predict if think else self._num_predict
        body: dict = {"model": self.model, "messages": messages,
                      "stream": False, "think": think,
                      "keep_alive": -1,
                      "options": {"num_ctx": self._num_ctx,
                                  "num_predict": num_predict}}
        if format is not None:
            body["format"] = format
        r = await self._get_client().post(f"{self.url}/api/chat", json=body,
                                          headers={"X-Consumer": _CONSUMER_NAME})
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
