import asyncio, json, httpx
from typing import Callable

class OllamaClient:
    def __init__(self, ollama_url: str, model: str):
        self.url   = ollama_url.rstrip("/")
        self.model = model
        self._client: httpx.AsyncClient | None = None
        self._loop: object | None = None

    def _get_client(self) -> httpx.AsyncClient:
        try:
            current_loop: object | None = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        loop_changed = self._loop is not None and self._loop is not current_loop
        if self._client is None or self._client.is_closed or loop_changed:
            self._client = httpx.AsyncClient(timeout=60)
            self._loop = current_loop
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._loop = None

    async def chat(self, system: str, user: str, format: dict | str | None = None) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        body: dict = {"model": self.model, "messages": messages,
                      "stream": False, "think": False}
        if format is not None:
            body["format"] = format
        r = await self._get_client().post(f"{self.url}/api/chat", json=body)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    async def chat_with_tools(
        self,
        system: str,
        user: str,
        tools: list[dict],
        tool_handler: Callable[[str, dict], str],
        max_rounds: int = 6,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        client = self._get_client()
        for _ in range(max_rounds):
            r = await client.post(
                f"{self.url}/api/chat",
                json={"model": self.model, "messages": messages,
                      "tools": tools, "stream": False, "think": False},
            )
            r.raise_for_status()
            msg = r.json()["message"]
            messages.append(msg)

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                return msg.get("content", "").strip()

            for tc in tool_calls:
                fn   = tc["function"]
                name = fn["name"]
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                result = tool_handler(name, args)
                messages.append({
                    "role":    "tool",
                    "content": result if isinstance(result, str) else json.dumps(result),
                })

        return messages[-1].get("content", "").strip()
