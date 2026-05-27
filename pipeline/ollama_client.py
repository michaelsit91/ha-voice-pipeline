import json, httpx
from typing import Callable

class OllamaClient:
    def __init__(self, ollama_url: str, model: str):
        self.url   = ollama_url.rstrip("/")
        self.model = model

    async def chat(self, system: str, user: str, format: dict | str | None = None) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        body: dict = {"model": self.model, "messages": messages,
                      "stream": False, "think": False}
        if format is not None:
            body["format"] = format
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(f"{self.url}/api/chat", json=body)
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
        async with httpx.AsyncClient(timeout=60) as c:
            for _ in range(max_rounds):
                r = await c.post(
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
