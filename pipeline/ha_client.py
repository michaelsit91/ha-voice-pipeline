import asyncio, httpx

CONTROLLABLE_DOMAINS = {"light", "switch", "fan", "media_player", "climate", "cover", "input_boolean"}


class HAClient:
    def __init__(self, ha_url: str, token: str):
        self._url = ha_url.rstrip("/")
        self._hdrs = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        self._client: httpx.AsyncClient | None = None
        self._loop: object | None = None  # tracks which event loop owns _client

    def _get_client(self) -> httpx.AsyncClient:
        """Return the shared client, creating it lazily and recreating on loop change.

        In production there is one event loop per process lifetime, so the client
        is created once and pooled indefinitely. In tests with per-function event
        loops, the client is transparently recreated when the loop changes.
        Only recreates on loop change when a loop was previously tracked
        (avoids overwriting a test-injected mock when _loop is still None).
        """
        try:
            current_loop: object | None = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        loop_changed = self._loop is not None and self._loop is not current_loop
        if self._client is None or self._client.is_closed or loop_changed:
            self._client = httpx.AsyncClient()
            self._loop = current_loop
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._loop = None

    async def get_entities(self) -> list[dict]:
        r = await self._get_client().get(f"{self._url}/api/states", headers=self._hdrs, timeout=10)
        r.raise_for_status()
        raw = [
            {
                "entity_id": s["entity_id"],
                "name": s["attributes"].get("friendly_name", s["entity_id"]),
                "state": s["state"],
                "_attr_count": len(s.get("attributes", {})),
            }
            for s in r.json()
            if s["entity_id"].split(".")[0] in CONTROLLABLE_DOMAINS
            and s["state"] != "unavailable"
        ]
        by_name: dict[str, dict] = {}
        for e in raw:
            existing = by_name.get(e["name"])
            if existing is None or e["_attr_count"] > existing["_attr_count"]:
                by_name[e["name"]] = e
        return [{"entity_id": e["entity_id"], "name": e["name"], "state": e["state"]}
                for e in by_name.values()]

    async def get_areas(self) -> list[dict]:
        _TMPL = (
            '{% set r = namespace(a=[]) %}'
            '{% for aid in areas() %}'
            '{% set r.a = r.a + [{"area_id": aid, "name": area_name(aid)}] %}'
            '{% endfor %}{{ r.a | tojson }}'
        )
        r = await self._get_client().post(
            f"{self._url}/api/template",
            headers=self._hdrs,
            json={"template": _TMPL},
            timeout=10,
        )
        if r.status_code in (404, 400):
            return []
        r.raise_for_status()
        import json
        return json.loads(r.text)

    async def get_state(self, entity_id: str) -> dict:
        r = await self._get_client().get(
            f"{self._url}/api/states/{entity_id}",
            headers=self._hdrs,
            timeout=10,
        )
        r.raise_for_status()
        s = r.json()
        return {
            "entity_id": s["entity_id"],
            "state": s["state"],
            "attributes": s.get("attributes", {}),
        }

    async def call_service(
        self,
        domain: str,
        service: str,
        entity_id: str | list[str] | None = None,
        *,
        area_id: str | None = None,
        **kwargs,
    ) -> dict:
        payload: dict = {**kwargs}
        if entity_id is not None:
            payload["entity_id"] = entity_id
        if area_id is not None:
            payload["area_id"] = area_id
        r = await self._get_client().post(
            f"{self._url}/api/services/{domain}/{service}",
            headers=self._hdrs,
            json=payload,
            timeout=15,
        )
        r.raise_for_status()
        return r.json() if r.content else {}
