import os, time, httpx
from pipeline._http import PooledClient

CONTROLLABLE_DOMAINS = {"light", "switch", "fan", "media_player", "climate", "cover", "input_boolean"}


class HAClient(PooledClient):
    def __init__(self, ha_url: str, token: str):
        self._url = ha_url.rstrip("/")
        self._hdrs = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        self._client: httpx.AsyncClient | None = None
        self._loop: object | None = None  # tracks which event loop owns _client
        self._client_timeout = None  # HA passes per-call timeouts
        self._cache_ttl = float(os.getenv("HA_CACHE_TTL_S", "10"))
        self._areas_ttl = float(os.getenv("HA_AREAS_TTL_S", "300"))
        self._cache: dict[str, tuple[float, object]] = {}

    def clear_cache(self) -> None:
        """Invalidate cached entities/areas so the next call refetches."""
        self._cache.clear()

    async def _cached(self, key: str, fetch, ttl: float) -> object:
        """Return a TTL-cached fetch result. ttl<=0 disables caching."""
        if ttl > 0:
            hit = self._cache.get(key)
            if hit is not None and (time.monotonic() - hit[0]) < ttl:
                return hit[1]
        value = await fetch()
        if ttl > 0:
            self._cache[key] = (time.monotonic(), value)
        return value

    async def get_entities(self) -> list[dict]:
        return await self._cached("entities", self._fetch_entities, self._cache_ttl)

    async def get_areas(self) -> list[dict]:
        return await self._cached("areas", self._fetch_areas, self._areas_ttl)

    async def _fetch_entities(self) -> list[dict]:
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

    async def _fetch_areas(self) -> list[dict]:
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
