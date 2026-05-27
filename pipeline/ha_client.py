import httpx

CONTROLLABLE_DOMAINS = {"light", "switch", "fan", "media_player", "climate", "cover", "input_boolean"}


class HAClient:
    def __init__(self, ha_url: str, token: str):
        self._url = ha_url.rstrip("/")
        self._hdrs = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async def get_entities(self) -> list[dict]:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{self._url}/api/states", headers=self._hdrs, timeout=10)
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
            and s["state"] != "unavailable"  # exclude offline/unreachable devices
        ]
        # Deduplicate by friendly name: when multiple entities share the same name
        # (e.g. light.X and switch.X from Z2M), keep the one with the most attributes
        # since more attributes = more capable (e.g. light with brightness > switch).
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
        async with httpx.AsyncClient() as c:
            r = await c.post(
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
        async with httpx.AsyncClient() as c:
            r = await c.get(
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
            payload["entity_id"] = entity_id  # HA accepts str or list
        if area_id is not None:
            payload["area_id"] = area_id
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{self._url}/api/services/{domain}/{service}",
                headers=self._hdrs,
                json=payload,
                timeout=15,
            )
            r.raise_for_status()
        return r.json() if r.content else {}
