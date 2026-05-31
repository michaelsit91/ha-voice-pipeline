import asyncio, logging, re
import httpx

log = logging.getLogger("pipeline")


def _satellite_slug(entity_id: str) -> str:
    """assist_satellite.respeaker_lite_assist_satellite → respeaker_lite"""
    s = re.sub(r"^assist_satellite\.", "", entity_id)
    return re.sub(r"_assist_satellite$", "", s)


def _physical_player_slug(entity_id: str) -> str:
    """media_player.respeaker_lite_media_player(_N)? → respeaker_lite"""
    s = re.sub(r"^media_player\.", "", entity_id)
    return re.sub(r"_media_player(_\d+)?$", "", s)


async def _discover_satellite_players(
    url: str, hdrs: dict, client: httpx.AsyncClient
) -> dict[str, str]:
    """Build {satellite_slug: ma_player_entity_id} from HA /api/states."""
    r = await client.get(f"{url}/api/states", headers=hdrs, timeout=10)
    r.raise_for_status()
    states = r.json()

    satellite_slugs = {
        _satellite_slug(s["entity_id"])
        for s in states
        if s["entity_id"].startswith("assist_satellite.")
    }

    ma_players = [
        s for s in states
        if s["entity_id"].startswith("media_player.")
        and s.get("attributes", {}).get("mass_player_type") == "player"
    ]

    result: dict[str, str] = {}

    # Primary: match via active_queue attribute (set when player is active)
    for ma in ma_players:
        aq = ma.get("attributes", {}).get("active_queue") or ""
        if aq.startswith("media_player."):
            slug = _physical_player_slug(aq)
            if slug in satellite_slugs:
                result[slug] = ma["entity_id"]

    # Fallback: slug match on entity_id for idle players
    for ma in ma_players:
        if ma["entity_id"] in result.values():
            continue
        ma_slug = _physical_player_slug(ma["entity_id"])
        for sat_slug in satellite_slugs:
            if sat_slug not in result and sat_slug == ma_slug:
                result[sat_slug] = ma["entity_id"]

    return result


class MusicAssistantClient:
    def __init__(self, ha_url: str, token: str, config_entry_id: str):
        self._url = ha_url.rstrip("/")
        self._hdrs = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self._config_entry_id = config_entry_id
        self._satellite_map: dict[str, str] = {}
        self._client: httpx.AsyncClient | None = None
        self._loop: object | None = None

    def _get_client(self) -> httpx.AsyncClient:
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

    async def discover(self) -> None:
        """Populate satellite→player map from HA. Call once at startup."""
        self._satellite_map = await _discover_satellite_players(
            self._url, self._hdrs, self._get_client()
        )
        log.info(
            "MUSIC | discovered %d satellite player(s): %s",
            len(self._satellite_map),
            self._satellite_map,
        )

    def resolve_player(self, satellite_slug: str | None) -> str | None:
        """Return MA player entity_id for slug, or first discovered player."""
        if satellite_slug and satellite_slug in self._satellite_map:
            return self._satellite_map[satellite_slug]
        if self._satellite_map:
            return next(iter(self._satellite_map.values()))
        return None

    async def search(
        self,
        name: str,
        media_type: str = "track",
        artist: str | None = None,
        limit: int = 3,
    ) -> list[dict]:
        """Search MA. Returns list of {uri, name, artist} dicts."""
        payload: dict = {
            "config_entry_id": self._config_entry_id,
            "name": name,
            "media_type": [media_type],
            "limit": limit,
        }
        if artist:
            payload["artist"] = artist
        r = await self._get_client().post(
            f"{self._url}/api/services/music_assistant/search",
            headers=self._hdrs,
            json=payload,
            params={"return_response": ""},
            timeout=10,
        )
        r.raise_for_status()
        sr = r.json().get("service_response", {})
        items = sr.get(f"{media_type}s", [])
        results = []
        for item in items:
            uri = item.get("uri", "")
            n = item.get("name", "")
            if item.get("media_type") == "artist":
                a = n
            else:
                artists = item.get("artists", [])
                a = artists[0].get("name", "") if artists else ""
            results.append({"uri": uri, "name": n, "artist": a})
        return results
