import asyncio, httpx


class PooledClient:
    """Mixin: a lazily-created, event-loop-aware pooled httpx.AsyncClient.

    Subclasses must set in __init__: self._client = None, self._loop = None,
    and self._client_timeout (float seconds, or None for httpx default).
    In production there is one loop per process so the client is pooled for
    life; under per-function test loops it is transparently recreated.
    """

    _client: httpx.AsyncClient | None
    _loop: object | None
    _client_timeout: float | None

    def _get_client(self) -> httpx.AsyncClient:
        try:
            current_loop: object | None = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        loop_changed = self._loop is not None and self._loop is not current_loop
        if self._client is None or self._client.is_closed or loop_changed:
            kwargs = {} if self._client_timeout is None else {"timeout": self._client_timeout}
            self._client = httpx.AsyncClient(**kwargs)
            self._loop = current_loop
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._loop = None
