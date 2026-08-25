"""Fuzzy music search via Music Assistant, re-ranked with rapidfuzz.

MA's search service fans out to its enabled providers (Spotify included) using
MA's own auth, so the pipeline never touches Spotify tokens directly — two
clients refreshing the same PKCE refresh-token chain is what got the previous
token revoked. We ask MA for a wide result set and re-rank it against the heard
query so the best acoustic match wins even when STT misheard the words.
"""
from __future__ import annotations

import logging

from pipeline.music_assistant_client import MusicAssistantClient

log = logging.getLogger("pipeline")

_SEARCH_LIMIT = 10


def _similarity(a: str, b: str) -> float:
    """Fuzzy similarity 0–100 (rapidfuzz in prod, difflib fallback)."""
    try:
        from rapidfuzz import fuzz
        return fuzz.token_set_ratio(a, b)
    except ImportError:
        import difflib
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100


def rank_results(query: str, artist_hint: str, items: list[dict]) -> list[dict]:
    """Re-rank MA search items ({uri, name, artist}) by fuzzy similarity to the
    heard query.

    Returns [{uri, name, artist, score}] best-first. artist_hint (from
    'play X by Y') boosts items whose artist matches Y.
    """
    ranked = []
    for item in items:
        if not item:
            continue
        name = item.get("name", "")
        artist = item.get("artist", "")
        score = max(_similarity(query, name),
                    _similarity(query, f"{name} {artist}"))
        if artist_hint:
            score = 0.7 * score + 0.3 * _similarity(artist_hint, artist)
        ranked.append({
            "uri": item.get("uri", ""),
            "name": name,
            "artist": artist,
            "score": round(score, 1),
        })
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked


class SpotifySearch:
    """Fuzzy catalog search backed by Music Assistant's search service."""

    def __init__(self, ma: MusicAssistantClient) -> None:
        self._ma = ma

    async def search(self, query: str, media_type: str = "track",
                     artist: str | None = None) -> list[dict]:
        """Best-first [{uri, name, artist, score}]. Raises on failure so the
        caller can fall back to a raw Music Assistant query."""
        items = await self._ma.search(query, media_type=media_type,
                                      artist=artist, limit=_SEARCH_LIMIT)
        return rank_results(query, artist or "", items)
