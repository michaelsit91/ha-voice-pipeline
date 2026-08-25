"""Spotify voice playback: fast-path parsing, fuzzy re-rank, resolution chain."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipeline.agents.fast_intent import _match_play
from pipeline.spotify_search import rank_results
from pipeline.agents.executor import _resolve_media, _run_music_step
from pipeline.music_assistant_client import MusicAssistantClient


# ── fast-path play parsing ──────────────────────────────────────────────────

class TestMatchPlay:
    def test_track(self):
        m = _match_play("play blinding lights")
        assert m == {"kind": "music", "query": "blinding lights", "artist": None,
                     "media_type": "track", "needs_player": True}

    def test_track_by_artist(self):
        m = _match_play("play blinding lights by the weeknd")
        assert m["query"] == "blinding lights"
        assert m["artist"] == "the weeknd"

    def test_album(self):
        m = _match_play("play the album random access memories")
        assert m["media_type"] == "album"
        assert m["query"] == "random access memories"

    def test_playlist(self):
        m = _match_play("play playlist workout mix")
        assert m["media_type"] == "playlist"
        assert m["query"] == "workout mix"

    def test_artist_style(self):
        m = _match_play("play some abba")
        assert m["media_type"] == "artist"
        assert m["query"] == "abba"

    def test_songs_by(self):
        m = _match_play("play songs by queen")
        assert m["media_type"] == "artist"
        assert m["query"] == "queen"

    def test_vague_goes_to_llm(self):
        assert _match_play("play some music") is None
        assert _match_play("play something relaxing") is None
        assert _match_play("play anything") is None

    def test_non_play_ignored(self):
        assert _match_play("turn on the lights") is None

    def test_put_on(self):
        assert _match_play("put on hotel california")["query"] == "hotel california"

    def test_trailing_punctuation_stripped(self):
        assert _match_play("play yesterday.")["query"] == "yesterday"


# ── fuzzy re-ranking ────────────────────────────────────────────────────────

def _ma_track(name, artist, id_="id1"):
    return {"name": name, "artist": artist, "uri": f"spotify://track/{id_}"}

class TestRankResults:
    def test_best_acoustic_match_wins(self):
        items = [_ma_track("Blinded by the Light", "Manfred Mann", "a"),
                 _ma_track("Blinding Lights", "The Weeknd", "b")]
        ranked = rank_results("blinding lights", "", items)
        assert ranked[0]["name"] == "Blinding Lights"
        assert ranked[0]["uri"] == "spotify://track/b"

    def test_artist_hint_boosts(self):
        items = [_ma_track("Yesterday", "Boyz II Men", "a"),
                 _ma_track("Yesterday", "The Beatles", "b")]
        ranked = rank_results("yesterday", "the beatles", items)
        assert ranked[0]["uri"] == "spotify://track/b"

    def test_null_items_skipped(self):
        ranked = rank_results("x", "", [None, _ma_track("X", "Y", "a")])
        assert len(ranked) == 1


# ── resolution chain ────────────────────────────────────────────────────────

def _ma(search_results=None, search_exc=None):
    ma = MagicMock(spec=MusicAssistantClient)
    if search_exc:
        ma.search = AsyncMock(side_effect=search_exc)
    else:
        ma.search = AsyncMock(return_value=search_results or [])
    return ma

def _sp_search(results=None, exc=None):
    sp = MagicMock()
    if exc:
        sp.search = AsyncMock(side_effect=exc)
    else:
        sp.search = AsyncMock(return_value=results or [])
    return sp


class TestResolveMedia:
    def test_fuzzy_search_first(self):
        sp = _sp_search([{"uri": "spotify://track/abc",
                          "name": "Blinding Lights", "artist": "The Weeknd", "score": 95}])
        ma = _ma()
        best = asyncio.run(_resolve_media("blinding lights", None, "track", ma, sp))
        assert best["uri"] == "spotify://track/abc"
        assert best["name"] == "Blinding Lights"
        ma.search.assert_not_called()

    def test_falls_back_to_ma_when_fuzzy_search_down(self):
        sp = _sp_search(exc=RuntimeError("MA search unavailable"))
        ma = _ma([{"uri": "spotify--x://track/zzz", "name": "Song", "artist": "A"}])
        best = asyncio.run(_resolve_media("song", None, "track", ma, sp))
        assert best["uri"] == "spotify--x://track/zzz"

    def test_ma_only_when_no_spotify(self):
        ma = _ma([{"uri": "u", "name": "n", "artist": "a"}])
        best = asyncio.run(_resolve_media("q", None, "track", ma, None))
        assert best["uri"] == "u"

    def test_none_when_nothing_found(self):
        assert asyncio.run(_resolve_media("q", None, "track", _ma(), None)) is None


class TestRunMusicStep:
    def _step(self, **kw):
        return {"query": "blinding lights", "artist": None, "media_type": "track",
                "entity_id": "media_player.respeaker", **kw}

    def test_happy_path_speaks_match(self):
        sp = _sp_search([{"uri": "spotify://track/abc",
                          "name": "Blinding Lights", "artist": "The Weeknd", "score": 95}])
        ha = MagicMock(); ha.call_service = AsyncMock()
        out = asyncio.run(_run_music_step(self._step(), ha, _ma(), sp))
        assert out == "Playing Blinding Lights by The Weeknd."
        ha.call_service.assert_awaited_once()
        assert ha.call_service.await_args.kwargs["media_id"] == "spotify://track/abc"

    def test_direct_uri_rejected_falls_back_to_ma_uri(self):
        sp = _sp_search([{"uri": "spotify://track/abc",
                          "name": "Blinding Lights", "artist": "The Weeknd", "score": 95}])
        ma = _ma([{"uri": "spotify--x://track/abc", "name": "Blinding Lights",
                   "artist": "The Weeknd"}])
        ha = MagicMock()
        ha.call_service = AsyncMock(side_effect=[RuntimeError("bad media_id"), None])
        out = asyncio.run(_run_music_step(self._step(), ha, ma, sp))
        assert out == "Playing Blinding Lights by The Weeknd."
        assert ha.call_service.await_count == 2
        assert ha.call_service.await_args.kwargs["media_id"] == "spotify--x://track/abc"

    def test_no_player_spoken_error(self):
        out = asyncio.run(_run_music_step(self._step(entity_id=None),
                                          MagicMock(), _ma(), None))
        assert "speaker" in out

    def test_not_found_spoken_error(self):
        ha = MagicMock(); ha.call_service = AsyncMock()
        out = asyncio.run(_run_music_step(self._step(), ha, _ma(), None))
        assert out.startswith("Sorry, I couldn't find")


# ── player re-discovery ─────────────────────────────────────────────────────

class TestResolvePlayerFresh:
    def _client(self, mapping):
        c = MusicAssistantClient("http://ha", "tok", "entry")
        c._satellite_map = dict(mapping)
        return c

    def test_known_slug_no_rediscovery(self):
        c = self._client({"respeaker_lite": "media_player.ma_1"})
        c.discover = AsyncMock()
        out = asyncio.run(c.resolve_player_fresh("respeaker_lite"))
        assert out == "media_player.ma_1"
        c.discover.assert_not_called()

    def test_unknown_slug_rediscovers(self):
        c = self._client({})
        async def _found():
            c._satellite_map = {"office": "media_player.ma_2"}
        c.discover = AsyncMock(side_effect=_found)
        out = asyncio.run(c.resolve_player_fresh("office"))
        assert out == "media_player.ma_2"
        c.discover.assert_awaited_once()

    def test_unknown_after_rediscovery_returns_none_not_wrong_room(self):
        c = self._client({"other_room": "media_player.ma_9"})
        c.discover = AsyncMock()
        out = asyncio.run(c.resolve_player_fresh("bedroom"))
        assert out is None  # never route music to an arbitrary room
