"""Tests for music stop/pause voice commands."""
import pytest
import re
from pipeline.agents.planner import _HESITATION_PATTERNS


# ── Task 1: hesitation pattern ────────────────────────────────────────────────

def test_stop_the_music_not_hesitation():
    assert not _HESITATION_PATTERNS.search("stop the music")

def test_stop_playing_not_hesitation():
    assert not _HESITATION_PATTERNS.search("stop playing")

def test_stop_that_song_not_hesitation():
    assert not _HESITATION_PATTERNS.search("stop that song")

def test_stop_the_track_not_hesitation():
    assert not _HESITATION_PATTERNS.search("stop the track")

def test_bare_stop_is_still_hesitation():
    assert _HESITATION_PATTERNS.search("stop")

def test_stop_with_intermediate_words_not_hesitation():
    # "(?:\w+\s+)*" allows any words between stop and the music keyword
    assert not _HESITATION_PATTERNS.search("stop all the music")

def test_stop_alone_in_context_is_still_hesitation():
    assert _HESITATION_PATTERNS.search("please stop")
