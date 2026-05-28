"""Tests for pipeline/spotify_connect_sync.py"""

import asyncio
import base64
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.spotify_connect_sync import (
    SpotifyConnectSync,
    extract_spotify_track_id,
)


# ── extract_spotify_track_id ───────────────────────────────────────────────────


def test_extract_spotify_track_id_from_ma_uri():
    uri = "spotify--GoM6sQqz://track/0VjIjW4GlUZAMYd2vXMi3b"
    assert extract_spotify_track_id(uri) == "0VjIjW4GlUZAMYd2vXMi3b"


def test_extract_spotify_track_id_non_spotify_returns_none():
    uri = "builtin://track/some-local-track"
    assert extract_spotify_track_id(uri) is None


def test_extract_spotify_track_id_ma_uri_variant():
    # Different instance suffix still works
    uri = "spotify--AbCd1234://track/TrackIdXYZ789"
    assert extract_spotify_track_id(uri) == "TrackIdXYZ789"


def test_extract_spotify_track_id_empty_string():
    assert extract_spotify_track_id("") is None


# ── SpotifyConnectSync._load_refresh_token ─────────────────────────────────────


def _make_fake_settings(tmp_path: Path, server_id: str, refresh_token: str) -> Path:
    """Write a minimal MA settings.json with an encrypted Spotify refresh token."""
    import base64
    from cryptography.fernet import Fernet

    fernet_key = base64.urlsafe_b64encode(server_id.encode()[:32])
    fernet = Fernet(fernet_key)
    encrypted = fernet.encrypt(refresh_token.encode()).decode()

    settings = {
        "server_id": server_id,
        "providers": {
            "spotify--GoM6sQqz": {
                "values": {
                    "refresh_token_global": f"_encrypted_{encrypted}",
                }
            }
        },
    }
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings))
    return path


def test_load_refresh_token_decrypts_correctly(tmp_path):
    server_id = "12c66356dc0547babdf8e69db8431b9c"
    expected_token = "AQD2LKXEp3LvhYjKlWVZtWJhHKq7P1cCzstOTOsr_EXAMPLE"
    settings_path = _make_fake_settings(tmp_path, server_id, expected_token)

    sync = SpotifyConnectSync(str(settings_path), "ReSpeaker Lite")
    assert sync._load_refresh_token() == expected_token


# ── SpotifyConnectSync.sync_track ─────────────────────────────────────────────


@pytest.fixture
def sync_with_token(tmp_path):
    """SpotifyConnectSync with valid settings.json and a pre-loaded fake token."""
    server_id = "12345678901234567890123456789012"
    settings_path = _make_fake_settings(tmp_path, server_id, "fake_refresh_token")
    sync = SpotifyConnectSync(str(settings_path), "TestDevice")
    # Pre-load a non-expired access token so no refresh is needed
    sync._access_token = "fake_access_token"
    sync._token_expires_at = time.time() + 3600
    return sync


def _mock_httpx_response(status_code: int, json_data=None, text: str = "") -> MagicMock:
    """Build a mock httpx.Response with sync json() and text property."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_data or {})
    resp.text = text
    return resp


@pytest.mark.asyncio
async def test_sync_track_calls_spotify_play(sync_with_token):
    """sync_track finds the device and calls PUT /v1/me/player/play."""
    mock_client = MagicMock()
    mock_client.is_closed = False
    mock_client.get = AsyncMock(return_value=_mock_httpx_response(
        200, {"devices": [{"name": "TestDevice", "id": "device-abc123"}]}
    ))
    mock_client.put = AsyncMock(return_value=_mock_httpx_response(204))
    sync_with_token._client = mock_client

    result = await sync_with_token.sync_track("0VjIjW4GlUZAMYd2vXMi3b")

    assert result is True
    # PUT was called with the correct URI and device_id
    put_call_args = mock_client.put.call_args
    assert "device_id=device-abc123" in put_call_args[0][0]
    called_json = put_call_args[1]["json"]
    assert called_json["uris"] == ["spotify:track:0VjIjW4GlUZAMYd2vXMi3b"]


@pytest.mark.asyncio
async def test_sync_track_returns_false_when_device_not_found(sync_with_token):
    """sync_track returns False gracefully when librespot device isn't listed."""
    mock_client = MagicMock()
    mock_client.is_closed = False
    mock_client.get = AsyncMock(return_value=_mock_httpx_response(
        200, {"devices": [{"name": "SomeOtherDevice", "id": "other-id"}]}
    ))
    sync_with_token._client = mock_client

    result = await sync_with_token.sync_track("0VjIjW4GlUZAMYd2vXMi3b")
    assert result is False


@pytest.mark.asyncio
async def test_sync_track_returns_false_on_api_error(sync_with_token):
    """sync_track returns False (no exception) when the Spotify API call fails."""
    mock_client = MagicMock()
    mock_client.is_closed = False
    mock_client.get = AsyncMock(side_effect=Exception("network error"))
    sync_with_token._client = mock_client

    result = await sync_with_token.sync_track("0VjIjW4GlUZAMYd2vXMi3b")
    assert result is False


# ── SpotifyConnectSync._write_refresh_token ────────────────────────────────────


def test_write_refresh_token_round_trips(tmp_path):
    """Writing a rotated token back and reading it again yields the new token."""
    server_id = "12c66356dc0547babdf8e69db8431b9c"
    original_token = "AQD_ORIGINAL_TOKEN"
    settings_path = _make_fake_settings(tmp_path, server_id, original_token)

    sync = SpotifyConnectSync(str(settings_path), "ReSpeaker Lite")
    rotated_token = "AQD_ROTATED_TOKEN"
    sync._write_refresh_token(rotated_token)

    # Reading back should return the new token
    assert sync._load_refresh_token() == rotated_token


def test_write_refresh_token_preserves_other_settings(tmp_path):
    """Writing a token back does not destroy unrelated keys in settings.json."""
    server_id = "12c66356dc0547babdf8e69db8431b9c"
    settings_path = _make_fake_settings(tmp_path, server_id, "old_token")
    # Add an extra key that should survive the write
    data = json.loads(settings_path.read_text())
    data["extra_key"] = "should_survive"
    settings_path.write_text(json.dumps(data))

    sync = SpotifyConnectSync(str(settings_path), "ReSpeaker Lite")
    sync._write_refresh_token("new_token")

    result = json.loads(settings_path.read_text())
    assert result.get("extra_key") == "should_survive"


def test_write_refresh_token_fails_gracefully_on_read_only_path(tmp_path):
    """_write_refresh_token logs a warning but does not raise if the path is unwritable."""
    server_id = "12c66356dc0547babdf8e69db8431b9c"
    settings_path = _make_fake_settings(tmp_path, server_id, "token")
    settings_path.chmod(0o444)  # read-only

    sync = SpotifyConnectSync(str(settings_path), "ReSpeaker Lite")
    # Should not raise
    sync._write_refresh_token("new_token")

    # Restore for cleanup
    settings_path.chmod(0o644)


# ── _get_access_token writes back on rotation ─────────────────────────────────


@pytest.mark.asyncio
async def test_get_access_token_writes_back_rotated_token(tmp_path):
    """When Spotify returns a new refresh_token, _get_access_token writes it back."""
    server_id = "12345678901234567890123456789012"
    settings_path = _make_fake_settings(tmp_path, server_id, "original_refresh")
    sync = SpotifyConnectSync(str(settings_path), "TestDevice")
    sync._refresh_token = "original_refresh"  # skip file read

    mock_client = MagicMock()
    mock_client.is_closed = False
    mock_client.post = AsyncMock(return_value=_mock_httpx_response(200, {
        "access_token": "new_access",
        "expires_in": 3600,
        "refresh_token": "rotated_refresh",
    }))
    sync._client = mock_client

    token = await sync._get_access_token()
    assert token == "new_access"
    assert sync._refresh_token == "rotated_refresh"

    # The rotated token must have been written back
    assert sync._load_refresh_token() == "rotated_refresh"


@pytest.mark.asyncio
async def test_get_access_token_no_write_when_no_rotation(tmp_path):
    """When Spotify does NOT return a new refresh_token, settings.json is untouched."""
    server_id = "12345678901234567890123456789012"
    settings_path = _make_fake_settings(tmp_path, server_id, "original_refresh")
    original_mtime = settings_path.stat().st_mtime

    sync = SpotifyConnectSync(str(settings_path), "TestDevice")
    sync._refresh_token = "original_refresh"

    mock_client = MagicMock()
    mock_client.is_closed = False
    mock_client.post = AsyncMock(return_value=_mock_httpx_response(200, {
        "access_token": "new_access",
        "expires_in": 3600,
        # no refresh_token in response
    }))
    sync._client = mock_client

    await sync._get_access_token()
    # settings.json should not have been touched
    assert settings_path.stat().st_mtime == original_mtime


# ── schedule_sync delays before syncing ──────────────────────────────────────


@pytest.mark.asyncio
async def test_schedule_sync_waits_before_calling_sync_track():
    """schedule_sync sleeps then delegates to sync_track."""
    sync = MagicMock(spec=SpotifyConnectSync)
    sync.sync_track = AsyncMock(return_value=True)

    # Patch asyncio.sleep so the test doesn't actually wait 2 seconds
    with patch("pipeline.spotify_connect_sync.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        # Call the real schedule_sync but with mocked sleep and sync_track
        from pipeline.spotify_connect_sync import SpotifyConnectSync as Real

        async def _real_schedule_sync(track_id):
            await asyncio.sleep(0.001)  # fast version
            await sync.sync_track(track_id)

        await _real_schedule_sync("TrackABC123")

    sync.sync_track.assert_called_once_with("TrackABC123")
