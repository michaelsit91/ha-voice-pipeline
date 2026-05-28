"""Spotify Connect sync: after MA starts playing a Spotify track, transfer it to
the librespot device so the phone's Spotify app shows what's playing.

How it works
------------
MA's Spotify streaming (OAuth) and its Spotify Connect plugin (librespot) are two
separate systems.  When MA plays via its queue the phone doesn't see it.  After
play starts we:

1. Extract the Spotify track ID from the MA search-result URI (available immediately).
2. Authenticate to Spotify using the refresh token from MA's encrypted settings.
3. After a brief delay (MA needs ~1-2 s to start its queue stream), call
   PUT /v1/me/player/play with the track URI and the librespot device ID.
4. Spotify streams to librespot; MA detects the "playing" event, calls
   select_source(respeaker, spotify_connect), and the phone sees it.

MA uses PKCE flow (no client secret), so we can refresh tokens with client_id only.

Token rotation
--------------
Spotify PKCE can rotate the refresh token on each use.  MA handles this by writing
the new token back to settings.json immediately after every refresh (its own
`_update_config_value` call).  We do exactly the same: after a successful token
exchange we write the new refresh token back to settings.json with the same Fernet
encryption MA uses.  This keeps the token chain coherent:

  settings.json  ←──written by whoever refreshes last
        ↑ read              ↑ write
   MA (every ~50 min)   us (every ~60 min, access-token cache)

Because the access token lasts 3600 s and we cache it, we only call the refresh
endpoint once per hour — well within MA's own in-memory cache window.

Requires the MA data directory to be mounted *read-write* in docker-compose.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import aiohttp

log = logging.getLogger("pipeline")

_SPOTIFY_TOKEN_URL   = "https://accounts.spotify.com/api/token"
_SPOTIFY_DEVICES_URL = "https://api.spotify.com/v1/me/player/devices"
_SPOTIFY_PLAY_URL    = "https://api.spotify.com/v1/me/player/play"

# MA's global Spotify app client ID (PKCE — no client_secret needed)
_MA_CLIENT_ID = "2eb96f9b37494be1824999d58028a305"

# Wait this many seconds after play_media before triggering librespot takeover.
# Gives MA enough time to start its queue stream before we interrupt it.
_SYNC_DELAY_SECONDS = 2.0

# Regex to pull the bare track ID out of any MA Spotify URI, e.g.
#   spotify--GoM6sQqz://track/0VjIjW4GlUZAMYd2vXMi3b
_SPOTIFY_TRACK_ID_RE = re.compile(r"spotify[^/]*://track/([A-Za-z0-9]+)")


def extract_spotify_track_id(ma_uri: str) -> str | None:
    """Return the Spotify track ID from an MA URI, or None if not a Spotify track."""
    m = _SPOTIFY_TRACK_ID_RE.search(ma_uri)
    return m.group(1) if m else None


class SpotifyConnectSync:
    """Bridges MA queue playback to Spotify Connect so the phone Spotify app syncs."""

    def __init__(
        self,
        ma_settings_path: str,
        librespot_device_name: str = "ReSpeaker Lite",
    ) -> None:
        self._settings_path = Path(ma_settings_path)
        self._device_name = librespot_device_name
        self._refresh_token: str | None = None
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0
        # Cached aiohttp session — created on first use, closed on GC
        self._session: aiohttp.ClientSession | None = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _session_or_create(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _fernet_and_sp_key(self) -> tuple[Any, dict[str, Any], str, str]:
        """Return (Fernet, settings_dict, sp_provider_key, server_id).

        Shared by _load_refresh_token and _write_refresh_token.
        """
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:
            raise RuntimeError(
                "cryptography package required: pip install cryptography"
            ) from exc

        settings: dict[str, Any] = json.loads(self._settings_path.read_text())
        server_id: str = settings["server_id"]
        fernet_key = base64.urlsafe_b64encode(server_id.encode()[:32])
        fernet = Fernet(fernet_key)

        providers: dict[str, Any] = settings.get("providers", {})
        sp_key = next(
            (k for k in providers if k.startswith("spotify--") and "connect" not in k),
            None,
        )
        if sp_key is None:
            raise RuntimeError("No Spotify provider found in MA settings.json")

        return fernet, settings, sp_key, server_id

    def _load_refresh_token(self) -> str:
        """Decrypt the Spotify refresh token from MA's settings.json."""
        fernet, settings, sp_key, _ = self._fernet_and_sp_key()
        providers: dict[str, Any] = settings.get("providers", {})

        encrypted: str = (
            providers[sp_key].get("values", {}).get("refresh_token_global", "")
        )
        if not encrypted:
            raise RuntimeError("No Spotify refresh_token_global in MA settings.json")
        if encrypted.startswith("_encrypted_"):
            encrypted = encrypted[len("_encrypted_"):]

        return fernet.decrypt(encrypted.encode()).decode()

    def _write_refresh_token(self, new_token: str) -> None:
        """Write a rotated refresh token back into MA's settings.json.

        Uses the same Fernet key and ``_encrypted_`` prefix that MA itself uses,
        so the next time MA (or we) read the file they find a valid token.
        The write is atomic: we stage to a temp file then rename.
        """
        try:
            fernet, settings, sp_key, _ = self._fernet_and_sp_key()
        except Exception as exc:
            log.warning("SPOTIFY_SYNC | could not open settings for token write-back: %s", exc)
            return

        encrypted_value = "_encrypted_" + fernet.encrypt(new_token.encode()).decode()
        settings["providers"][sp_key].setdefault("values", {})["refresh_token_global"] = (
            encrypted_value
        )

        # Atomic write: stage beside the target, then rename
        tmp_path = self._settings_path.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(json.dumps(settings))
            os.replace(tmp_path, self._settings_path)
            log.debug("SPOTIFY_SYNC | rotated refresh token written back to settings.json")
        except OSError as exc:
            log.warning("SPOTIFY_SYNC | token write-back failed (read-only volume?): %s", exc)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    async def _get_access_token(self) -> str:
        """Return a valid Spotify access token, refreshing if expired or missing."""
        if self._access_token and time.time() < self._token_expires_at - 60:
            return self._access_token

        if self._refresh_token is None:
            self._refresh_token = self._load_refresh_token()

        session = self._session_or_create()
        async with session.post(
            _SPOTIFY_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
                "client_id": _MA_CLIENT_ID,
            },
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(
                    f"Spotify token refresh failed ({resp.status}): {body[:200]}"
                )
            data: dict[str, Any] = await resp.json()

        self._access_token = data["access_token"]
        self._token_expires_at = time.time() + data.get("expires_in", 3600)
        if new_rt := data.get("refresh_token"):
            # Spotify rotated the token — update our cache AND write back to
            # settings.json so MA finds a valid token on its next refresh cycle.
            self._refresh_token = new_rt
            self._write_refresh_token(new_rt)

        log.debug("SPOTIFY_SYNC | access token refreshed (expires in ~%ds)", data.get("expires_in", 3600))
        return self._access_token  # type: ignore[return-value]

    async def _get_device_id(self) -> str | None:
        """Return the Spotify device ID for the configured librespot instance."""
        token = await self._get_access_token()
        session = self._session_or_create()
        async with session.get(
            _SPOTIFY_DEVICES_URL,
            headers={"Authorization": f"Bearer {token}"},
        ) as resp:
            if resp.status != 200:
                log.warning("SPOTIFY_SYNC | devices fetch failed: HTTP %s", resp.status)
                return None
            data = await resp.json()

        for device in data.get("devices", []):
            if device.get("name") == self._device_name:
                log.debug(
                    "SPOTIFY_SYNC | found device %r → id=%s",
                    self._device_name, device["id"],
                )
                return device["id"]

        log.warning(
            "SPOTIFY_SYNC | device %r not found. Available: %s",
            self._device_name,
            [d.get("name") for d in data.get("devices", [])],
        )
        return None

    # ── Public API ────────────────────────────────────────────────────────────

    async def sync_track(self, spotify_track_id: str) -> bool:
        """Play a Spotify track on the librespot device.

        Blocks until the Spotify API call completes (or fails).
        Returns True if the play command was accepted.
        """
        try:
            device_id = await self._get_device_id()
            if not device_id:
                return False

            token = await self._get_access_token()
            track_uri = f"spotify:track:{spotify_track_id}"
            session = self._session_or_create()

            async with session.put(
                f"{_SPOTIFY_PLAY_URL}?device_id={device_id}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={"uris": [track_uri]},
            ) as resp:
                if resp.status not in (200, 204):
                    body = await resp.text()
                    log.warning(
                        "SPOTIFY_SYNC | play failed HTTP %s: %s", resp.status, body[:200]
                    )
                    return False

            log.info(
                "SPOTIFY_SYNC | %s → %r (device %s)",
                track_uri, self._device_name, device_id,
            )
            return True

        except Exception as exc:
            log.warning("SPOTIFY_SYNC | error: %s", exc)
            return False

    async def schedule_sync(self, spotify_track_id: str) -> None:
        """Fire-and-forget: wait _SYNC_DELAY_SECONDS then push to Spotify Connect."""
        await asyncio.sleep(_SYNC_DELAY_SECONDS)
        await self.sync_track(spotify_track_id)
