"""Tile fetching with a persistent on-disk cache.

Rebuilding the same area never re-downloads tiles: every fetched tile lands in
a normalized cache layout ({root}/{provider}/{z}/{x}/{y}.jpg) and later builds
run from it. Tiles that returned 404 leave a .missing marker so a re-run does
not hit the network again and records the hole honestly.

On this dev host IPv6 connects hang while IPv4 works; urllib resolves the IPv6
address first and stalls, so the fetcher pins socket resolution to IPv4.
"""

from __future__ import annotations

import contextlib
import json
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .providers import Provider

USER_AGENT = "geoloc-mapprep/0.1 (internal dev)"


class TileNotFoundError(Exception):
    pass


class TileFetchError(Exception):
    pass


@contextlib.contextmanager
def force_ipv4():
    real_getaddrinfo = socket.getaddrinfo

    def v4_only(host, port, family=0, *args, **kwargs):
        if family in (0, socket.AF_UNSPEC):
            results = real_getaddrinfo(host, port, socket.AF_INET, *args, **kwargs)
            if results:
                return results
        return real_getaddrinfo(host, port, family, *args, **kwargs)

    socket.getaddrinfo = v4_only
    try:
        yield
    finally:
        socket.getaddrinfo = real_getaddrinfo


def _http_get(url: str, timeout_s: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise TileNotFoundError(url) from exc
        raise TileFetchError(f"HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise TileFetchError(f"{exc.reason} for {url}") from exc


def http_get_with_retries(url: str, timeout_s: float = 15.0, retries: int = 3) -> bytes:
    delay = 0.5
    for attempt in range(retries):
        try:
            return _http_get(url, timeout_s)
        except TileNotFoundError:
            raise
        except TileFetchError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2.0
    raise TileFetchError(url)


def cache_root_for(provider: Provider, cache_root: Path) -> Path:
    return cache_root / provider.id


def tile_path(provider: Provider, cache_root: Path, x: int, y: int, z: int) -> Path:
    return cache_root_for(provider, cache_root) / provider.cache_rel_path(x, y, z)


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def find_cached_tile(provider: Provider, cache_root: Path, x: int, y: int, z: int) -> Path | None:
    base = cache_root_for(provider, cache_root) / f"{z}/{x}/{y}"
    for ext in _IMAGE_EXTS:
        candidate = base.with_suffix(ext)
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def missing_marker(provider: Provider, cache_root: Path, x: int, y: int, z: int) -> Path:
    return cache_root_for(provider, cache_root) / f"{z}/{x}/{y}.missing"


def meta_path(provider: Provider, cache_root: Path) -> Path:
    return cache_root_for(provider, cache_root) / "tiles_meta.json"


def load_meta(provider: Provider, cache_root: Path) -> dict:
    path = meta_path(provider, cache_root)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_meta(provider: Provider, cache_root: Path, meta: dict) -> None:
    path = meta_path(provider, cache_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    tmp.replace(path)


def record_tile_meta(
    provider: Provider,
    cache_root: Path,
    x: int,
    y: int,
    z: int,
    *,
    capture_date: str | None = None,
    source: str = "http",
) -> None:
    meta = load_meta(provider, cache_root)
    key = f"{z}/{x}/{y}"
    entry = dict(meta.get(key, {}))
    entry["source"] = source
    entry["downloaded_at"] = datetime.now(timezone.utc).isoformat()
    if capture_date:
        entry["capture_date"] = capture_date
    meta[key] = entry
    save_meta(provider, cache_root, meta)


def fetch_tile(
    provider: Provider,
    x: int,
    y: int,
    z: int,
    cache_root: Path,
    *,
    offline: bool = False,
    timeout_s: float = 15.0,
    retries: int = 3,
) -> bool:
    """Ensure the tile is in the cache. Returns True on success, False if missing."""
    if find_cached_tile(provider, cache_root, x, y, z) is not None:
        return True
    marker = missing_marker(provider, cache_root, x, y, z)
    if marker.exists():
        return False
    if offline:
        return False
    url = provider.tile_url(x, y, z)
    try:
        data = http_get_with_retries(url, timeout_s=timeout_s, retries=retries)
    except TileNotFoundError:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        return False
    path = tile_path(provider, cache_root, x, y, z)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    record_tile_meta(provider, cache_root, x, y, z)
    return True


def clear_missing_markers(provider: Provider, cache_root: Path) -> int:
    root = cache_root_for(provider, cache_root)
    count = 0
    for marker in root.glob("**/*.missing"):
        marker.unlink()
        count += 1
    return count
