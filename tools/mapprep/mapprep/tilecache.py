"""Import imagery from a pre-downloaded tile cache into the normalized cache.

Two layouts are supported:

  - ``qgis_xyz``: QGIS "Generate XYZ Tiles (Directory)" export, ``{z}/{x}/{y}.image``
  - ``sasplanet``: SAS.Planet cache, ``z{z+1}/{x}/{y}.jpg`` (its zoom is standard
    zoom + 1; a ``z_offset`` of 1 compensates)

The importer only copies bytes and records provenance; no re-fetch, no re-encode.
"""

from __future__ import annotations

import re
from pathlib import Path

from .fetch import find_cached_tile, record_tile_meta
from .providers import Provider

_TILE_RE = re.compile(
    r"(?:^|[/\\])z?(?P<z>\d{1,2})[/\\](?P<x>\d+)[/\\](?P<y>\d+)(?:\.(?P<ext>jpe?g|png))$",
    re.IGNORECASE,
)

LAYOUTS = {
    "qgis_xyz": {"z_offset": 0, "exts": ("jpg", "jpeg", "png", "image")},
    "sasplanet": {"z_offset": 1, "exts": ("jpg", "jpeg", "png")},
}


def import_cache(
    src_root: Path,
    provider: Provider,
    cache_root: Path,
    layout: str = "sasplanet",
    z_offset: int | None = None,
) -> int:
    layouts = {name: spec["exts"] for name, spec in LAYOUTS.items()}
    if layout not in layouts:
        raise ValueError(f"unknown layout {layout!r}; known: {sorted(layouts)}")
    if z_offset is None:
        z_offset = LAYOUTS[layout]["z_offset"]
    exts = set(layouts[layout])

    imported = 0
    for src_file in sorted(src_root.rglob("*")):
        if not src_file.is_file() or src_file.suffix.lower().lstrip(".") not in exts:
            continue
        match = _TILE_RE.search(str(src_file))
        if not match:
            continue
        z = int(match.group("z")) - z_offset
        x = int(match.group("x"))
        y = int(match.group("y"))
        if z < provider.min_zoom or z > provider.max_zoom:
            continue
        if find_cached_tile(provider, cache_root, x, y, z) is not None:
            continue
        dst = cache_root / provider.id / f"{z}/{x}/{y}{src_file.suffix.lower()}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src_file.read_bytes())
        record_tile_meta(provider, cache_root, x, y, z, source=str(src_file))
        imported += 1
    return imported
