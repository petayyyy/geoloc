"""OSM semantic layer for the geopack (T06).

Fetches OSM ways over the corridor via the Overpass API (or accepts a
pre-fetched fragment), rasterizes them to the terrain-class grid, and writes
the result as a uint8 COG at the requested GSD (default 1 m/px).

The class table and rasterization rules live in osm.py; this module owns the
fetch + grid + COG plumbing so the whole geopack builds from one command.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer

from .georef import PixelGrid
from .osm import rasterize_overpass

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Public Overpass instances, tried in order. The main one runs a small fixed
# pool of query slots and answers "504 Gateway Timeout" (not 429) the moment
# they are all busy -- a transient load signal, not a bad query, and the
# reason a geopack build could die after both ortho layers were already built.
OVERPASS_URLS = (
    OVERPASS_URL,
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)

# Server-side conditions worth another attempt. 400 is NOT here on purpose:
# that means the query itself is malformed, and retrying just wastes slots.
OVERPASS_RETRY_CODES = frozenset({429, 500, 502, 503, 504})

USER_AGENT = "geoloc-mapprep/0.1 (internal dev)"


class OsmFetchError(RuntimeError):
    pass


@dataclass
class SemanticResult:
    grid: PixelGrid
    semantic_path: Path
    gsd_m: float
    extract_date: str | None
    source: str


def overpass_query(bounds_wgs84: dict) -> str:
    south, west, north, east = (
        bounds_wgs84["south"],
        bounds_wgs84["west"],
        bounds_wgs84["north"],
        bounds_wgs84["east"],
    )
    # Overpass QL bbox filters take four bare numbers, not a quoted string --
    # way["k"]("s,w,n,e") is a syntax error (HTTP 400), way["k"](s,w,n,e) is not.
    bbox = f"{south},{west},{north},{east}"
    return (
        "[out:json][timeout:60];\n"
        "(\n"
        f'  way["building"]({bbox});\n'
        f'  way["natural"~"water|wood"]({bbox});\n'
        f'  way["waterway"]({bbox});\n'
        f'  way["highway"]({bbox});\n'
        f'  way["landuse"~"reservoir|basin|farmland|farmyard|forest|forestry"]({bbox});\n'
        f'  way["water"]({bbox});\n'
        f'  way["farmland"]({bbox});\n'
        ");\n"
        "out body;\n"
        ">;\n"
        "out skel qt;\n"
    )


def fetch_overpass(
    bounds_wgs84: dict,
    timeout_s: float = 90.0,
    urls: tuple[str, ...] = OVERPASS_URLS,
    attempts: int = 4,
    backoff_s: float = 5.0,
    sleep=time.sleep,
) -> str:
    """POST the corridor query to Overpass, rotating endpoints and retrying.

    Overpass rejects with 504 whenever its query slots are all busy, so a
    single-shot fetch fails for reasons that have nothing to do with the
    corridor. Each attempt moves to the next endpoint in `urls` and waits
    `backoff_s * attempt` seconds -- deterministic, no jitter, so tests are
    reproducible.

    A 400 aborts immediately: that is a malformed query, and no amount of
    retrying fixes it (see `overpass_query`'s note about quoted bbox filters).
    """
    if not urls:
        raise OsmFetchError("fetch_overpass: no Overpass endpoints configured")
    data = urllib.parse.urlencode({"data": overpass_query(bounds_wgs84)}).encode("utf-8")
    failures = []
    for attempt in range(attempts):
        url = urls[attempt % len(urls)]
        request = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code not in OVERPASS_RETRY_CODES:
                raise OsmFetchError(f"failed to query Overpass at {url}: {exc}") from exc
            failures.append(f"{url}: {exc}")
        except (urllib.error.URLError, OSError) as exc:
            failures.append(f"{url}: {exc}")
        if attempt + 1 < attempts:
            sleep(backoff_s * (attempt + 1))
    joined = "; ".join(failures)
    raise OsmFetchError(
        f"failed to query Overpass after {attempts} attempt(s) [{joined}]. "
        "These are load errors, not query errors -- the public instances "
        "throttle hard. Re-run the build (tiles and DEM are cached, so it "
        "resumes here), or raise semantic.overpass_attempts / "
        "semantic.overpass_backoff_s in the corridor config."
    )


def build_semantic(
    bounds_wgs84: dict,
    gsd_m: float,
    semantic_path: Path,
    crs_epsg: str,
    *,
    overpass_text: str | None = None,
    offline: bool = False,
    extract_date: str | None = None,
    road_half_width_m: float = 3.0,
    water_line_half_width_m: float = 2.0,
    overviews: tuple[int, ...] = (2, 4, 8),
    overpass_urls: tuple[str, ...] = OVERPASS_URLS,
    overpass_attempts: int = 4,
    overpass_backoff_s: float = 5.0,
    overpass_timeout_s: float = 90.0,
) -> SemanticResult:
    """Rasterize OSM semantics onto a 1-class uint8 grid and write a COG."""
    if overpass_text is None:
        if offline:
            raise OsmFetchError(
                "OSM semantics require network or a pre-fetched fragment; pass "
                "overpass_text or build with network access"
            )
        overpass_text = fetch_overpass(
            bounds_wgs84,
            timeout_s=overpass_timeout_s,
            urls=overpass_urls,
            attempts=overpass_attempts,
            backoff_s=overpass_backoff_s,
        )

    to_utm = Transformer.from_crs("EPSG:4326", crs_epsg, always_xy=True)
    west, south, east, north = (
        bounds_wgs84["west"],
        bounds_wgs84["south"],
        bounds_wgs84["east"],
        bounds_wgs84["north"],
    )
    corners = to_utm.transform([west, east, east, west], [north, north, south, south])
    grid = PixelGrid.from_bounds(
        min(corners[0]), max(corners[0]), min(corners[1]), max(corners[1]), gsd_m
    )

    array = rasterize_overpass(
        overpass_text,
        grid,
        to_utm,
        road_half_width_m=road_half_width_m,
        water_line_half_width_m=water_line_half_width_m,
    )

    semantic_path.parent.mkdir(parents=True, exist_ok=True)
    _write_semantic(semantic_path, grid, array, crs_epsg, overviews)

    return SemanticResult(
        grid=grid,
        semantic_path=semantic_path,
        gsd_m=gsd_m,
        extract_date=extract_date,
        source="osm",
    )


def _write_semantic(
    path: Path, grid: PixelGrid, array: np.ndarray, crs_epsg: str, overviews
) -> None:
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 1,
        "dtype": "uint8",
        "crs": crs_epsg,
        "transform": grid.affine(),
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "DEFLATE",
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(array, 1)
        ds.build_overviews(list(overviews), rasterio.enums.Resampling.nearest)

    from .gdalcog import to_cog

    tmp = path.with_suffix(path.suffix + ".cogtmp")
    to_cog(path, tmp, compress="DEFLATE")
    tmp.replace(path)
