"""Geoid undulation model for vertical-datum conversion (T06).

Copernicus GLO-30 ships orthometric heights above EGM2008; GNSS and lidar work
in ellipsoidal heights. Converting one into the other needs the geoid
undulation N(lat, lon):

    ellipsoidal = orthometric + N
    orthometric = ellipsoidal - N

The difference at 44.8 deg N is ~15-20 m, so skipping this is a systematic
scale error in true-ortho (see docs/plan/tasks/T06-dem-osm-geopack.md and
prompts/P7-infra.md).

The standard 2.5-arcmin EGM2008 grid is fetched from the PROJ CDN and cached on
disk, mirroring how basemap tiles are fetched. Without a grid the conversion
must NOT guess: `resolve_egm2008_grid` raises rather than fabricate an
undulation (P0 rule 5).
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import rasterio

EGM2008_GRID_FILENAME = "us_nga_egm08_25.tif"
EGM2008_GRID_URL = f"https://cdn.proj.org/{EGM2008_GRID_FILENAME}"
_DEFAULT_CACHE_DIR = Path("~/.cache/geoloc/geoid").expanduser()

USER_AGENT = "geoloc-mapprep/0.1 (internal dev)"


class GeoidUnavailableError(RuntimeError):
    pass


class GridGeoid:
    """Nearest-neighbour sampling of a geoid-undulation grid GeoTIFF.

    The grid stores N in metres with a north-up transform over WGS84 lon/lat.
    Sampling is nearest-neighbour: the 2.5-arcmin EGM2008 grid is smooth enough
    that nearest-neighbour error is far below the 0.5 m T06-U-01 budget.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._ds = rasterio.open(self.path)

    def close(self) -> None:
        self._ds.close()

    def undulation(self, lat: float, lon: float) -> float:
        return float(self.undulation_array(np.array([lat]), np.array([lon]))[0])

    def undulation_array(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        shape = lat.shape
        # rasterio.transform.rowcol always returns flat sequences, regardless
        # of the input array's shape, so the 2D grid case (e.g. a DEM raster
        # of lat/lon per pixel) must be raveled going in and reshaped coming
        # back out.
        rows, cols = rasterio.transform.rowcol(self._ds.transform, lon.ravel(), lat.ravel())
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        cols = np.clip(cols, 0, self._ds.width - 1)
        rows = np.clip(rows, 0, self._ds.height - 1)
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        window = rasterio.windows.Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
        values = self._ds.read(1, window=window)
        return values[rows - r0, cols - c0].reshape(shape)


def resolve_egm2008_grid(cache_dir: Path | None = None, offline: bool = False) -> Path:
    cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    path = cache_dir / EGM2008_GRID_FILENAME
    if path.exists() and path.stat().st_size > 0:
        return path
    if offline:
        raise GeoidUnavailableError(
            f"EGM2008 geoid grid not cached at {path}; build with network access once "
            "to fetch it (or set the geoid cache dir). Refusing to guess the undulation."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    request = urllib.request.Request(EGM2008_GRID_URL, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
    except (urllib.error.URLError, OSError) as exc:
        raise GeoidUnavailableError(
            f"failed to download EGM2008 geoid grid from {EGM2008_GRID_URL}: {exc}"
        ) from exc
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


class ConstantGeoid:
    """A constant-undulation geoid, used in tests and for diagnostic checks."""

    def __init__(self, value: float):
        self.value = float(value)

    def undulation(self, lat: float, lon: float) -> float:
        return self.value

    def undulation_array(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return np.full(np.broadcast(lat, lon).shape, self.value, dtype=np.float32)
