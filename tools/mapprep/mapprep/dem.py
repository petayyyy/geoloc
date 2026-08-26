"""Copernicus GLO-30 DEM layer for the geopack (T06).

GLO-30 is a 1-arcsec (~30 m) DSM delivered as 1x1 degree COG tiles in EGM2008
orthometric heights. This module fetches the tile(s) covering the corridor,
reprojects them to the mission CRS, resamples onto the requested grid, converts
the vertical datum EGM2008 -> ellipsoid with the geoid model, and writes the
result as a COG.

Honesty rules carried over from T05:
- the requested grid is documented as-is (`gsd`), the source resolution is
  recorded separately as `native_gsd`, and the manifest note states that the
  source was interpolated when the target grid is finer than native;
- GLO-30 is a DSM (buildings and vegetation included), not a DTM; that is
  recorded in the manifest so T29 does not double-count OSM extrusion;
- a missing geoid model fails the build rather than guessing the undulation.
"""

from __future__ import annotations

import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp
from pyproj import Transformer

from .georef import PixelGrid

GLO30_NODATA = -9999.0
GLO30_BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
_DEFAULT_CACHE_DIR = Path("~/.cache/geoloc/dem").expanduser()

USER_AGENT = "geoloc-mapprep/0.1 (internal dev)"


class DemFetchError(RuntimeError):
    pass


@dataclass
class DemResult:
    grid: PixelGrid
    dem_path: Path
    target_gsd_m: float
    native_gsd_m: float
    source_tiles: list[str]
    source_datum: str
    vertical_datum: str
    geoid_model: str
    valid_ratio: float


def tile_names_for(bounds_wgs84: dict) -> list[str]:
    """GLO-30 1x1 degree COG tile names covering the corridor."""
    west, south, east, north = (
        bounds_wgs84["west"],
        bounds_wgs84["south"],
        bounds_wgs84["east"],
        bounds_wgs84["north"],
    )
    names = []
    for lat in range(int(math.floor(south)), int(math.ceil(north))):
        for lon in range(int(math.floor(west)), int(math.ceil(east))):
            lat_suffix = f"{'S' if lat < 0 else 'N'}{abs(lat):02d}_00"
            lon_suffix = f"{'W' if lon < 0 else 'E'}{abs(lon):03d}_00"
            names.append(f"Copernicus_DSM_COG_10_{lat_suffix}_{lon_suffix}_DEM.tif")
    return names


def fetch_dem_tiles(
    bounds_wgs84: dict, cache_dir: Path | None = None, offline: bool = False
) -> list[Path]:
    cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in tile_names_for(bounds_wgs84):
        path = cache_dir / name
        if path.exists() and path.stat().st_size > 0:
            paths.append(path)
            continue
        if offline:
            raise DemFetchError(f"DEM tile {name} not cached at {path}; build with network once")
        # Each tile lives in an S3 "directory" of the same name as the file
        # (see the AWS Open Data Registry layout for Copernicus DEM GLO-30).
        stem = name[: -len(".tif")]
        url = f"{GLO30_BUCKET}/{stem}/{name}"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
        except (urllib.error.URLError, OSError) as exc:
            raise DemFetchError(f"failed to download {url}: {exc}") from exc
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        paths.append(path)
    return paths


def _to_wgs84_transformer(crs_epsg: str) -> Transformer:
    return Transformer.from_crs(crs_epsg, "EPSG:4326", always_xy=True)


def build_dem(
    bounds_wgs84: dict,
    target_gsd_m: float,
    source_paths: list[Path],
    dem_path: Path,
    crs_epsg: str,
    *,
    geoid,
    overviews: tuple[int, ...] = (2, 4, 8),
    geoid_model: str = "egm2008",
    source_datum: str = "EGM2008",
) -> DemResult:
    """Assemble the DEM layer: reproject, resample, datum-convert, write COG."""
    to_utm = Transformer.from_crs("EPSG:4326", crs_epsg, always_xy=True)
    west, south, east, north = (
        bounds_wgs84["west"],
        bounds_wgs84["south"],
        bounds_wgs84["east"],
        bounds_wgs84["north"],
    )
    corners = to_utm.transform([west, east, east, west], [north, north, south, south])
    grid = PixelGrid.from_bounds(
        min(corners[0]), max(corners[0]), min(corners[1]), max(corners[1]), target_gsd_m
    )

    accumulator = np.zeros((grid.height, grid.width), dtype=np.float64)
    counts = np.zeros((grid.height, grid.width), dtype=np.uint32)
    native_gsd_m = 0.0
    for source_path in source_paths:
        with rasterio.open(source_path) as src:
            native_gsd_m = max(native_gsd_m, float(src.transform.a) * 111320.0)
            dst = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_crs=src.crs,
                src_transform=src.transform,
                src_nodata=src.nodata,
                dst_transform=grid.affine(),
                dst_crs=crs_epsg,
                dst_nodata=np.nan,
                resampling=rasterio.enums.Resampling.bilinear,
                num_threads=1,
            )
        valid = ~np.isnan(dst)
        accumulator[valid] += dst[valid]
        counts[valid] += 1

    orthometric = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    covered = counts > 0
    orthometric[covered] = accumulator[covered] / counts[covered]

    to_wgs84 = _to_wgs84_transformer(crs_epsg)
    cols = np.arange(grid.width, dtype=np.float64)[None, :]
    rows = np.arange(grid.height, dtype=np.float64)[:, None]
    east_grid = grid.origin_east + (cols + 0.5) * grid.gsd
    north_grid = grid.origin_north - (rows + 0.5) * grid.gsd
    lon, lat = to_wgs84.transform(
        np.broadcast_to(east_grid, orthometric.shape),
        np.broadcast_to(north_grid, orthometric.shape),
    )
    undulation = geoid.undulation_array(np.asarray(lat), np.asarray(lon))

    ellipsoidal = np.full_like(orthometric, GLO30_NODATA, dtype=np.float32)
    ellipsoidal[covered] = orthometric[covered] + undulation[covered]

    dem_path.parent.mkdir(parents=True, exist_ok=True)
    _write_dem(dem_path, grid, ellipsoidal, crs_epsg, overviews)

    return DemResult(
        grid=grid,
        dem_path=dem_path,
        target_gsd_m=target_gsd_m,
        native_gsd_m=round(native_gsd_m, 3) if native_gsd_m else 0.0,
        source_tiles=[p.name for p in source_paths],
        source_datum=source_datum,
        vertical_datum="ellipsoid",
        geoid_model=geoid_model,
        valid_ratio=float(covered.sum()) / float(covered.size),
    )


def _write_dem(path: Path, grid: PixelGrid, array: np.ndarray, crs_epsg: str, overviews) -> None:
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 1,
        "dtype": "float32",
        "crs": crs_epsg,
        "transform": grid.affine(),
        "nodata": GLO30_NODATA,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "DEFLATE",
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(array, 1)
        ds.build_overviews(list(overviews), rasterio.enums.Resampling.average)

    from .gdalcog import to_cog

    tmp = path.with_suffix(path.suffix + ".cogtmp")
    to_cog(path, tmp, compress="DEFLATE")
    tmp.replace(path)
