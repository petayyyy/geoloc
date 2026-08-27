"""Copernicus DEM layer for the geopack (T06).

GLO-30 is a 1-arcsec (~30 m) DSM delivered as 1x1 degree COG tiles in EGM2008
orthometric heights. **Its public release has holes**: the open AWS dataset
omits several countries entirely -- Armenia among them (for N39 the 30 m
bucket stops at E043, and the AMtown site needs E044). GLO-90 (3-arcsec,
~90 m) covers those tiles, so a corridor may declare an explicit fallback:

    dem:
      source: copernicus_glo30
      fallback_sources: [copernicus_glo90]

The fallback is never silent. It is opt-in per corridor, the coarser tile is
named in `DemResult.source_tiles`, the real resolution is measured off the
raster into `native_gsd`, and `manifest.layers.dem.source` lists every dataset
actually used. This module fetches the tile(s) covering the corridor,
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
GLO90_BUCKET = "https://copernicus-dem-90m.s3.amazonaws.com"
_DEFAULT_CACHE_DIR = Path("~/.cache/geoloc/dem").expanduser()


@dataclass(frozen=True)
class DemSource:
    """One Copernicus DEM product: its bucket and its tile-name infix."""

    source_id: str
    bucket: str
    infix: str  # the arcsec code in the tile name: 10 = 1", 30 = 3"
    nominal_gsd_m: float


DEM_SOURCES = {
    "copernicus_glo30": DemSource("copernicus_glo30", GLO30_BUCKET, "10", 30.0),
    "copernicus_glo90": DemSource("copernicus_glo90", GLO90_BUCKET, "30", 90.0),
}
DEFAULT_DEM_SOURCE = "copernicus_glo30"


def dem_source(source_id: str) -> DemSource:
    try:
        return DEM_SOURCES[source_id]
    except KeyError:
        raise DemFetchError(
            f"unknown DEM source {source_id!r} (known: {sorted(DEM_SOURCES)})"
        ) from None


def source_id_of_tile(tile_name: str) -> str:
    """Which dataset a cached tile came from, read back off its own name."""
    for src in DEM_SOURCES.values():
        if tile_name.startswith(f"Copernicus_DSM_COG_{src.infix}_"):
            return src.source_id
    return "unknown"

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
    source_datasets: list[str]  # which Copernicus product(s) each tile came from
    source_datum: str
    vertical_datum: str
    geoid_model: str
    valid_ratio: float


def tile_names_for(bounds_wgs84: dict, source_id: str = DEFAULT_DEM_SOURCE) -> list[str]:
    """1x1 degree COG tile names covering the corridor, for one DEM source."""
    infix = dem_source(source_id).infix
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
            names.append(f"Copernicus_DSM_COG_{infix}_{lat_suffix}_{lon_suffix}_DEM.tif")
    return names


def _download_tile(source: DemSource, name: str, dest: Path) -> bool:
    """Fetch one tile into `dest`. False if the source simply doesn't have it."""
    # Each tile lives in an S3 "directory" of the same name as the file
    # (see the AWS Open Data Registry layout for the Copernicus DEM).
    stem = name[: -len(".tif")]
    url = f"{source.bucket}/{stem}/{name}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False  # not a failure: this product has no tile here
        raise DemFetchError(f"failed to download {url}: {exc}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise DemFetchError(f"failed to download {url}: {exc}") from exc
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return True


def fetch_dem_tiles(
    bounds_wgs84: dict,
    cache_dir: Path | None = None,
    offline: bool = False,
    source_id: str = DEFAULT_DEM_SOURCE,
    fallback_source_ids: tuple[str, ...] = (),
) -> list[Path]:
    """Download (or reuse) the DEM tiles covering the corridor.

    Tries `source_id` first and each of `fallback_source_ids` in turn, per
    tile. A fallback only ever triggers on a genuine 404 -- the tile is absent
    from that product, as GLO-30 tiles are over the countries excluded from
    its public release. Network and other HTTP errors still fail the build.
    """
    cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    order = [source_id, *fallback_source_ids]
    sources = [dem_source(sid) for sid in order]
    paths = []
    for index in range(len(tile_names_for(bounds_wgs84, source_id))):
        names = [tile_names_for(bounds_wgs84, src.source_id)[index] for src in sources]
        def _is_cached(name: str) -> bool:
            path = cache_dir / name
            return path.exists() and bool(path.stat().st_size)

        cached = next((cache_dir / n for n in names if _is_cached(n)), None)
        if cached is not None:
            paths.append(cached)
            continue
        if offline:
            raise DemFetchError(
                f"DEM tile {names[0]} not cached in {cache_dir}; build with network once"
            )
        for src, name in zip(sources, names):
            if _download_tile(src, name, cache_dir / name):
                paths.append(cache_dir / name)
                break
        else:
            tried = ", ".join(f"{s.source_id}:{n}" for s, n in zip(sources, names))
            raise DemFetchError(
                f"no DEM coverage for tile {names[0]} in any configured source ({tried}). "
                "Copernicus GLO-30's public release excludes several countries; add "
                "`dem.fallback_sources: [copernicus_glo90]` to the corridor config if the "
                "corridor is in one of them."
            )
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
        source_datasets=sorted({source_id_of_tile(p.name) for p in source_paths}),
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
