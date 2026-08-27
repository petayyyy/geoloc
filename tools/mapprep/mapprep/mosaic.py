"""Mosaic assembly: XYZ tiles -> UTM COG with an overview pyramid.

Deterministic by construction: tiles are processed in sorted order, warping is
single-threaded, and every tile writes exactly the pixels inside its own
projected quad, so no write order can change the result.

Upsampling is forbidden: a source whose native GSD is coarser than the target
builds at its native GSD instead, and the manifest records the honest value.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp
from pyproj import Transformer

from . import webmercator
from .coverage import (
    DEFAULT_PLACEHOLDER_MIN_REPEATS,
    DEFAULT_PLACEHOLDER_STD,
    find_placeholder_tiles,
)
from .fetch import fetch_tile, find_cached_tile, load_meta
from .georef import PixelGrid
from .providers import Provider

# Source tiles are plain JPEGs without georeferencing by design.
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


@dataclass
class BuildResult:
    layer_name: str
    provider_id: str
    zoom: int
    native_gsd_m: float
    mosaic_gsd_m: float
    grid: PixelGrid
    ortho_path: Path
    mask_path: Path
    tiles_expected: int
    tiles_fetched: int
    placeholder_tiles: int = 0
    missing_tiles: list[tuple[int, int]] = field(default_factory=list)
    seam_count: int = 0
    upsampling_refused: bool = False


def _transformer_4326(crs_epsg: str) -> Transformer:
    return Transformer.from_crs("EPSG:4326", crs_epsg, always_xy=True)


def _transformer_3857(crs_epsg: str) -> Transformer:
    return Transformer.from_crs("EPSG:3857", crs_epsg, always_xy=True)


def _project(
    transformer: Transformer, points: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    xs, ys = zip(*points)
    exs, nys = transformer.transform(xs, ys)
    return list(zip(exs, nys))


def _point_in_quad(
    corners_px: list[tuple[float, float]], px: np.ndarray, py: np.ndarray
) -> np.ndarray:
    area = 0.0
    for i in range(4):
        x1, y1 = corners_px[i]
        x2, y2 = corners_px[(i + 1) % 4]
        area += x1 * y2 - x2 * y1
    sign = 1.0 if area >= 0 else -1.0
    inside = np.ones(px.shape, dtype=bool)
    for i in range(4):
        ax, ay = corners_px[i]
        bx, by = corners_px[(i + 1) % 4]
        cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
        inside &= cross * sign <= 0.0
    return inside


def select_zoom(
    provider: Provider,
    bounds_wgs84: dict,
    cache_root: Path,
    min_zoom: int,
    max_zoom: int,
    offline: bool,
    placeholder_std: float = DEFAULT_PLACEHOLDER_STD,
) -> int:
    """Pick the finest zoom that actually has imagery (probed at corners + centre)."""
    west, south, east, north = (
        bounds_wgs84["west"],
        bounds_wgs84["south"],
        bounds_wgs84["east"],
        bounds_wgs84["north"],
    )
    for z in range(max_zoom, min_zoom - 1, -1):
        tiles = webmercator.tiles_for_bounds(west, south, east, north, z)
        if not tiles:
            continue
        probe = [
            webmercator.lonlat_to_tile(lon, lat, z)
            for lon, lat in [
                (west, south),
                (east, south),
                (west, north),
                (east, north),
                ((west + east) / 2, (south + north) / 2),
            ]
        ]
        if not all(fetch_tile(provider, x, y, z, cache_root, offline=offline) for x, y in probe):
            continue
        # HTTP 200 is not coverage: a provider without local imagery at this
        # zoom serves a placeholder card. Step down rather than build a
        # basemap out of them (found 2026-08-27: Esri has this village at
        # z17/z18 but answers z19 with "Map data not yet available").
        probe_paths = [
            path
            for path in (find_cached_tile(provider, cache_root, x, y, z) for x, y in probe)
            if path is not None
        ]
        blank = len(
            find_placeholder_tiles(probe_paths, placeholder_std, min_repeats=len(probe) // 2 + 1)
        )
        if blank > len(probe) // 2:
            print(
                f"  zoom {z}: {blank}/{len(probe)} probe tiles are placeholders, stepping down",
                flush=True,
            )
            continue
        return z
    raise RuntimeError(f"no zoom level with real imagery between {min_zoom} and {max_zoom}")


def build_layer(
    provider: Provider,
    bounds_wgs84: dict,
    target_gsd_m: float,
    zoom: int,
    cache_root: Path,
    ortho_path: Path,
    mask_path: Path,
    crs_epsg: str,
    *,
    offline: bool = False,
    overviews: tuple[int, ...] = (2, 4, 8),
    enrich_dates: bool = False,
    cloud_polygons: list[list[tuple[float, float]]] | None = None,
    placeholder_std: float = DEFAULT_PLACEHOLDER_STD,
    placeholder_min_repeats: int = DEFAULT_PLACEHOLDER_MIN_REPEATS,
    placeholder_max_frac: float = 0.2,
) -> BuildResult:
    west, south, east, north = (
        bounds_wgs84["west"],
        bounds_wgs84["south"],
        bounds_wgs84["east"],
        bounds_wgs84["north"],
    )
    center_lat = (south + north) / 2.0
    native_gsd = webmercator.gsd_m(center_lat, zoom)
    upsampling_refused = target_gsd_m < native_gsd * 0.999
    mosaic_gsd = native_gsd if upsampling_refused else target_gsd_m
    if upsampling_refused:
        warnings.warn(
            f"target gsd {target_gsd_m} would upsample native {native_gsd:.4f} m/px; "
            f"building at native gsd instead (upsampling forbidden)",
            stacklevel=2,
        )

    transformer = _transformer_4326(crs_epsg)
    corners_utm = _project(
        transformer, [(west, north), (east, north), (east, south), (west, south)]
    )
    grid = PixelGrid.from_bounds(
        min(p[0] for p in corners_utm),
        max(p[0] for p in corners_utm),
        min(p[1] for p in corners_utm),
        max(p[1] for p in corners_utm),
        mosaic_gsd,
    )

    tiles = sorted(webmercator.tiles_for_bounds(west, south, east, north, zoom))
    missing: list[tuple[int, int]] = []
    for index, (x, y) in enumerate(tiles):
        if not fetch_tile(provider, x, y, zoom, cache_root, offline=offline):
            missing.append((x, y))
        if (index + 1) % 50 == 0 or index + 1 == len(tiles):
            print(
                f"  tiles: {index + 1}/{len(tiles)} (cache hits, missing so far: {len(missing)})",
                flush=True,
            )

    # Served-but-empty tiles are found layer-wide, because "the same card
    # repeated" is only visible across the whole set.
    cached = {
        (x, y): find_cached_tile(provider, cache_root, x, y, zoom)
        for x, y in tiles
        if (x, y) not in set(missing)
    }
    blank_paths = find_placeholder_tiles(
        [path for path in cached.values() if path is not None],
        placeholder_std,
        placeholder_min_repeats,
    )
    placeholders = [xy for xy, path in cached.items() if path is not None and path in blank_paths]
    if placeholders:
        print(
            f"  {len(placeholders)}/{len(tiles)} tiles carry no imagery "
            f"(provider placeholder); excluded and marked invalid",
            flush=True,
        )
    # Out of the mosaic and invalid in the validity mask, exactly like a
    # tile the server never had.
    missing = sorted(set(missing) | set(placeholders))
    fetched = len(tiles) - len(missing)
    missing_set = set(missing)
    if tiles and len(placeholders) > placeholder_max_frac * len(tiles):
        raise RuntimeError(
            f"{ortho_path.stem}: {len(placeholders)}/{len(tiles)} tiles from "
            f"{provider.id} at zoom {zoom} are placeholders ('no imagery here'), over the "
            f"{placeholder_max_frac:.0%} limit. The provider has no coverage at this zoom "
            "for this corridor -- lower the layer's max_zoom, or use another provider. "
            "Refusing to build a basemap that would report validity it does not have."
        )
    meta = load_meta(provider, cache_root) if enrich_dates else {}

    ortho_path.parent.mkdir(parents=True, exist_ok=True)
    _create_tiled_dataset(ortho_path, grid, 3, "JPEG", crs_epsg)

    mask = np.zeros((grid.height, grid.width), dtype=np.uint8)
    for x, y in tiles:
        if (x, y) in missing_set:
            continue
        _place_tile(ortho_path, grid, transformer, provider, cache_root, x, y, zoom, crs_epsg)
        _paint_quad(
            mask,
            grid,
            _project(transformer, webmercator.tile_corners_lonlat(x, y, zoom)),
        )

    seam_count = _mark_seams(
        mask,
        grid,
        _transformer_3857(crs_epsg),
        tiles,
        meta,
        zoom,
        enrich_dates,
    )
    if cloud_polygons:
        from .mask import apply_polygons_to_mask

        apply_polygons_to_mask(mask, grid, transformer, cloud_polygons)

    _build_overviews(ortho_path, overviews, rasterio.enums.Resampling.average)
    _write_mask(mask_path, mask, grid, overviews, crs_epsg)
    _to_cog(ortho_path, "JPEG")
    _to_cog(mask_path, "DEFLATE")

    return BuildResult(
        layer_name=ortho_path.stem,
        provider_id=provider.id,
        zoom=zoom,
        native_gsd_m=native_gsd,
        mosaic_gsd_m=mosaic_gsd,
        grid=grid,
        ortho_path=ortho_path,
        mask_path=mask_path,
        tiles_expected=len(tiles),
        tiles_fetched=fetched,
        placeholder_tiles=len(placeholders),
        missing_tiles=missing,
        seam_count=seam_count,
        upsampling_refused=upsampling_refused,
    )


def _create_tiled_dataset(
    path: Path, grid: PixelGrid, count: int, compress: str, crs_epsg: str
) -> None:
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": count,
        "dtype": "uint8",
        "crs": crs_epsg,
        "transform": grid.affine(),
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": compress,
    }
    if compress.upper() == "JPEG":
        profile["JPEG_QUALITY"] = 85
        profile["photometric"] = "YCBCR" if count == 3 else "MINISBLACK"
    with rasterio.open(path, "w", **profile):
        pass


def _place_tile(
    ortho_path: Path,
    grid: PixelGrid,
    transformer: Transformer,
    provider: Provider,
    cache_root: Path,
    x: int,
    y: int,
    z: int,
    crs_epsg: str,
) -> None:
    src_path = find_cached_tile(provider, cache_root, x, y, z)
    if src_path is None:
        return
    with rasterio.open(src_path) as src:
        array = src.read()
    west, south, east, north = webmercator.tile_bounds_3857(x, y, z)
    src_transform = rasterio.transform.from_bounds(west, south, east, north, 256, 256)

    corners_utm = _project(transformer, webmercator.tile_corners_lonlat(x, y, z))
    col_min, row_min, col_max, row_max = grid.window(
        min(p[0] for p in corners_utm),
        max(p[0] for p in corners_utm),
        min(p[1] for p in corners_utm),
        max(p[1] for p in corners_utm),
    )
    if col_max <= col_min or row_max <= row_min:
        return
    height, width = row_max - row_min, col_max - col_min
    dst = np.zeros((3, height, width), dtype=np.uint8)
    rasterio.warp.reproject(
        source=array,
        destination=dst,
        src_transform=src_transform,
        src_crs="EPSG:3857",
        dst_transform=grid.affine(col_min, row_min),
        dst_crs=crs_epsg,
        resampling=rasterio.enums.Resampling.bilinear,
        num_threads=1,
    )

    quad_px = [grid.utm_to_pixel(e, n) for e, n in corners_utm]
    px = np.arange(col_min, col_max, dtype=np.float64)[None, :] + 0.5
    py = np.arange(row_min, row_max, dtype=np.float64)[:, None] + 0.5
    centers = np.stack(np.meshgrid(px[0], py[:, 0], indexing="xy"))
    inside = _point_in_quad(quad_px, centers[0], centers[1])
    dst[:, ~inside] = 0
    with rasterio.open(ortho_path, "r+") as dataset:
        dataset.write(dst, window=rasterio.windows.Window(col_min, row_min, width, height))


def _paint_quad(mask: np.ndarray, grid: PixelGrid, corners_utm) -> None:
    col_min, row_min, col_max, row_max = grid.window(
        min(p[0] for p in corners_utm),
        max(p[0] for p in corners_utm),
        min(p[1] for p in corners_utm),
        max(p[1] for p in corners_utm),
    )
    if col_max <= col_min or row_max <= row_min:
        return
    quad_px = [grid.utm_to_pixel(e, n) for e, n in corners_utm]
    px = np.arange(col_min, col_max, dtype=np.float64)[None, :] + 0.5
    py = np.arange(row_min, row_max, dtype=np.float64)[:, None] + 0.5
    centers = np.stack(np.meshgrid(px[0], py[:, 0], indexing="xy"))
    inside = _point_in_quad(quad_px, centers[0], centers[1])
    mask[row_min:row_max, col_min:col_max][inside] = 255


def _mark_seams(
    mask: np.ndarray,
    grid: PixelGrid,
    transformer: Transformer,
    tiles: list[tuple[int, int]],
    meta: dict,
    zoom: int,
    enrich_dates: bool,
) -> int:
    if not enrich_dates:
        return 0
    seam_count = 0
    for tile_a, tile_b in _adjacent_pairs(tiles):
        date_a = meta.get(f"{zoom}/{tile_a[0]}/{tile_a[1]}", {}).get("capture_date")
        date_b = meta.get(f"{zoom}/{tile_b[0]}/{tile_b[1]}", {}).get("capture_date")
        if not date_a or not date_b or date_a == date_b:
            continue
        seam_count += _draw_edge(mask, grid, transformer, _shared_edge(tile_a, tile_b, zoom))
    return seam_count


def _adjacent_pairs(tiles: list[tuple[int, int]]):
    by_col: dict[int, list[int]] = {}
    for x, y in tiles:
        by_col.setdefault(x, []).append(y)
    for x, ys in by_col.items():
        for y1, y2 in zip(sorted(ys), sorted(ys)[1:]):
            if y2 == y1 + 1:
                yield (x, y1), (x, y2)
    for x, ys in by_col.items():
        for y in ys:
            if y in by_col.get(x + 1, []):
                yield (x, y), (x + 1, y)


def _shared_edge(tile_a: tuple[int, int], tile_b: tuple[int, int], z: int):
    if tile_a[0] == tile_b[0]:
        lower = min(tile_a[1], tile_b[1]) + 1
        west, _, east, north = webmercator.tile_bounds_3857(tile_a[0], lower, z)
        return [(west, north), (east, north)]
    left = min(tile_a[0], tile_b[0])
    west, south, east, north = webmercator.tile_bounds_3857(left, tile_a[1], z)
    del west, north
    return [
        (east, south),
        (east, south + (webmercator.tile_bounds_3857(left, tile_a[1], z)[3] - south)),
    ]


def _draw_edge(mask: np.ndarray, grid: PixelGrid, transformer, edge) -> int:
    (e1, n1), (e2, n2) = _project(transformer, edge)
    steps = max(2, int(math.hypot(e2 - e1, n2 - n1) / (grid.gsd * 0.5)))
    marked = 0
    for i in range(steps + 1):
        t = i / steps
        col, row = grid.utm_to_pixel(e1 + (e2 - e1) * t, n1 + (n2 - n1) * t)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                ri, ci = int(round(row)) + dr, int(round(col)) + dc
                if 0 <= ri < grid.height and 0 <= ci < grid.width and mask[ri, ci] == 255:
                    mask[ri, ci] = 0
                    marked += 1
    return marked


def _build_overviews(path: Path, overviews: tuple[int, ...], resampling) -> None:
    with rasterio.open(path, "r+") as ds:
        ds.build_overviews(list(overviews), resampling)


def _write_mask(
    mask_path: Path, mask: np.ndarray, grid: PixelGrid, overviews, crs_epsg: str
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
    with rasterio.open(mask_path, "w", **profile) as ds:
        ds.write(mask, 1)
    _build_overviews(mask_path, overviews, rasterio.enums.Resampling.nearest)


def _to_cog(path: Path, compress: str) -> None:
    from .gdalcog import to_cog

    tmp = path.with_suffix(path.suffix + ".cogtmp")
    to_cog(path, tmp, compress=compress)
    tmp.replace(path)
