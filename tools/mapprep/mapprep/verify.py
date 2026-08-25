"""Georeferencing self-checks for a built geopack.

Real-world ground control points arrive with T09 (RTK/GPS track). Until then
the honest checks available are: (1) the stored geotransform round-trips
exactly against the manifest grid, (2) tile-corner GCPs derived from the
slippy-map math land inside their own quads within tolerance, (3) the overview
pyramid matches the base downsampling, and (4) an optional cross-provider NCC
shift estimate that quantifies how far the two providers disagree.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

from . import webmercator
from .georef import PixelGrid, round_trip_error_px


@dataclass
class GcpResult:
    name: str
    east: float
    north: float
    col: float
    row: float
    error_px: float


def verify_geotransform(manifest: dict, geopack_dir: Path) -> list[str]:
    problems = []
    bounds = manifest["bounds"]
    for layer in manifest["layers"].values():
        if not layer.get("file", "").endswith(".tif"):
            continue
        path = geopack_dir / layer["file"]
        with rasterio.open(path) as ds:
            transform = ds.transform
            if ds.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") != "COG":
                problems.append(f"{layer['file']}: not a COG (missing LAYOUT=COG)")
        grid = _grid_from_bounds(bounds, layer["gsd"])
        for name, expected, actual in [
            ("origin_east", grid.origin_east, transform.c),
            ("origin_north", grid.origin_north, transform.f),
            ("gsd", grid.gsd, transform.a),
            ("gsd_row", grid.gsd, -transform.e),
        ]:
            if abs(expected - actual) > 1e-6 * max(1.0, abs(expected)):
                problems.append(
                    f"{layer['file']}: {name} differs: manifest {expected}, file {actual}"
                )
    return problems


def _grid_from_bounds(bounds: dict, gsd: float) -> PixelGrid:
    return PixelGrid.from_bounds(
        bounds["east_min"], bounds["east_max"], bounds["north_min"], bounds["north_max"], gsd
    )


def sample_tile_corner_gcps(
    manifest: dict, geopack_dir: Path, max_gcps: int = 40
) -> list[GcpResult]:
    gcps: list[GcpResult] = []
    transformer = Transformer.from_crs("EPSG:4326", manifest["crs"], always_xy=True)
    for layer_name, layer in manifest["layers"].items():
        if not layer_name.startswith("ortho_"):
            continue
        bounds_wgs84 = layer.get("bounds_wgs84")
        zoom = layer.get("source_zoom")
        if not bounds_wgs84 or zoom is None:
            continue
        tiles = webmercator.tiles_for_bounds(
            bounds_wgs84["west"],
            bounds_wgs84["south"],
            bounds_wgs84["east"],
            bounds_wgs84["north"],
            zoom,
        )
        step = max(1, len(tiles) // max_gcps)
        with rasterio.open(geopack_dir / layer["file"]) as ds:
            transform = ds.transform
            grid = PixelGrid(transform.c, transform.f, transform.a, ds.width, ds.height)
            for x, y in tiles[::step][:max_gcps]:
                lon, lat = webmercator.tile_corners_lonlat(x, y, zoom)[0]
                east, north = transformer.transform(lon, lat)
                col, row = grid.utm_to_pixel(east, north)
                error_px = round_trip_error_px(grid, col, row)
                gcps.append(
                    GcpResult(
                        name=f"{layer_name}_tile_{x}_{y}",
                        east=east,
                        north=north,
                        col=float(col),
                        row=float(row),
                        error_px=float(error_px),
                    )
                )
    return gcps


def write_gcps(gcps: list[GcpResult], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "east_m", "north_m", "col_px", "row_px", "error_px"])
        for gcp in gcps:
            writer.writerow(
                [
                    gcp.name,
                    f"{gcp.east:.6f}",
                    f"{gcp.north:.6f}",
                    f"{gcp.col:.6f}",
                    f"{gcp.row:.6f}",
                    f"{gcp.error_px:.3e}",
                ]
            )


def check_pyramid(path: Path) -> dict:
    with rasterio.open(path) as ds:
        if not ds.overviews(1):
            return {"overviews": [], "problems": ["no overviews found"]}
        overviews = ds.overviews(1)
        problems = []
        for level in overviews:
            ovr = ds.read(1, out_shape=(ds.height // level, ds.width // level))
            decimated = ds.read(
                1,
                out_shape=(ds.height // level, ds.width // level),
                resampling=rasterio.enums.Resampling.average,
            )
            diff = np.abs(ovr.astype(np.int32) - decimated.astype(np.int32))
            if int(diff.max()) != 0:
                problems.append(
                    f"overview {level}x deviates from base downsampling (max {diff.max()})"
                )
        return {"overviews": overviews, "problems": problems}


def cross_provider_offset(manifest: dict, geopack_dir: Path, max_shift_px: int = 12) -> dict:
    """Estimate the cross-provider georeferencing disagreement.

    Both layers are sampled on the same metric footprint (not the same pixel
    count), normalized cross-correlation gives the shift, and the median over
    several windows is reported. A low peak NCC means the two providers are
    too dissimilar (season, capture date) for the estimate to be meaningful.
    """
    ortho = {name: layer for name, layer in manifest["layers"].items() if name.startswith("ortho_")}
    if len(ortho) < 2:
        return {"skipped": True, "reason": "fewer than two ortho layers"}
    (name_a, layer_a), (name_b, layer_b) = list(ortho.items())[:2]
    gsd = min(layer_a["gsd"], layer_b["gsd"])
    footprint_m = 128.0
    bounds = manifest["bounds"]
    center = {
        "east": (bounds["east_min"] + bounds["east_max"]) / 2,
        "north": (bounds["north_min"] + bounds["north_max"]) / 2,
    }
    span_e = bounds["east_max"] - bounds["east_min"]
    span_n = bounds["north_max"] - bounds["north_min"]
    offsets = []
    for kx, ky in [(0, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
        east = center["east"] + kx * span_e * 0.2
        north = center["north"] + ky * span_n * 0.2
        samples = []
        for layer in (layer_a, layer_b):
            path = geopack_dir / layer["file"]
            size = int(round(footprint_m / layer["gsd"]))
            with rasterio.open(path) as ds:
                col = int((east - ds.transform.c) / ds.transform.a)
                row = int((ds.transform.f - north) / ds.transform.a)
                win = Window(col - size // 2, row - size // 2, size, size)
                win = win.intersection(Window(0, 0, ds.width, ds.height))
                if win.width < size // 2 or win.height < size // 2:
                    break
                array = ds.read(window=win, out_shape=(ds.count, win.height, win.width))
            gray = array.astype(np.float32).mean(axis=0)
            gray -= gray.mean()
            gray /= max(float(gray.std()), 1e-6)
            samples.append(gray)
        if len(samples) < 2:
            continue
        a, b = samples
        side = min(a.shape[0], b.shape[0], a.shape[1], b.shape[1])
        a, b = a[:side, :side], b[:side, :side]
        best_score = -1.0
        best_shift = (0, 0)
        for dy in range(-max_shift_px, max_shift_px + 1):
            for dx in range(-max_shift_px, max_shift_px + 1):
                a_shifted = a[
                    max(0, dy) : side + min(0, dy),
                    max(0, dx) : side + min(0, dx),
                ]
                b_shifted = b[
                    max(0, -dy) : side + min(0, -dy),
                    max(0, -dx) : side + min(0, -dx),
                ]
                score = float((a_shifted * b_shifted).mean())
                if score > best_score:
                    best_score = score
                    best_shift = (dx, dy)
        offsets.append((best_score, best_shift))
    if not offsets:
        return {"skipped": True, "reason": "no overlapping windows"}
    scores = [score for score, _ in offsets]
    dx = int(np.median([shift[0] for _, shift in offsets]))
    dy = int(np.median([shift[1] for _, shift in offsets]))
    peak = float(np.max(scores))
    return {
        "skipped": False,
        "ncc_score": peak,
        "shift_px": [dx, dy],
        "offset_m": math.hypot(dx, dy) * gsd,
        "layers": [name_a, name_b],
        "reliable": peak >= 0.15,
        "windows": len(offsets),
    }
