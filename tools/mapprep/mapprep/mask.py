"""Validity-mask overlays: cloud polygons.

Cloud areas supplied by an operator (GeoJSON polygon rings in WGS84) are
zeroed in the mask. There is no automated cloud detector in this task; a hole
is a hole, and nothing is filled.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .georef import PixelGrid


def load_geojson_polygons(path: Path) -> list[list[tuple[float, float]]]:
    """Return the outer rings of all polygons in a GeoJSON file."""
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rings = []

    def collect(geometry: dict) -> None:
        if not isinstance(geometry, dict):
            return
        kind = geometry.get("type")
        if kind == "Polygon":
            for i, ring in enumerate(geometry.get("coordinates") or []):
                if i == 0:
                    rings.append([(pt[0], pt[1]) for pt in ring])
        elif kind == "MultiPolygon":
            for polygon in geometry.get("coordinates") or []:
                rings.append([(pt[0], pt[1]) for pt in polygon[0]])
        elif kind == "GeometryCollection":
            for child in geometry.get("geometries") or []:
                collect(child)

    features = payload.get("features")
    if isinstance(features, list):
        for feature in features:
            collect(feature.get("geometry"))
    else:
        collect(payload)
    return rings


def apply_polygon_to_mask(
    mask: np.ndarray,
    grid: PixelGrid,
    transformer,
    ring_wgs84: list[tuple[float, float]],
) -> int:
    """Zero the mask inside a WGS84 ring; return the number of pixels cleared."""
    corners_utm = []
    for lon, lat in ring_wgs84:
        east, north = transformer.transform(lon, lat)
        corners_utm.append((east, north))
    col_min, row_min, col_max, row_max = grid.window(
        min(p[0] for p in corners_utm),
        max(p[0] for p in corners_utm),
        min(p[1] for p in corners_utm),
        max(p[1] for p in corners_utm),
    )
    if col_max <= col_min or row_max <= row_min:
        return 0
    vertices = np.array([grid.utm_to_pixel(e, n) for e, n in corners_utm], dtype=np.float64)
    px = np.arange(col_min, col_max, dtype=np.float64)[None, :] + 0.5
    py = np.arange(row_min, row_max, dtype=np.float64)[:, None] + 0.5
    east_grid, north_grid = np.meshgrid(px[0], py[:, 0], indexing="xy")
    inside = np.zeros(east_grid.shape, dtype=bool)
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        if y1 == y2:
            continue
        condition = (north_grid > min(y1, y2)) & (north_grid <= max(y1, y2))
        x_intersect = (north_grid - y1) * (x2 - x1) / (y2 - y1) + x1
        inside ^= condition & (east_grid <= x_intersect)
    cleared = int(mask[row_min:row_max, col_min:col_max][inside].sum() > 0)
    mask[row_min:row_max, col_min:col_max][inside] = 0
    return cleared


def apply_polygons_to_mask(mask, grid, transformer, rings) -> int:
    cleared = 0
    for ring in rings:
        cleared += apply_polygon_to_mask(mask, grid, transformer, ring)
    return cleared
