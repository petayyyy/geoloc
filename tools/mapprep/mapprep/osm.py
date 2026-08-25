"""OSM semantic rasterization (T06).

Overpass JSON (or a Geofabrik-derived fragment) is parsed, each element is
mapped to one of the terrain classes, and the geometry is rasterized onto a
uint8 grid at the requested resolution in the mission CRS:

    0 background, 1 road, 2 building, 3 water, 4 farmland, 5 forest

`road` and `water` are the stable classes; `farmland` changes between seasons
(05-metrics.md section 5). Painting is deterministic: classes are applied in a
fixed priority order, later classes overwrite earlier ones.
"""

from __future__ import annotations

import json
import math

import numpy as np
import rasterio.features

from .georef import PixelGrid

CLASS_BACKGROUND = 0
CLASS_ROAD = 1
CLASS_BUILDING = 2
CLASS_WATER = 3
CLASS_FARMLAND = 4
CLASS_FOREST = 5

CLASS_NAMES = ("background", "road", "building", "water", "farmland", "forest")

# Ascending paint order: farmland/forest are the base, roads cut through them,
# buildings sit on top, water (the stable class) wins.
PAINT_ORDER = (CLASS_FARMLAND, CLASS_FOREST, CLASS_ROAD, CLASS_BUILDING, CLASS_WATER)

_WATER_LANDUSE = {"reservoir", "basin"}
_FOREST_LANDUSE = {"forest", "forestry"}
_FARMLAND_LANDUSE = {"farmland", "farmyard"}


def classify_tags(tags: dict) -> int | None:
    """Map an OSM tag dict to a terrain class id, or None if unclassified.

    Order matches the paint priority (highest first): water is the stable
    class and wins over anything else; buildings win over roads; roads win
    over farmland/forest.
    """
    if not tags:
        return None
    if (
        tags.get("natural") == "water"
        or "waterway" in tags
        or tags.get("landuse") in _WATER_LANDUSE
        or "water" in tags
    ):
        return CLASS_WATER
    if "building" in tags or "building:part" in tags:
        return CLASS_BUILDING
    if "highway" in tags:
        return CLASS_ROAD
    if tags.get("landuse") in _FARMLAND_LANDUSE or "farmland" in tags:
        return CLASS_FARMLAND
    if tags.get("landuse") in _FOREST_LANDUSE or tags.get("natural") == "wood":
        return CLASS_FOREST
    return None


def parse_overpass_json(text: str) -> tuple[dict, list[dict]]:
    """Parse an Overpass JSON response into (nodes, ways).

    Nodes are {id: (lon, lat)}; ways are the raw element dicts in document order.
    """
    payload = json.loads(text)
    nodes: dict[int, tuple[float, float]] = {}
    ways: list[dict] = []
    for element in payload.get("elements", []):
        if element.get("type") == "node":
            nodes[element["id"]] = (element["lon"], element["lat"])
        elif element.get("type") == "way":
            ways.append(element)
    return nodes, ways


def extract_geometries(nodes, ways, transformer):
    """Resolve ways to (polygons, lines) of (class_id, utm_coords)."""
    polygons: list[tuple[int, list[tuple[float, float]]]] = []
    lines: list[tuple[int, list[tuple[float, float]]]] = []
    for way in ways:
        class_id = classify_tags(way.get("tags", {}))
        if class_id is None:
            continue
        coords = []
        for node_id in way.get("nodes", []):
            if node_id in nodes:
                lon, lat = nodes[node_id]
                east, north = transformer.transform(lon, lat)
                coords.append((east, north))
        if len(coords) < 2:
            continue
        is_closed = len(coords) >= 4 and coords[0] == coords[-1]
        if class_id == CLASS_ROAD or (class_id == CLASS_WATER and not is_closed):
            lines.append((class_id, coords))
        elif is_closed:
            polygons.append((class_id, coords))
    return polygons, lines


def rasterize(
    polygons,
    lines,
    grid: PixelGrid,
    *,
    road_half_width_m: float = 3.0,
    water_line_half_width_m: float = 2.0,
) -> np.ndarray:
    """Rasterize classified OSM geometry onto a uint8 class grid.

    `polygons` are (class_id, [(east, north), ...]) closed rings in the mission
    CRS; `lines` are (class_id, [(east, north), ...]) polylines.
    """
    array = np.zeros((grid.height, grid.width), dtype=np.uint8)
    transform = grid.affine()
    for class_id in PAINT_ORDER:
        shapes = []
        for cid, coords in polygons:
            if cid == class_id:
                shapes.append((_polygon(coords), cid))
        if shapes:
            painted = rasterio.features.rasterize(
                shapes,
                out_shape=array.shape,
                transform=transform,
                fill=CLASS_BACKGROUND,
                dtype=np.uint8,
            )
            array[painted == class_id] = class_id
        for cid, coords in lines:
            if cid == class_id:
                half = road_half_width_m if cid == CLASS_ROAD else water_line_half_width_m
                _draw_thick_line(array, grid, coords, half, cid)
    return array


def rasterize_overpass(
    text: str,
    grid: PixelGrid,
    transformer,
    *,
    road_half_width_m: float = 3.0,
    water_line_half_width_m: float = 2.0,
) -> np.ndarray:
    """Parse an Overpass JSON fragment and rasterize it onto `grid`."""
    nodes, ways = parse_overpass_json(text)
    polygons, lines = extract_geometries(nodes, ways, transformer)
    return rasterize(
        polygons,
        lines,
        grid,
        road_half_width_m=road_half_width_m,
        water_line_half_width_m=water_line_half_width_m,
    )


def _polygon(coords: list[tuple[float, float]]) -> dict:
    return {"type": "Polygon", "coordinates": [[[e, n] for e, n in coords]]}


def _draw_thick_line(
    array: np.ndarray, grid: PixelGrid, coords, half_width_m: float, value: int
) -> None:
    half_px = max(1.0, half_width_m / grid.gsd)
    for (e0, n0), (e1, n1) in zip(coords, coords[1:]):
        c0, r0 = grid.utm_to_pixel(e0, n0)
        c1, r1 = grid.utm_to_pixel(e1, n1)
        cmin = int(math.floor(min(c0, c1) - half_px - 1))
        cmax = int(math.ceil(max(c0, c1) + half_px + 1))
        rmin = int(math.floor(min(r0, r1) - half_px - 1))
        rmax = int(math.ceil(max(r0, r1) + half_px + 1))
        cmin = max(0, cmin)
        rmin = max(0, rmin)
        cmax = min(grid.width, cmax)
        rmax = min(grid.height, rmax)
        if cmax <= cmin or rmax <= rmin:
            continue
        cols = np.arange(cmin, cmax, dtype=np.float64)[None, :] + 0.5
        rows = np.arange(rmin, rmax, dtype=np.float64)[:, None] + 0.5
        dx = c1 - c0
        dy = r1 - r0
        seg_len2 = dx * dx + dy * dy
        px = cols - (c0 + 0.5)
        py = rows - (r0 + 0.5)
        if seg_len2 == 0.0:
            dist2 = px * px + py * py
        else:
            t = np.clip((px * dx + py * dy) / seg_len2, 0.0, 1.0)
            proj_x = (c0 + 0.5) + t * dx
            proj_y = (r0 + 0.5) + t * dy
            dist2 = (cols - proj_x) ** 2 + (rows - proj_y) ** 2
        array[rmin:rmax, cmin:cmax][dist2 <= half_px * half_px] = value
