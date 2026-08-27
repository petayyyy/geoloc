"""Terrain-class breakdown (05-metrics.md section 5).

The classifier here is the **single** implementation of the class table -- the
same fractions and precedence that ``tools/mapprep/mapprep/classifier.py`` uses
to *name* the classes. The metrics harness re-implements the decision on top of
already-assigned class labels so it has no rasterio/georeferencing dependency:
runners stamp a ``terrain`` class onto every fix (from the OSM semantic raster),
and this module only splits metrics by that label.

Class table (first match wins):

    water     water > 50%
    urban     building > 15%
    farmland  farmland > 60%
    forest    forest > 60%
    roads     road > 5% and building < 3%
    suburban  3% <= building <= 15%
    background otherwise
"""

from __future__ import annotations

import numpy as np

TERRAIN_CLASSES = ("urban", "suburban", "roads", "farmland", "forest", "water", "background")

# Semantic raster class codes (matches tools/mapprep/mapprep/osm.py).
CLASS_BACKGROUND = 0
CLASS_ROAD = 1
CLASS_BUILDING = 2
CLASS_WATER = 3
CLASS_FARMLAND = 4
CLASS_FOREST = 5

URBAN_BUILDING_MIN = 0.15
SUBURBAN_BUILDING_MIN = 0.03
ROADS_ROAD_MIN = 0.05
ROADS_BUILDING_MAX = 0.03
FARMLAND_MIN = 0.60
FOREST_MIN = 0.60
WATER_MIN = 0.50


def class_fractions(semantic: np.ndarray) -> dict[int, float]:
    """Fraction of raster pixels in each class (over all pixels)."""
    array = np.asarray(semantic)
    total = array.size
    if total == 0:
        return {}
    counts = np.bincount(array.ravel(), minlength=6)
    return {i: float(counts[i]) / total for i in range(6)}


def classify_fractions(fractions: dict[int, float]) -> str:
    """Return the terrain class name for a fractions dict, first-match-wins."""
    building = fractions.get(CLASS_BUILDING, 0.0)
    road = fractions.get(CLASS_ROAD, 0.0)
    water = fractions.get(CLASS_WATER, 0.0)
    farmland = fractions.get(CLASS_FARMLAND, 0.0)
    forest = fractions.get(CLASS_FOREST, 0.0)

    if water > WATER_MIN:
        return "water"
    if building > URBAN_BUILDING_MIN:
        return "urban"
    if farmland > FARMLAND_MIN:
        return "farmland"
    if forest > FOREST_MIN:
        return "forest"
    if road > ROADS_ROAD_MIN and building < ROADS_BUILDING_MAX:
        return "roads"
    if building >= SUBURBAN_BUILDING_MIN:
        return "suburban"
    return "background"


def classify(semantic: np.ndarray) -> str:
    return classify_fractions(class_fractions(semantic))


def by_terrain(
    rec: np.ndarray, fix_metrics_fn, bias: tuple[float, float] = (0.0, 0.0), bias_mode: str = "with"
) -> dict[str, dict]:
    """Apply a fix-level metric table per terrain class.

    ``fix_metrics_fn`` is ``metrics.fix_level_table``; every class present in the
    data gets its own table, so an urban average never hides a forest failure.
    """
    out: dict[str, dict] = {}
    for cls in TERRAIN_CLASSES:
        mask = rec["terrain"] == cls
        if not np.any(mask):
            continue
        out[cls] = fix_metrics_fn(rec[mask], bias=bias, bias_mode=bias_mode)
    return out


def terrain_counts(rec: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for cls in TERRAIN_CLASSES:
        n = int(np.sum(rec["terrain"] == cls))
        if n:
            out[cls] = n
    return out


def assign_terrain_class(
    east: np.ndarray,
    north: np.ndarray,
    semantic_raster: np.ndarray,
    transform,
    window_radius_m: float = 130.0,
) -> np.ndarray:
    """Stamp a terrain class onto each (east, north) point by sampling the OSM
    semantic raster over a patch-sized window centred on the point.

    ``transform`` is a GDAL-style affine tuple ``(a, b, c, d, e, f)`` with a
    north-up grid (``a = gsd``, ``e = -gsd``, ``c/f`` the top-left origin).
    ``semantic_raster`` is the uint8 class raster (1 px = 1 m in the geopack).
    The window radius defaults to the ortho patch radius, so a fix's class is
    "the terrain it actually looked at".
    """
    e = np.asarray(east, dtype=np.float64)
    n = np.asarray(north, dtype=np.float64)
    out = np.full(e.shape, "background", dtype="U16")
    a, _b, c, _d, ee, f = transform
    gsd = abs(a)
    h, w = semantic_raster.shape
    for i in range(len(e)):
        col = (e[i] - c) / a
        row = (n[i] - f) / ee
        half_px = window_radius_m / gsd
        c0, c1 = int(np.floor(col - half_px)), int(np.ceil(col + half_px))
        r0, r1 = int(np.floor(row - half_px)), int(np.ceil(row + half_px))
        c0, c1 = max(0, c0), min(w, c1)
        r0, r1 = max(0, r0), min(h, r1)
        if c1 <= c0 or r1 <= r0:
            continue
        out[i] = classify(semantic_raster[r0:r1, c0:c1])
    return out
