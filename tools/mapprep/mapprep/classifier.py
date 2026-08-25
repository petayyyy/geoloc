"""Terrain classification from the OSM semantic raster (T06).

Implements docs/plan/testing/05-metrics.md section 5: every fix-level metric is
reported with a per-class breakdown, so the classifier that names the classes
must match that table exactly.

    | Класс | Правило                                     |
    |-------|---------------------------------------------|
    | urban | building > 15%                              |
    | suburban | 3% <= building <= 15%                    |
    | roads | road > 5% and building < 3%                 |
    | farmland | farmland > 60%                           |
    | forest | forest > 60%                                |
    | water | water > 50% (expected to fail, IFR must be 0)|

Fractions are taken over the whole raster (background is a legitimate class).
Precedence, when several rules hold at once, is fixed and documented below.
"""

from __future__ import annotations

import numpy as np

from .osm import (
    CLASS_BUILDING,
    CLASS_FARMLAND,
    CLASS_FOREST,
    CLASS_ROAD,
    CLASS_WATER,
)

TERRAIN_CLASSES = ("urban", "suburban", "roads", "farmland", "forest", "water", "background")

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


def classify(semantic: np.ndarray) -> str:
    """Return the terrain class name for a semantic raster.

    Precedence (first match wins) mirrors 05-metrics.md section 5:
    water, then urban, then farmland, then forest, then roads, then suburban,
    and finally ``background`` when nothing else applies.
    """
    fractions = class_fractions(semantic)
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
