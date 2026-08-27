"""Terrain classifier parity with the mapprep class table (05-metrics section 5)."""

import numpy as np

from geoloc_metrics.terrain import (
    CLASS_BUILDING,
    CLASS_FARMLAND,
    CLASS_FOREST,
    CLASS_ROAD,
    CLASS_WATER,
    classify,
    classify_fractions,
)


def _semantic(fractions):
    # Build a 100x100 raster from a dict of class->fraction (remainder background).
    arr = np.zeros((100, 100), dtype=np.uint8)
    n = arr.size
    pos = 0
    for cls, frac in sorted(fractions.items()):
        cnt = int(round(frac * n))
        arr.ravel()[pos : pos + cnt] = cls
        pos += cnt
    return arr


def test_urban():
    assert classify(_semantic({CLASS_BUILDING: 0.20})) == "urban"


def test_suburban():
    assert classify(_semantic({CLASS_BUILDING: 0.05})) == "suburban"


def test_roads():
    assert classify(_semantic({CLASS_ROAD: 0.10})) == "roads"


def test_farmland():
    assert classify(_semantic({CLASS_FARMLAND: 0.70})) == "farmland"


def test_forest():
    assert classify(_semantic({CLASS_FOREST: 0.70})) == "forest"


def test_water_takes_precedence():
    # water > 50% wins even when building also qualifies.
    assert classify(_semantic({CLASS_WATER: 0.6, CLASS_BUILDING: 0.3})) == "water"


def test_background():
    assert classify(_semantic({})) == "background"


def test_precedence_order():
    # urban beats farmland when both hold: building > 15% wins over farmland > 60%.
    f = {CLASS_BUILDING: 0.2, CLASS_FARMLAND: 0.7}
    assert classify_fractions(f) == "urban"
