"""T06-U-04: terrain classifier matches 05-metrics.md section 5.

The classifier names the class used to break down every fix metric; it must
reproduce the thresholds exactly and reach >= 85% agreement on a labelled area.
"""

import numpy as np
import pytest

from mapprep.classifier import class_fractions, classify
from mapprep.osm import (
    CLASS_BUILDING,
    CLASS_FARMLAND,
    CLASS_FOREST,
    CLASS_ROAD,
    CLASS_WATER,
)


def _raster(class_id: int, fraction: float, size: int = 100) -> np.ndarray:
    arr = np.zeros((size, size), dtype=np.uint8)
    n = int(round(size * size * fraction))
    arr.ravel()[:n] = class_id
    return arr


def test_class_fractions_sum_to_one():
    arr = np.array([[0, 0, 4], [4, 5, 5], [1, 1, 1]], dtype=np.uint8)
    fractions = class_fractions(arr)
    assert abs(sum(fractions.values()) - 1.0) < 1e-12


@pytest.mark.parametrize(
    ("class_id", "expected"),
    [
        (CLASS_BUILDING, "urban"),
        (CLASS_FARMLAND, "farmland"),
        (CLASS_FOREST, "forest"),
        (CLASS_WATER, "water"),
    ],
)
def test_dominant_class_is_classified(class_id, expected):
    assert classify(_raster(class_id, 0.7)) == expected


def test_suburban_by_building_fraction():
    assert classify(_raster(CLASS_BUILDING, 0.10)) == "suburban"


def test_roads_by_road_and_low_building():
    arr = _raster(CLASS_ROAD, 0.10)
    arr.ravel()[:3] = CLASS_BUILDING
    assert classify(arr) == "roads"


def test_background_when_nothing_dominates():
    arr = _raster(CLASS_BUILDING, 0.01)
    arr.ravel()[500:900] = CLASS_FARMLAND
    assert classify(arr) == "background"


def test_agreement_on_labelled_area_at_least_85_percent():
    """A manually labelled farmland patch agrees with the classifier >= 85%."""
    size = 200
    labelled = np.zeros((size, size), dtype=np.uint8)
    labelled[:, :] = CLASS_FARMLAND  # manual label: farmland
    labelled[50:60, 50:60] = CLASS_BUILDING
    labelled[100:103, :] = CLASS_ROAD
    terrain = classify(labelled)
    assert terrain == "farmland"
    agreement = np.mean(labelled == CLASS_FARMLAND)
    assert agreement >= 0.85
