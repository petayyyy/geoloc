"""T06-U-02: a synthetic OSM fragment rasterizes to the expected class grid."""

import json

import numpy as np
from pyproj import Transformer

from mapprep.georef import PixelGrid
from mapprep.osm import (
    CLASS_BUILDING,
    CLASS_FARMLAND,
    CLASS_FOREST,
    CLASS_ROAD,
    CLASS_WATER,
    classify_tags,
    rasterize_overpass,
)

EPSG = "EPSG:32637"
TO_UTM = Transformer.from_crs("EPSG:4326", EPSG, always_xy=True)

WEST, SOUTH, EAST, NORTH = 39.918, 44.822, 39.926, 44.835


def _grid(gsd=1.0):
    corners = TO_UTM.transform([WEST, EAST, EAST, WEST], [NORTH, NORTH, SOUTH, SOUTH])
    return PixelGrid.from_bounds(
        min(corners[0]), max(corners[0]), min(corners[1]), max(corners[1]), gsd
    )


def _pixel(lon, lat, grid):
    east, north = TO_UTM.transform(lon, lat)
    col, row = grid.utm_to_pixel(east, north)
    return int(round(col)), int(round(row))


def _fragment():
    """A farmland square, a building square inside it, and a road line above."""
    nodes = [
        {"type": "node", "id": 1, "lat": NORTH, "lon": WEST},
        {"type": "node", "id": 2, "lat": NORTH, "lon": EAST},
        {"type": "node", "id": 3, "lat": SOUTH, "lon": EAST},
        {"type": "node", "id": 4, "lat": SOUTH, "lon": WEST},
        {"type": "node", "id": 5, "lat": 44.8295, "lon": 39.921},
        {"type": "node", "id": 6, "lat": 44.8295, "lon": 39.923},
        {"type": "node", "id": 7, "lat": 44.8275, "lon": 39.923},
        {"type": "node", "id": 8, "lat": 44.8275, "lon": 39.921},
        {"type": "node", "id": 9, "lat": 44.8310, "lon": WEST},
        {"type": "node", "id": 10, "lat": 44.8310, "lon": EAST},
    ]
    ways = [
        {"type": "way", "id": 100, "nodes": [1, 2, 3, 4, 1], "tags": {"landuse": "farmland"}},
        {"type": "way", "id": 101, "nodes": [5, 6, 7, 8, 5], "tags": {"building": "yes"}},
        {"type": "way", "id": 102, "nodes": [9, 10], "tags": {"highway": "residential"}},
    ]
    return json.dumps({"elements": nodes + ways})


def test_classify_tags_mapping():
    cases = [
        ({"building": "yes"}, CLASS_BUILDING),
        ({"highway": "residential"}, CLASS_ROAD),
        ({"natural": "water"}, CLASS_WATER),
        ({"waterway": "river"}, CLASS_WATER),
        ({"landuse": "farmland"}, CLASS_FARMLAND),
        ({"landuse": "forest"}, CLASS_FOREST),
        ({"natural": "wood"}, CLASS_FOREST),
        ({"landuse": "reservoir"}, CLASS_WATER),
        ({"building": "yes", "natural": "water"}, CLASS_WATER),
        ({"name": "untagged"}, None),
    ]
    for tags, expected in cases:
        assert classify_tags(tags) == expected, tags


def test_synthetic_fragment_rasterizes_expected_classes():
    grid = _grid()
    arr = rasterize_overpass(_fragment(), grid, TO_UTM)

    building_col, building_row = _pixel(39.922, 44.8285, grid)
    assert arr[building_row, building_col] == CLASS_BUILDING

    road_col, road_row = _pixel(39.922, 44.8310, grid)
    assert arr[road_row, road_col] == CLASS_ROAD

    farm_col, farm_row = _pixel(39.9245, 44.8230, grid)
    assert arr[farm_row, farm_col] == CLASS_FARMLAND


def test_water_wins_over_building():
    """Paint order: water (stable class) overwrites an overlapping building."""
    fragment = json.dumps(
        {
            "elements": [
                {"type": "node", "id": 1, "lat": 44.830, "lon": 39.921},
                {"type": "node", "id": 2, "lat": 44.830, "lon": 39.923},
                {"type": "node", "id": 3, "lat": 44.828, "lon": 39.923},
                {"type": "node", "id": 4, "lat": 44.828, "lon": 39.921},
                {
                    "type": "way",
                    "id": 100,
                    "nodes": [1, 2, 3, 4, 1],
                    "tags": {"building": "yes"},
                },
                {
                    "type": "way",
                    "id": 101,
                    "nodes": [1, 2, 3, 4, 1],
                    "tags": {"natural": "water"},
                },
            ]
        }
    )
    grid = _grid()
    arr = rasterize_overpass(fragment, grid, TO_UTM)
    col, row = _pixel(39.922, 44.829, grid)
    assert arr[row, col] == CLASS_WATER


def test_empty_fragment_yields_background():
    grid = _grid()
    arr = rasterize_overpass(json.dumps({"elements": []}), grid, TO_UTM)
    assert np.all(arr == 0)
