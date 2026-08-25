"""Web Mercator tile math."""

import math

import pytest

from mapprep import webmercator


def test_lonlat_to_tile_known_values():
    assert webmercator.lonlat_to_tile(0.0, 0.0, 0) == (0, 0)
    assert webmercator.lonlat_to_tile(0.0, 0.0, 1) == (1, 1)
    assert webmercator.lonlat_to_tile(180.0, 0.0, 1) == (1, 1)
    assert webmercator.lonlat_to_tile(-180.0, 0.0, 1) == (0, 1)
    x, y = webmercator.lonlat_to_tile(39.922, 44.8285, 18)
    assert 0 <= x < 2**18 and 0 <= y < 2**18


def test_tile_center_round_trip():
    for z in (1, 10, 19):
        tiles = [(0, 0), (2**z - 1, 2**z - 1)]
        if z > 1:
            tiles.append((5, 7))
        for x, y in tiles:
            lon, lat = webmercator.tile_center_lonlat(x, y, z)
            assert webmercator.lonlat_to_tile(lon, lat, z) == (x, y)


def test_quadkey_round_trip():
    for x, y, z in [(0, 0, 1), (3, 5, 3), (117, 58, 7), (160149, 94455, 18)]:
        quadkey = webmercator.tile_to_quadkey(x, y, z)
        assert webmercator.quadkey_to_tile(quadkey) == (x, y, z)


def test_tile_bounds_cover_mercator_square():
    west, south, east, north = webmercator.tile_bounds_3857(0, 0, 0)
    assert west == -webmercator.HALF_CIRCUMFERENCE_M
    assert east == webmercator.HALF_CIRCUMFERENCE_M
    assert south == -webmercator.HALF_CIRCUMFERENCE_M
    assert north == webmercator.HALF_CIRCUMFERENCE_M


def test_gsd_scales_with_zoom_and_latitude():
    gsd_equator_18 = webmercator.gsd_m(0.0, 18)
    assert gsd_equator_18 == pytest.approx(0.59716, abs=1e-3)
    assert webmercator.gsd_m(0.0, 19) == pytest.approx(gsd_equator_18 / 2, abs=1e-6)
    assert webmercator.gsd_m(44.8285, 18) < gsd_equator_18


def test_tiles_for_bounds_cover_exactly():
    tiles = webmercator.tiles_for_bounds(39.918, 44.822, 39.926, 44.835, 18)
    xs = {x for x, _ in tiles}
    ys = {y for _, y in tiles}
    assert len(tiles) == len(xs) * len(ys)
    assert all(0 <= x < 2**18 for x in xs)
    assert all(0 <= y < 2**18 for y in ys)


def test_mercator_projection_identity():
    lat = 44.8285
    merc_y = math.asinh(math.tan(math.radians(lat))) * webmercator.EARTH_RADIUS_M
    assert merc_y < webmercator.HALF_CIRCUMFERENCE_M
