"""Web Mercator (EPSG:3857) tile math, pure Python.

Slippy-map XYZ convention: x from west, y from the north (row 0 = 85.05 N).
Bing quadkey is the bit-interleaved form of (x, y, z). Esri World Imagery and
Yandex use the same XYZ numbering as OSM.
"""

from __future__ import annotations

import math

EARTH_RADIUS_M = 6378137.0
HALF_CIRCUMFERENCE_M = math.pi * EARTH_RADIUS_M
TILE_SIZE = 256


def _clamp_lat(lat: float) -> float:
    return max(-85.0511287798066, min(85.0511287798066, lat))


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 1 << z
    x = int((lon + 180.0) / 360.0 * n)
    lat = _clamp_lat(lat)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return x, y


def tile_center_lonlat(x: int, y: int, z: int) -> tuple[float, float]:
    n = 1 << z
    lon = (x + 0.5) / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 0.5) / n))))
    return lon, lat


def tile_bounds_3857(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    n = 1 << z
    size = 2.0 * HALF_CIRCUMFERENCE_M / n
    west = -HALF_CIRCUMFERENCE_M + x * size
    east = west + size
    north = HALF_CIRCUMFERENCE_M - y * size
    south = north - size
    return west, south, east, north


def tile_corners_lonlat(x: int, y: int, z: int) -> list[tuple[float, float]]:
    west, south, east, north = tile_bounds_3857(x, y, z)

    def to_lon(merc_x: float) -> float:
        return merc_x / HALF_CIRCUMFERENCE_M * 180.0

    def to_lat(merc_y: float) -> float:
        return math.degrees(math.atan(math.sinh(merc_y / EARTH_RADIUS_M)))

    return [
        (to_lon(west), to_lat(north)),
        (to_lon(east), to_lat(north)),
        (to_lon(east), to_lat(south)),
        (to_lon(west), to_lat(south)),
    ]


def gsd_m(lat: float, z: int) -> float:
    return (
        2.0 * HALF_CIRCUMFERENCE_M / TILE_SIZE / (1 << z) * math.cos(math.radians(_clamp_lat(lat)))
    )


def tile_to_quadkey(x: int, y: int, z: int) -> str:
    key = ""
    for i in range(z, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        key += str(digit)
    return key


def quadkey_to_tile(quadkey: str) -> tuple[int, int, int]:
    x = y = 0
    z = len(quadkey)
    for i, ch in enumerate(quadkey):
        digit = int(ch)
        mask = 1 << (z - i - 1)
        if digit & 1:
            x |= mask
        if digit & 2:
            y |= mask
    return x, y, z


def tiles_for_bounds(
    west: float, south: float, east: float, north: float, z: int
) -> list[tuple[int, int]]:
    x_min, y_max = lonlat_to_tile(west, south, z)
    x_max, y_min = lonlat_to_tile(east, north, z)
    return [(x, y) for y in range(y_min, y_max + 1) for x in range(x_min, x_max + 1)]
