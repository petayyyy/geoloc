"""Pixel <-> UTM georeferencing for the mosaic grid.

Conventions (README.md "Geometry conventions"):
  - pixel (0, 0) is the TOP-LEFT corner; centres at (col + 0.5, row + 0.5)
  - row increases southward
  - the grid origin is snapped to whole multiples of gsd so mosaics built
    independently for neighbouring corridors tile cleanly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from affine import Affine


@dataclass(frozen=True)
class PixelGrid:
    origin_east: float
    origin_north: float
    gsd: float
    width: int
    height: int

    def pixel_center_to_utm(self, col: float, row: float) -> tuple[float, float]:
        east = self.origin_east + (col + 0.5) * self.gsd
        north = self.origin_north - (row + 0.5) * self.gsd
        return east, north

    def utm_to_pixel(self, east: float, north: float) -> tuple[float, float]:
        col = (east - self.origin_east) / self.gsd - 0.5
        row = (self.origin_north - north) / self.gsd - 0.5
        return col, row

    def affine(self, col_offset: int = 0, row_offset: int = 0) -> Affine:
        return Affine.translation(
            self.origin_east + col_offset * self.gsd,
            self.origin_north - row_offset * self.gsd,
        ) * Affine.scale(self.gsd, -self.gsd)

    def window(
        self, east_min: float, east_max: float, north_min: float, north_max: float
    ) -> tuple[int, int, int, int]:
        col_min = int(math.floor((east_min - self.origin_east) / self.gsd))
        col_max = int(math.ceil((east_max - self.origin_east) / self.gsd))
        row_min = int(math.floor((self.origin_north - north_max) / self.gsd))
        row_max = int(math.ceil((self.origin_north - north_min) / self.gsd))
        col_min = max(0, min(self.width, col_min))
        col_max = max(0, min(self.width, col_max))
        row_min = max(0, min(self.height, row_min))
        row_max = max(0, min(self.height, row_max))
        return col_min, row_min, col_max, row_max

    @classmethod
    def from_bounds(
        cls,
        east_min: float,
        east_max: float,
        north_min: float,
        north_max: float,
        gsd: float,
    ) -> PixelGrid:
        if gsd <= 0:
            raise ValueError("gsd must be > 0")
        if east_max <= east_min or north_max <= north_min:
            raise ValueError("empty bounds")
        origin_east = math.floor(east_min / gsd) * gsd
        origin_north = math.ceil(north_max / gsd) * gsd
        width = int(math.ceil((east_max - origin_east) / gsd))
        height = int(math.ceil((origin_north - north_min) / gsd))
        return cls(origin_east, origin_north, gsd, width, height)


def round_trip_error_px(grid: PixelGrid, col: float, row: float) -> float:
    east, north = grid.pixel_center_to_utm(col, row)
    c2, r2 = grid.utm_to_pixel(east, north)
    return math.hypot(c2 - col, r2 - row)
