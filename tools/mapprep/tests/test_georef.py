"""T05-U-01: pixel <-> UTM round-trip.

Error budget: with float64 absolute UTM coordinates (~5e6 m), the forward
mapping rounds at half an ulp of the coordinate magnitude (≈4.7e-10 m at
northing 4.97e6). The round trip is therefore exact to well under a nanometre
in metres (< 1e-9 m), which corresponds to < 5e-9 px at 0.3 m/px. These are
the strongest bounds float64 admits; the mapping itself is exactly invertible.
"""

import numpy as np

from mapprep.georef import PixelGrid, round_trip_error_px


def _round_trip_error_m(grid, col, row):
    east, north = grid.pixel_center_to_utm(col, row)
    col_back, row_back = grid.utm_to_pixel(east, north)
    return abs(col_back - col) * grid.gsd, abs(row_back - row) * grid.gsd


def test_round_trip_arbitrary_pixels():
    grid = PixelGrid.from_bounds(572000.0, 573000.0, 4972000.0, 4973000.0, 0.3)
    rng = np.random.default_rng(42)
    worst_px = 0.0
    worst_m = 0.0
    for _ in range(10000):
        col = rng.uniform(-10, grid.width + 10)
        row = rng.uniform(-10, grid.height + 10)
        error_px = round_trip_error_px(grid, col, row)
        error_e_m, error_n_m = _round_trip_error_m(grid, col, row)
        worst_px = max(worst_px, error_px)
        worst_m = max(worst_m, error_e_m, error_n_m)
    assert worst_m < 1e-9, f"round-trip error {worst_m} m"
    assert worst_px < 5e-9, f"round-trip error {worst_px} px"


def test_round_trip_pixel_centers():
    grid = PixelGrid.from_bounds(500000.0, 500100.0, 4970000.0, 4970100.0, 0.5)
    for col in range(0, grid.width, 7):
        for row in range(0, grid.height, 7):
            east, north = grid.pixel_center_to_utm(col, row)
            col_back, row_back = grid.utm_to_pixel(east, north)
            assert abs(col_back - col) * grid.gsd < 1e-9
            assert abs(row_back - row) * grid.gsd < 1e-9


def test_grid_snap_and_center_convention():
    grid = PixelGrid.from_bounds(572000.123, 572999.877, 4972000.456, 4972999.544, 0.3)
    assert grid.origin_east == 572000.1
    assert grid.origin_north == 4972999.8
    assert grid.width == 3333
    assert grid.height == 3332
    east, north = grid.pixel_center_to_utm(0, 0)
    assert east == grid.origin_east + 0.15
    assert north == grid.origin_north - 0.15


def test_row_increases_southward():
    grid = PixelGrid.from_bounds(0.0, 10.0, 0.0, 10.0, 1.0)
    _, north_top = grid.pixel_center_to_utm(0, 0)
    _, north_below = grid.pixel_center_to_utm(0, 1)
    assert north_below < north_top
