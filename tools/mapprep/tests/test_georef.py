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


# --- manifest bounds must be the snapping base every layer shares ----------
#
# Regression 2026-08-27. `_manifest_bounds` reported the FIRST layer's raster
# bounds, which are already floored to that layer's own gsd. `verify` then
# re-derives every other layer's grid from them, so a geopack mixing 0.3 m and
# 0.5 m layers failed with
#   ortho_b.tif: origin_east differs: manifest 484744.5, file 484745.0
# The Maykop corridor's numbers happened to align, so this never fired there.

from pyproj import Transformer  # noqa: E402

from mapprep.cli import _manifest_bounds  # noqa: E402

AMTOWN_WGS84 = {"west": 44.8215, "south": 39.9180, "east": 44.8355, "north": 39.9265}
AMTOWN_CRS = "EPSG:32638"


def _layer_grid_as_the_mosaic_builds_it(bounds_wgs84, crs, gsd):
    """Exactly what mosaic.build_layer does: project the corners, then snap."""
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    west, east = bounds_wgs84["west"], bounds_wgs84["east"]
    south, north = bounds_wgs84["south"], bounds_wgs84["north"]
    easts, norths = transformer.transform([west, east, east, west], [north, north, south, south])
    return PixelGrid.from_bounds(min(easts), max(easts), min(norths), max(norths), gsd)


def _grid_as_verify_derives_it(manifest_bounds, gsd):
    return PixelGrid.from_bounds(
        manifest_bounds["east_min"],
        manifest_bounds["east_max"],
        manifest_bounds["north_min"],
        manifest_bounds["north_max"],
        gsd,
    )


def test_every_layer_gsd_snaps_the_same_from_manifest_bounds():
    bounds = _manifest_bounds(AMTOWN_WGS84, AMTOWN_CRS)
    for gsd in (0.3, 0.5, 1.0, 10.0):
        built = _layer_grid_as_the_mosaic_builds_it(AMTOWN_WGS84, AMTOWN_CRS, gsd)
        derived = _grid_as_verify_derives_it(bounds, gsd)
        assert derived == built, f"gsd {gsd}: verify would see {derived}, file has {built}"


def test_snapped_layer_bounds_are_not_a_valid_snapping_base():
    """Pin down the actual defect, so the fix can't quietly regress."""
    ortho_a = _layer_grid_as_the_mosaic_builds_it(AMTOWN_WGS84, AMTOWN_CRS, 0.3)
    from_first_layer = {
        "east_min": ortho_a.origin_east,
        "east_max": ortho_a.origin_east + ortho_a.width * ortho_a.gsd,
        "north_min": ortho_a.origin_north - ortho_a.height * ortho_a.gsd,
        "north_max": ortho_a.origin_north,
    }
    ortho_b = _layer_grid_as_the_mosaic_builds_it(AMTOWN_WGS84, AMTOWN_CRS, 0.5)
    assert _grid_as_verify_derives_it(from_first_layer, 0.5) != ortho_b


def test_manifest_bounds_lie_inside_every_layer_raster():
    """Snapping only ever expands outward, so `validate`'s coverage check holds."""
    bounds = _manifest_bounds(AMTOWN_WGS84, AMTOWN_CRS)
    for gsd in (0.3, 0.5, 10.0):
        grid = _layer_grid_as_the_mosaic_builds_it(AMTOWN_WGS84, AMTOWN_CRS, gsd)
        assert grid.origin_east <= bounds["east_min"]
        assert grid.origin_east + grid.width * grid.gsd >= bounds["east_max"]
        assert grid.origin_north >= bounds["north_max"]
        assert grid.origin_north - grid.height * grid.gsd <= bounds["north_min"]
