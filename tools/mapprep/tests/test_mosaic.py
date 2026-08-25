"""T05-U-02 (synthetic tiles assemble without seam shifts) and
T05-I-01 (pyramid levels match the base downsampling)."""

import numpy as np
import rasterio
from conftest import NATIVE_GSD, PROVIDER, X0, Y0, Z, tile_color
from pyproj import Transformer

from mapprep import webmercator
from mapprep.mosaic import build_layer

EPSG = "EPSG:32637"


def _build(tmp_path, cache_root, bounds, target_gsd, tiles_to_skip=()):
    for x, y in tiles_to_skip:
        (cache_root / PROVIDER.id / f"{Z}/{x}/{y}.jpg").unlink()
    return build_layer(
        PROVIDER,
        bounds,
        target_gsd,
        Z,
        cache_root,
        tmp_path / "ortho_a.tif",
        tmp_path / "validity_a.tif",
        EPSG,
        offline=True,
    )


def _sample_interior_color(ds, grid, x, y):
    lon, lat = webmercator.tile_center_lonlat(x, y, Z)
    transformer = Transformer.from_crs("EPSG:4326", EPSG, always_xy=True)
    east, north = transformer.transform(lon, lat)
    col, row = grid.utm_to_pixel(east, north)
    win = rasterio.windows.Window(int(col) - 2, int(row) - 2, 4, 4)
    return ds.read(window=win)


def _is_cog(ds) -> bool:
    return ds.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"


def test_tiles_assemble_without_seam_shifts(tmp_path, cache_root, bounds_2x2, filled_cache_2x2):
    result = _build(tmp_path, cache_root, bounds_2x2, NATIVE_GSD)
    assert result.tiles_fetched == 4
    assert not result.missing_tiles
    with rasterio.open(result.ortho_path) as ds:
        assert _is_cog(ds)
        assert ds.crs.to_epsg() == 32637
        assert ds.profile["tiled"] and ds.profile["blockxsize"] == 256
        for x, y in [(X0, Y0), (X0 + 1, Y0), (X0, Y0 + 1), (X0 + 1, Y0 + 1)]:
            sampled = _sample_interior_color(ds, result.grid, x, y)
            expected = tile_color(x, y).reshape(3, 1, 1)
            diff = np.abs(sampled.astype(np.int32) - expected.astype(np.int32))
            assert diff.max() <= 1, f"tile ({x},{y}) content shifted: max diff {diff.max()}"


def test_adjacent_tile_boundary_content(tmp_path, cache_root, bounds_2x2, filled_cache_2x2):
    result = _build(tmp_path, cache_root, bounds_2x2, NATIVE_GSD)
    transformer = Transformer.from_crs("EPSG:4326", EPSG, always_xy=True)
    with rasterio.open(result.ortho_path) as ds:
        for (x_a, y_a), (x_b, y_b) in [
            ((X0, Y0), (X0 + 1, Y0)),
            ((X0, Y0), (X0, Y0 + 1)),
        ]:
            lon_a, lat_a = webmercator.tile_center_lonlat(x_a, y_a, Z)
            lon_b, lat_b = webmercator.tile_center_lonlat(x_b, y_b, Z)
            mid_lon = (lon_a + lon_b) / 2
            mid_lat = (lat_a + lat_b) / 2
            east, north = transformer.transform(mid_lon, mid_lat)
            col, row = result.grid.utm_to_pixel(east, north)
            win_a = (
                rasterio.windows.Window(int(col) - 6, int(row) - 3, 3, 6)
                if y_a == y_b
                else rasterio.windows.Window(int(col) - 3, int(row) - 6, 6, 3)
            )
            inside_a = ds.read(window=win_a)
            expected_a = tile_color(x_a, y_a).reshape(3, 1, 1)
            diff = np.abs(inside_a.astype(np.int32) - expected_a.astype(np.int32))
            assert diff.max() <= 1, f"seam shift near tile ({x_a},{y_a}) boundary"


def test_downsampled_mosaic_preserves_mean(tmp_path, cache_root, bounds_2x2, filled_cache_2x2):
    native = _build(tmp_path, cache_root, bounds_2x2, NATIVE_GSD)
    doubled = _build(tmp_path, cache_root, bounds_2x2, NATIVE_GSD * 2)
    assert doubled.mosaic_gsd_m == NATIVE_GSD * 2
    assert abs(doubled.grid.width * 2 - native.grid.width) <= 2
    assert abs(doubled.grid.height * 2 - native.grid.height) <= 2


def test_upsampling_refused(tmp_path, cache_root, bounds_2x2, filled_cache_2x2):
    result = _build(tmp_path, cache_root, bounds_2x2, NATIVE_GSD / 2)
    assert result.upsampling_refused
    assert result.mosaic_gsd_m == result.native_gsd_m


def test_pyramid_levels_match_base_downsampling(tmp_path, cache_root, bounds_2x2, filled_cache_2x2):
    result = _build(tmp_path, cache_root, bounds_2x2, NATIVE_GSD)
    with rasterio.open(result.ortho_path) as ds:
        overviews = ds.overviews(1)
        assert overviews == [2, 4, 8]
        for level in overviews:
            ovr = ds.read(1, out_shape=(ds.height // level, ds.width // level))
            decimated = ds.read(
                1,
                out_shape=(ds.height // level, ds.width // level),
                resampling=rasterio.enums.Resampling.average,
            )
            diff = np.abs(ovr.astype(np.int32) - decimated.astype(np.int32))
            assert diff.max() == 0, f"overview {level}x deviates from base downsampling"


def test_pyramid_mask_levels_exact(tmp_path, cache_root, bounds_2x2, filled_cache_2x2):
    result = _build(tmp_path, cache_root, bounds_2x2, NATIVE_GSD)
    with rasterio.open(result.mask_path) as ds:
        assert ds.overviews(1) == [2, 4, 8]
        for level in (2, 4):
            ovr = ds.read(1, out_shape=(ds.height // level, ds.width // level))
            decimated = ds.read(
                1,
                out_shape=(ds.height // level, ds.width // level),
                resampling=rasterio.enums.Resampling.average,
            )
            diff = np.abs(ovr.astype(np.int32) - decimated.astype(np.int32))
            assert diff.max() == 0, f"mask overview {level}x deviates (lossless)"
