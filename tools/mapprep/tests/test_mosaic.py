"""T05-U-02 (synthetic tiles assemble without seam shifts) and
T05-I-01 (pyramid levels match the base downsampling)."""

import numpy as np
import pytest
import rasterio
from conftest import NATIVE_GSD, PROVIDER, X0, Y0, Z, tile_color, write_tile
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


# --- provider placeholder tiles (regression, 2026-08-27) --------------------
#
# Esri World Imagery answers any zoom it lacks locally with a grey "Map data
# not yet available" card -- a valid 200 response and a valid JPEG. Both
# geopacks built before this date had ortho_a made entirely of those cards,
# with validity_a reporting 100% valid, and passed verify/validate/classify.
# For a stack whose top metric is false-fix rate, a reference basemap that
# claims validity it does not have is the worst possible input.

from pathlib import Path  # noqa: E402

from mapprep.coverage import find_placeholder_tiles  # noqa: E402


def _write_raw_tile(cache_root, x, y, z, payload: bytes) -> Path:
    path = cache_root / PROVIDER.id / f"{z}/{x}/{y}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _placeholder_bytes() -> bytes:
    """A textureless card, as a real JPEG: flat grey with faint markings."""
    import io

    from PIL import Image

    arr = np.full((256, 256, 3), 205, dtype=np.uint8)
    arr[120:130, 40:210] = 212  # the faint text row
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _textured_bytes(seed: int) -> bytes:
    import io

    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def test_repeated_textureless_tiles_are_flagged(cache_root):
    card = _placeholder_bytes()
    paths = [_write_raw_tile(cache_root, X0 + i, Y0, Z, card) for i in range(6)]
    assert find_placeholder_tiles(paths) == set(paths)


def test_unique_flat_colour_tiles_are_not_flagged(cache_root):
    """A uniform field is textureless but real; only *repeated* cards count."""
    paths = []
    for i in range(6):
        write_tile(cache_root, X0 + i, Y0, Z, tile_color(X0 + i, Y0))
        paths.append(cache_root / PROVIDER.id / f"{Z}/{X0 + i}/{Y0}.jpg")
    assert find_placeholder_tiles(paths) == set()


def test_textured_tiles_are_never_flagged(cache_root):
    paths = [_write_raw_tile(cache_root, X0 + i, Y0, Z, _textured_bytes(i)) for i in range(6)]
    assert find_placeholder_tiles(paths) == set()


def test_a_few_identical_tiles_are_below_the_repeat_threshold(cache_root):
    card = _placeholder_bytes()
    paths = [_write_raw_tile(cache_root, X0 + i, Y0, Z, card) for i in range(2)]
    assert find_placeholder_tiles(paths, min_repeats=4) == set()


def test_detection_can_be_disabled(cache_root):
    card = _placeholder_bytes()
    paths = [_write_raw_tile(cache_root, X0 + i, Y0, Z, card) for i in range(6)]
    assert find_placeholder_tiles(paths, std_threshold=0.0) == set()


def test_mostly_placeholder_layer_refuses_to_build(cache_root, bounds_2x2, tmp_path):
    """Honest refusal: never ship a basemap that reports validity it lacks."""
    card = _placeholder_bytes()
    for dx in range(2):
        for dy in range(2):
            _write_raw_tile(cache_root, X0 + dx, Y0 + dy, Z, card)
    with pytest.raises(RuntimeError, match="are placeholders"):
        build_layer(
            PROVIDER,
            bounds_2x2,
            NATIVE_GSD,
            Z,
            cache_root,
            tmp_path / "ortho_a.tif",
            tmp_path / "validity_a.tif",
            "EPSG:32637",
            offline=True,
            placeholder_min_repeats=4,
        )
