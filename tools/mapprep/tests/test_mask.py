"""T05-U-03: missing tile -> zero validity mask in its bounds."""

import rasterio
from conftest import NATIVE_GSD, PROVIDER, X0, Y0, Z
from pyproj import Transformer

from mapprep import webmercator
from mapprep.mosaic import build_layer

EPSG = "EPSG:32637"


def test_missing_tile_yields_zero_mask(tmp_path, cache_root, bounds_2x2, filled_cache_2x2):
    result = build_layer(
        PROVIDER,
        bounds_2x2,
        NATIVE_GSD,
        Z,
        cache_root,
        tmp_path / "ortho_a.tif",
        tmp_path / "validity_a.tif",
        EPSG,
        offline=True,
    )
    with rasterio.open(result.mask_path) as ds:
        mask = ds.read(1)
    assert result.tiles_fetched == 4
    assert not result.missing_tiles
    assert mask.max() == 255
    assert mask.min() == 0


def _mask_at_tile(mask, grid, x, y):
    lon, lat = webmercator.tile_center_lonlat(x, y, Z)
    transformer = Transformer.from_crs("EPSG:4326", EPSG, always_xy=True)
    east, north = transformer.transform(lon, lat)
    col, row = grid.utm_to_pixel(east, north)
    return mask[int(row), int(col)]


def test_hole_matches_missing_tile_bounds(tmp_path, cache_root, bounds_2x2, filled_cache_2x2):
    hole = (X0 + 1, Y0 + 1)
    (cache_root / PROVIDER.id / f"{Z}/{hole[0]}/{hole[1]}.jpg").unlink()
    result = build_layer(
        PROVIDER,
        bounds_2x2,
        NATIVE_GSD,
        Z,
        cache_root,
        tmp_path / "ortho_a.tif",
        tmp_path / "validity_a.tif",
        EPSG,
        offline=True,
    )
    assert result.missing_tiles == [hole]
    assert result.tiles_fetched == 3
    with rasterio.open(result.mask_path) as ds:
        mask = ds.read(1)
    for x, y in [(X0, Y0), (X0 + 1, Y0), (X0, Y0 + 1)]:
        assert _mask_at_tile(mask, result.grid, x, y) == 255
    assert _mask_at_tile(mask, result.grid, hole[0], hole[1]) == 0


def test_mask_geometry_matches_ortho(tmp_path, cache_root, bounds_2x2, filled_cache_2x2):
    result = build_layer(
        PROVIDER,
        bounds_2x2,
        NATIVE_GSD,
        Z,
        cache_root,
        tmp_path / "ortho_a.tif",
        tmp_path / "validity_a.tif",
        EPSG,
        offline=True,
    )
    with rasterio.open(result.ortho_path) as ortho, rasterio.open(result.mask_path) as mask:
        assert (ortho.width, ortho.height) == (mask.width, mask.height)
        assert ortho.transform == mask.transform
        assert ortho.crs == mask.crs


def test_missing_marker_recorded_offline(tmp_path, cache_root, bounds_2x2, filled_cache_2x2):
    from mapprep.fetch import missing_marker

    hole = (X0, Y0)
    (cache_root / PROVIDER.id / f"{Z}/{hole[0]}/{hole[1]}.jpg").unlink()
    marker = missing_marker(PROVIDER, cache_root, hole[0], hole[1], Z)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    result = build_layer(
        PROVIDER,
        bounds_2x2,
        NATIVE_GSD,
        Z,
        cache_root,
        tmp_path / "ortho_a.tif",
        tmp_path / "validity_a.tif",
        EPSG,
        offline=True,
    )
    assert (hole[0], hole[1]) in result.missing_tiles


def test_seam_between_capture_dates_marked(tmp_path, cache_root, bounds_2x2, filled_cache_2x2):
    from mapprep.fetch import load_meta, record_tile_meta

    record_tile_meta(PROVIDER, cache_root, X0, Y0, Z, capture_date="2023-05")
    record_tile_meta(PROVIDER, cache_root, X0 + 1, Y0, Z, capture_date="2024-08")
    record_tile_meta(PROVIDER, cache_root, X0, Y0 + 1, Z, capture_date="2023-05")
    record_tile_meta(PROVIDER, cache_root, X0 + 1, Y0 + 1, Z, capture_date="2023-05")
    assert load_meta(PROVIDER, cache_root)
    result = build_layer(
        PROVIDER,
        bounds_2x2,
        NATIVE_GSD,
        Z,
        cache_root,
        tmp_path / "ortho_a.tif",
        tmp_path / "validity_a.tif",
        EPSG,
        offline=True,
        enrich_dates=True,
    )
    assert result.seam_count > 0
    with rasterio.open(result.mask_path) as ds:
        mask = ds.read(1)
    transformer = Transformer.from_crs("EPSG:4326", EPSG, always_xy=True)
    lon_a, lat_a = webmercator.tile_center_lonlat(X0, Y0, Z)
    lon_b, lat_b = webmercator.tile_center_lonlat(X0 + 1, Y0, Z)
    east, north = transformer.transform((lon_a + lon_b) / 2, (lat_a + lat_b) / 2)
    col, row = result.grid.utm_to_pixel(east, north)
    assert mask[int(row), int(col)] == 0
