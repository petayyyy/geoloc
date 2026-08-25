"""Import from pre-downloaded tile caches (SAS.Planet / QGIS XYZ layouts)."""

import numpy as np
import rasterio
from conftest import NATIVE_GSD, PROVIDER, X0, Y0, Z, tile_color

from mapprep import webmercator
from mapprep.mosaic import build_layer
from mapprep.tilecache import import_cache

EPSG = "EPSG:32637"


def _write_tile(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "PNG",
        "width": 256,
        "height": 256,
        "count": 3,
        "dtype": "uint8",
    }
    if path.suffix.lower() in (".jpg", ".jpeg"):
        profile = {**profile, "driver": "JPEG", "quality": 85}
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(np.broadcast_to(color.reshape(3, 1, 1), (3, 256, 256)))


def test_import_sasplanet_layout(tmp_path, cache_root, bounds_2x2):
    src = tmp_path / "sas_cache"
    for x, y in webmercator.tiles_for_bounds(
        bounds_2x2["west"], bounds_2x2["south"], bounds_2x2["east"], bounds_2x2["north"], Z
    ):
        _write_tile(src / f"z{Z + 1}" / str(x) / f"{y}.jpg", tile_color(x, y))
    imported = import_cache(src, PROVIDER, cache_root, layout="sasplanet")
    assert imported == 4
    for x, y in webmercator.tiles_for_bounds(
        bounds_2x2["west"], bounds_2x2["south"], bounds_2x2["east"], bounds_2x2["north"], Z
    ):
        assert (cache_root / PROVIDER.id / f"{Z}/{x}/{y}.jpg").exists()


def test_import_qgis_xyz_layout(tmp_path, cache_root, bounds_2x2):
    src = tmp_path / "qgis_xyz"
    x, y = X0, Y0
    _write_tile(src / str(Z) / str(x) / f"{y}.png", tile_color(x, y))
    imported = import_cache(src, PROVIDER, cache_root, layout="qgis_xyz")
    assert imported == 1
    assert (cache_root / PROVIDER.id / f"{Z}/{x}/{y}.png").exists()


def test_imported_cache_builds_identical_mosaic(tmp_path, cache_root, bounds_2x2):
    tiles = webmercator.tiles_for_bounds(
        bounds_2x2["west"], bounds_2x2["south"], bounds_2x2["east"], bounds_2x2["north"], Z
    )
    colors = {tile: tile_color(*tile) for tile in tiles}
    src = tmp_path / "sas_cache"
    for x, y in tiles:
        _write_tile(src / f"z{Z + 1}" / str(x) / f"{y}.jpg", colors[(x, y)])
    assert import_cache(src, PROVIDER, cache_root, layout="sasplanet") == len(tiles)
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
    assert result.tiles_fetched == len(tiles)
    with rasterio.open(result.ortho_path) as ds:
        data = ds.read()
    assert data.shape[0] == 3
    assert data.max() > 0


def test_import_skips_existing(tmp_path, cache_root):
    src = tmp_path / "sas_cache"
    _write_tile(src / f"z{Z + 1}" / str(X0) / f"{Y0}.jpg", tile_color(X0, Y0))
    assert import_cache(src, PROVIDER, cache_root, layout="sasplanet") == 1
    assert import_cache(src, PROVIDER, cache_root, layout="sasplanet") == 0
