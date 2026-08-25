"""Cloud polygons -> zero validity mask."""

import json

import rasterio
from conftest import NATIVE_GSD, PROVIDER, X0, Y0, Z
from pyproj import Transformer

from mapprep import webmercator
from mapprep.mask import apply_polygons_to_mask, load_geojson_polygons
from mapprep.mosaic import build_layer

EPSG = "EPSG:32637"


def _geojson(ring) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        ],
    }


def test_geojson_polygon_loading(tmp_path):
    ring = [[39.918, 44.822], [39.926, 44.822], [39.926, 44.835], [39.918, 44.835]]
    path = tmp_path / "clouds.geojson"
    path.write_text(json.dumps(_geojson(ring)), encoding="utf-8")
    rings = load_geojson_polygons(path)
    assert len(rings) == 1
    assert len(rings[0]) == 4


def test_cloud_polygon_zeroes_mask(tmp_path, cache_root, bounds_2x2, filled_cache_2x2):
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
    ring = webmercator.tile_corners_lonlat(X0, Y0, Z)
    cloud_polygons = [[(lon, lat) for lon, lat in ring]]
    with rasterio.open(result.mask_path) as ds:
        grid = result.grid
        mask = ds.read(1)
    transformer = Transformer.from_crs("EPSG:4326", EPSG, always_xy=True)
    cleared = apply_polygons_to_mask(mask, grid, transformer, cloud_polygons)
    assert cleared > 0
    lon, lat = webmercator.tile_center_lonlat(X0, Y0, Z)
    east, north = transformer.transform(lon, lat)
    col, row = grid.utm_to_pixel(east, north)
    assert mask[int(row), int(col)] == 0
    lon, lat = webmercator.tile_center_lonlat(X0 + 1, Y0, Z)
    east, north = transformer.transform(lon, lat)
    col, row = grid.utm_to_pixel(east, north)
    assert mask[int(row), int(col)] == 255
