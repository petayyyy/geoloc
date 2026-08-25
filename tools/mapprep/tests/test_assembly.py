"""T06-I-01: full geopack assembly (ortho + DEM + semantic), validator green."""

import json

import numpy as np
import rasterio
from affine import Affine
from conftest import NATIVE_GSD, PROVIDER, Z

from mapprep.dem import build_dem
from mapprep.geoid import ConstantGeoid
from mapprep.manifest import (
    add_dem_layer,
    add_ortho_layer,
    add_semantic_layer,
    new_manifest,
    write_manifest,
)
from mapprep.mosaic import build_layer
from mapprep.semantic import build_semantic
from mapprep.validator import validate_package
from mapprep.verify import verify_geotransform

EPSG = "EPSG:32637"


def _write_dem_source(path, bounds, value=200.0, step=0.001):
    west, south, east, north = bounds["west"], bounds["south"], bounds["east"], bounds["north"]
    width = int(np.ceil((east - west) / step)) + 1
    height = int(np.ceil((north - south) / step)) + 1
    transform = Affine(step, 0, west, 0, -step, north)
    array = np.full((height, width), value, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as ds:
        ds.write(array, 1)
    return path


def _farmland_fragment(bounds):
    west, south, east, north = bounds["west"], bounds["south"], bounds["east"], bounds["north"]
    return json.dumps(
        {
            "elements": [
                {"type": "node", "id": 1, "lat": north, "lon": west},
                {"type": "node", "id": 2, "lat": north, "lon": east},
                {"type": "node", "id": 3, "lat": south, "lon": east},
                {"type": "node", "id": 4, "lat": south, "lon": west},
                {
                    "type": "way",
                    "id": 10,
                    "nodes": [1, 2, 3, 4, 1],
                    "tags": {"landuse": "farmland"},
                },
            ]
        }
    )


def test_full_assembly_validator_green(tmp_path, cache_root, bounds_2x2, filled_cache_2x2):
    geopack = tmp_path / "mission.geopack"
    geopack.mkdir()
    bounds = bounds_2x2
    manifest = new_manifest("test-mission", EPSG, {}, {"lat": 44.8285, "lon": 39.922, "alt": 80.0})

    ortho = build_layer(
        PROVIDER,
        bounds,
        NATIVE_GSD,
        Z,
        cache_root,
        geopack / "ortho_a.tif",
        geopack / "validity_a.tif",
        EPSG,
        offline=True,
    )
    add_ortho_layer(manifest, "ortho_a", ortho, PROVIDER, bounds, EPSG)
    grid = ortho.grid
    manifest["bounds"] = {
        "east_min": grid.origin_east,
        "east_max": grid.origin_east + grid.width * grid.gsd,
        "north_min": grid.origin_north - grid.height * grid.gsd,
        "north_max": grid.origin_north,
    }

    dem_src = _write_dem_source(tmp_path / "dem_src.tif", bounds)
    dem = build_dem(
        bounds,
        10.0,
        [dem_src],
        geopack / "dem.tif",
        EPSG,
        geoid=ConstantGeoid(17.0),
    )
    add_dem_layer(manifest, dem, EPSG, bounds_wgs84=bounds)

    sem = build_semantic(
        bounds,
        1.0,
        geopack / "semantic.tif",
        EPSG,
        overpass_text=_farmland_fragment(bounds),
        offline=True,
        extract_date="2026-08-01",
    )
    add_semantic_layer(manifest, sem, EPSG, extract_date="2026-08-01")

    write_manifest(manifest, geopack / "manifest.yaml")

    assert verify_geotransform(manifest, geopack) == []
    assert validate_package(manifest, geopack) == []

    with rasterio.open(geopack / "dem.tif") as ds:
        array = ds.read(1)
        assert ds.crs.to_epsg() == 32637
        valid = array[array != ds.nodata]
        assert valid.size > 0
        assert np.allclose(valid, 217.0, atol=0.5), "orthometric 200 + geoid 17 -> ellipsoid"
