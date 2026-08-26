"""geo.py tests: UTM projection of the geopack origin and round trips."""

import numpy as np

from orthoproto.geo import GeoRef

# Maykop corridor geopack manifest bounds (EPSG:32637).
EAST_MIN, EAST_MAX = 572559.6, 573208.8
NORTH_MIN, NORTH_MAX = 4963587.0, 4965038.4


def test_origin_inside_geopack_bounds():
    geo = GeoRef.from_epsg("EPSG:32637")
    east, north = geo.lonlat_to_utm(lon_deg=39.922, lat_deg=44.8285)
    assert EAST_MIN < east < EAST_MAX
    assert NORTH_MIN < north < NORTH_MAX


def test_lonlat_utm_roundtrip():
    geo = GeoRef.from_epsg("EPSG:32637")
    rng = np.random.default_rng(7)
    lons = rng.uniform(39.91, 39.93, 50)
    lats = rng.uniform(44.82, 44.84, 50)
    east_north = geo.lonlat_to_utm_many(lons, lats)
    lon2, lat2 = geo.utm_to_lonlat(east_north[0, 0], east_north[0, 1])
    assert abs(lon2 - lons[0]) < 1e-9
    assert abs(lat2 - lats[0]) < 1e-9
