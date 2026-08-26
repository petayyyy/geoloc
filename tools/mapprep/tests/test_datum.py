"""T06-U-01: vertical datum conversion (EGM2008 -> ellipsoid).

The conversion is arithmetic (ellipsoidal = orthometric + N); the error budget
comes from the geoid model. Control points here use (a) a constant geoid where
the result is exact, and (b) a synthetic grid geoid with a known undulation
field, checked against the same control points with an error <= 0.5 m.
"""

import numpy as np
import rasterio
from affine import Affine

from mapprep.datum import ellipsoidal_to_orthometric, orthometric_to_ellipsoidal
from mapprep.geoid import ConstantGeoid, GridGeoid

CONTROL_POINTS = [
    (44.8225, 39.9185, 202.0),  # lat, lon, orthometric height (m)
    (44.8285, 39.9220, 188.5),
    (44.8345, 39.9255, 175.25),
]


def _write_geoid_grid(path, fn, west=39.0, east=41.0, south=44.0, north=46.0, step=0.1):
    width = int(round((east - west) / step))
    height = int(round((north - south) / step))
    transform = Affine(step, 0, west, 0, -step, north)
    lons = west + (np.arange(width) + 0.5) * step
    lats = north - (np.arange(height) + 0.5) * step
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    values = fn(lat_grid, lon_grid).astype(np.float32)
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
    ) as ds:
        ds.write(values, 1)
    return path


def _undulation_field(lat, lon):
    return 15.0 + 0.3 * (lat - 44.0) + 0.1 * (lon - 39.0)


def test_constant_geoid_conversion_is_exact():
    geoid = ConstantGeoid(17.3)
    for lat, lon, ortho in CONTROL_POINTS:
        ellip = orthometric_to_ellipsoidal(
            np.array([ortho]), np.array([lat]), np.array([lon]), geoid
        )
        assert abs(ellip[0] - (ortho + 17.3)) < 1e-4


def test_round_trip_is_identity():
    geoid = ConstantGeoid(12.7)
    lat = np.array([44.8285])
    lon = np.array([39.922])
    ortho = np.array([190.0])
    ellip = orthometric_to_ellipsoidal(ortho, lat, lon, geoid)
    back = ellipsoidal_to_orthometric(ellip, lat, lon, geoid)
    assert abs(back[0] - ortho[0]) < 1e-4


def test_grid_geoid_control_points_within_half_metre(tmp_path):
    path = _write_geoid_grid(tmp_path / "geoid.tif", _undulation_field)
    geoid = GridGeoid(path)
    try:
        for lat, lon, ortho in CONTROL_POINTS:
            n = geoid.undulation(lat, lon)
            expected_n = _undulation_field(np.array(lat), np.array(lon))
            assert abs(n - float(expected_n)) <= 0.5 * 0.1, "nearest-neighbour grid sampling"
            ellip = orthometric_to_ellipsoidal(
                np.array([ortho]), np.array([lat]), np.array([lon]), geoid
            )
            assert abs(ellip[0] - (ortho + n)) < 1e-4
            assert abs(ellip[0] - (ortho + float(expected_n))) <= 0.5
    finally:
        geoid.close()


def test_grid_geoid_array_sampling(tmp_path):
    path = _write_geoid_grid(tmp_path / "geoid.tif", _undulation_field)
    geoid = GridGeoid(path)
    try:
        lats = np.array([44.82, 44.83, 44.84])
        lons = np.array([39.91, 39.92, 39.93])
        values = geoid.undulation_array(lats, lons)
        expected = _undulation_field(lats, lons)
        assert np.allclose(values, expected, atol=0.05)
    finally:
        geoid.close()


def test_grid_geoid_2d_array_sampling(tmp_path):
    # dem.py samples the geoid on a 2D (height, width) lat/lon grid, not a
    # flat list of points; the result must keep that shape.
    path = _write_geoid_grid(tmp_path / "geoid.tif", _undulation_field)
    geoid = GridGeoid(path)
    try:
        lats = np.array([[44.82, 44.83], [44.84, 44.85]])
        lons = np.array([[39.91, 39.92], [39.93, 39.94]])
        values = geoid.undulation_array(lats, lons)
        assert values.shape == lats.shape
        expected = _undulation_field(lats, lons)
        assert np.allclose(values, expected, atol=0.05)
    finally:
        geoid.close()


def test_constant_geoid_array_broadcasts():
    geoid = ConstantGeoid(9.0)
    lats = np.zeros((3, 4))
    lons = np.zeros((3, 4))
    values = geoid.undulation_array(lats, lons)
    assert values.shape == (3, 4)
    assert np.allclose(values, 9.0)
