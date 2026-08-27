"""geopack.Field bilinear sampling tests (multiband + nodata handling)."""

from __future__ import annotations

import numpy as np
import rasterio.transform

from orthosim.geopack import Field


def _field_3band():
    # 4x4 RGB image at 1 m/px, origin (0, 4).
    r = np.arange(16, dtype=np.float64).reshape(4, 4)
    data = np.stack([r, r + 100, r + 200], axis=-1)
    return Field(
        data=data,
        transform=rasterio.transform.from_origin(0.0, 4.0, 1.0, 1.0),
    )


def test_multiband_bilinear_exact_at_centres():
    f = _field_3band()
    # pixel centre (col=1, row=1) -> (east=1.5, north=2.5)
    v = f.sample(np.array([1.5]), np.array([2.5]))
    assert v.shape == (1, 3)
    expected = f.data[1, 1]
    assert np.allclose(v[0], expected)


def test_multiband_outside_is_nan():
    f = _field_3band()
    v = f.sample(np.array([100.0]), np.array([100.0]))
    assert np.all(np.isnan(v))


def test_single_band_nodata_invalidates_cell():
    z = np.full((4, 4), 10.0, dtype=np.float64)
    z[0, 0] = -9999.0
    f = Field(
        data=z,
        transform=rasterio.transform.from_origin(0.0, 4.0, 1.0, 1.0),
        nodata=-9999.0,
    )
    # straddling the nodata corner
    v = f.sample(np.array([0.5]), np.array([3.5]))
    assert np.isnan(v[0])
