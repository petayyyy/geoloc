"""Geodetic helpers: WGS84 <-> geopack UTM CRS.

Nothing here reinterprets a fix: whether a capture's `latitude`/`longitude`
fields need swapping is a capture-config decision (`rtk.swap_latlon`, applied
once in `bagio.Capture.read_rtk`), never implicit in the math. Getting that
choice wrong is not a relabelling -- metres-per-degree differ per axis and per
latitude, so it anisotropically distorts the whole track. `rtkcheck` exists to
catch it; see README.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer


@dataclass
class GeoRef:
    """WGS84 <-> UTM converter for the geopack CRS.

    `Transformer.transform` is called with `always_xy=True`, so inputs are
    `(lon_deg, lat_deg)` and outputs `(easting_m, northing_m)` -- the same
    (x, y) order as the UTM raster axes.
    """

    crs: str
    _transformer: Transformer

    @classmethod
    def from_epsg(cls, epsg: str) -> GeoRef:
        return cls(crs=epsg, _transformer=Transformer.from_crs("EPSG:4326", epsg, always_xy=True))

    def lonlat_to_utm(self, lon_deg: float, lat_deg: float) -> tuple[float, float]:
        east, north = self._transformer.transform(lon_deg, lat_deg)
        return float(east), float(north)

    def lonlat_to_utm_many(self, lon_deg, lat_deg):
        """Vectorized: returns (N, 2) float64 array of (east, north) metres."""
        lon = np.asarray(lon_deg, dtype=np.float64)
        lat = np.asarray(lat_deg, dtype=np.float64)
        east, north = self._transformer.transform(lon, lat)
        return np.column_stack([np.asarray(east), np.asarray(north)])

    def utm_to_lonlat(self, east_m: float, north_m: float) -> tuple[float, float]:
        lon, lat = self._transformer.transform(east_m, north_m, direction="INVERSE")
        return float(lon), float(lat)
