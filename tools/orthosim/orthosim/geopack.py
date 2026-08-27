"""Geopack access for OrthoSim: georeferenced in-memory rasters + manifest.

Unlike the orthoproto prototype (which also reads the lidar DSM and bag), this
module only needs the *static map* side of a geopack: the two ortho providers
(A = map, B = query), the Copernicus DEM and the OSM semantic raster. It is
deliberately independent of any recorded capture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import yaml

# OSM semantic class ids (matches tools/mapprep/mapprep/osm.py and the manifest).
CLASS_BACKGROUND = 0
CLASS_ROAD = 1
CLASS_BUILDING = 2
CLASS_WATER = 3
CLASS_FARMLAND = 4
CLASS_FOREST = 5


@dataclass
class Field:
    """A georeferenced in-memory raster.

    ``data`` is ``(H, W)`` for single-band or ``(H, W, C)`` for multiband.
    ``transform`` is a rasterio affine mapping ``(col, row) -> (east, north)``;
    row increases southward, so ``transform.e`` is negative for north-up rasters.
    """

    data: np.ndarray
    transform: rasterio.Affine
    nodata: float | None = None

    @property
    def gsd(self) -> float:
        return float(self.transform.a)

    @property
    def east_min(self) -> float:
        return float(self.transform.c)

    @property
    def north_max(self) -> float:
        return float(self.transform.f)

    def _sample_channel(self, east: np.ndarray, north: np.ndarray, ch: int = 0) -> np.ndarray:
        # GDAL/geopack convention: the affine maps the pixel's top-left corner,
        # and pixel centres sit at (col + 0.5, row + 0.5) -- subtract 0.5 so an
        # integer index falls on a pixel centre (README geometry conventions).
        col = (east - self.transform.c) / self.transform.a - 0.5
        row = (north - self.transform.f) / self.transform.e - 0.5
        c0 = np.floor(col).astype(np.int64)
        r0 = np.floor(row).astype(np.int64)
        fc, fr = col - c0, row - r0
        h, w = self.data.shape[:2]
        valid = (c0 >= 0) & (c0 < w) & (r0 >= 0) & (r0 < h)
        c1 = np.clip(c0 + 1, 0, w - 1)
        r1 = np.clip(r0 + 1, 0, h - 1)
        c0 = np.clip(c0, 0, w - 1)
        r0 = np.clip(r0, 0, h - 1)

        if self.data.ndim == 2:
            src = self.data
        else:
            src = self.data[..., ch]

        out = np.zeros(east.shape, dtype=np.float64)
        if valid.any():
            p00 = src[r0, c0].astype(np.float64)
            p10 = src[r0, c1].astype(np.float64)
            p01 = src[r1, c0].astype(np.float64)
            p11 = src[r1, c1].astype(np.float64)
            vals = (p00 * (1 - fc) + p10 * fc) * (1 - fr) + (p01 * (1 - fc) + p11 * fc) * fr
            if self.nodata is not None:
                any_nodata = (
                    (p00 == self.nodata)
                    | (p10 == self.nodata)
                    | (p01 == self.nodata)
                    | (p11 == self.nodata)
                )
                vals = np.where(any_nodata, np.nan, vals)
            out[valid] = vals[valid]
        out[~valid] = np.nan
        return out

    def sample(self, east: np.ndarray, north: np.ndarray) -> np.ndarray:
        """Bilinear sample. Returns ``(N,)`` or ``(N, C)`` matching band count."""
        east = np.asarray(east, dtype=np.float64)
        north = np.asarray(north, dtype=np.float64)
        if self.data.ndim == 2:
            return self._sample_channel(east, north)
        channels = [self._sample_channel(east, north, ch) for ch in range(self.data.shape[2])]
        return np.stack(channels, axis=-1)


def open_field(path: Path, *, nodata: float | None = None) -> Field:
    """Read a raster file into an in-memory :class:`Field`."""
    with rasterio.open(path) as src:
        arr = src.read()
        if arr.ndim == 3:
            arr = np.moveaxis(arr, 0, -1)
        transform = src.transform
        file_nodata = src.nodata
    return Field(data=arr, transform=transform, nodata=file_nodata if nodata is None else nodata)


@dataclass
class GeoPack:
    """Static map side of a mission geopack (directory, per 03-interfaces.md §4)."""

    dir: Path

    @property
    def manifest(self) -> dict:
        with open(self.dir / "manifest.yaml", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def layer(self, name: str) -> dict:
        return self.manifest["layers"][name]

    def layer_path(self, name: str) -> Path:
        return self.dir / self.layer(name)["file"]

    def open_layer(self, name: str) -> Field:
        return open_field(self.layer_path(name))

    @property
    def crs(self) -> str:
        return self.manifest["crs"]

    @property
    def provider(self, name: str) -> str:
        return self.layer(name).get("provider", name)

    def open_ortho(self, name: str) -> Field:
        return self.open_layer(name)

    def open_dem(self) -> Field:
        return self.open_layer("dem")

    def open_semantic(self) -> Field | None:
        if "semantic" not in self.manifest["layers"]:
            return None
        return self.open_layer("semantic")
