"""Height fields for OrthoSim rendering.

v1 uses the Copernicus DEM (or a flat plane) directly. v2 composes a synthetic
DSM by extruding OSM building footprints out of the DEM so that the rendered
query carries real building parallax (T11 / A-DSM-05) and can feed the ray-cast
shadow model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geopack import CLASS_BUILDING, Field


@dataclass
class FlatTerrain:
    """Constant-height ground plane (v1 placeholder)."""

    height: float = 0.0

    def z(self, east: np.ndarray, north: np.ndarray) -> np.ndarray:
        east = np.asarray(east, dtype=np.float64)
        return np.full(east.shape, self.height, dtype=np.float64)


@dataclass
class DemTerrain:
    """Terrain sampled bilinearly from a DEM field (Copernicus GLO-30)."""

    dem: Field

    def z(self, east: np.ndarray, north: np.ndarray) -> np.ndarray:
        return self.dem.sample(east, north)


@dataclass
class BuildingTerrain:
    """v2 synthetic DSM: DEM ground + extruded OSM building footprints.

    Each semantic cell classified as ``building`` is raised to ``ground +
    building_height``. A small margin (``pad_cells``) dilates the footprint so
    the walls are visible from oblique viewpoints; this is the cheapest model
    that still produces true building parallax and cast shadows.
    """

    dem: Field
    semantic: Field
    building_height: float = 15.0
    pad_cells: int = 1

    def __post_init__(self) -> None:
        footprint = self.semantic.data == CLASS_BUILDING
        if self.pad_cells > 0:
            footprint = _dilate(footprint, self.pad_cells)
        # Bilinear-sample the *binary* footprint mask, not the raw class ids --
        # interpolating class ids turns any nonzero class into "building".
        self._footprint_field = Field(
            data=footprint.astype(np.float64), transform=self.semantic.transform, nodata=0.0
        )

    def z(self, east: np.ndarray, north: np.ndarray) -> np.ndarray:
        ground = self.dem.sample(east, north)
        is_building = self._footprint_field.sample(east, north) > 0.5
        z = np.where(is_building, ground + self.building_height, ground)
        # Where the DEM has no coverage but the footprint is a building, still
        # report the building top (never fabricate a ground height).
        z = np.where(np.isnan(z) & is_building, self.building_height, z)
        return z


def _dilate(mask: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return mask
    out = mask.copy()
    h, w = mask.shape
    for _ in range(k):
        prev = out
        out = prev.copy()
        out[1:, :] |= prev[:-1, :]
        out[:-1, :] |= prev[1:, :]
        out[:, 1:] |= prev[:, :-1]
        out[:, :-1] |= prev[:, 1:]
    return out


def terrain_z(terrain, east: np.ndarray, north: np.ndarray) -> np.ndarray:
    """Uniform accessor: a terrain object (flat / DEM / synthetic)."""
    e = np.asarray(east, dtype=np.float64)
    n = np.asarray(north, dtype=np.float64)
    return np.asarray(terrain.z(e, n), dtype=np.float64)
