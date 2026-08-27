"""Scene: the static map side of an OrthoSim pair.

Ties the two ortho providers (A = map, B = query), the terrain height field and
the optional OSM semantic raster together. Cross-provider enforcement lives in
the generator (render.py), not here -- the scene simply exposes both providers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dsm import BuildingTerrain, DemTerrain, FlatTerrain
from .geopack import (
    CLASS_BUILDING,
    CLASS_FARMLAND,
    CLASS_FOREST,
    CLASS_ROAD,
    CLASS_WATER,
    Field,
    GeoPack,
)


@dataclass
class Scene:
    ortho_a: Field
    ortho_b: Field
    terrain: object
    semantic: Field | None = None
    crs: str = "EPSG:32637"

    def sample(self, provider: str, east: np.ndarray, north: np.ndarray) -> np.ndarray:
        field = self.ortho_a if provider == "a" else self.ortho_b
        return field.sample(east, north)

    def z(self, east: np.ndarray, north: np.ndarray) -> np.ndarray:
        return np.asarray(self.terrain.z(np.asarray(east), np.asarray(north)), dtype=np.float64)


def build_scene(
    geopack: GeoPack, terrain_mode: str = "dem", building_height: float = 15.0
) -> Scene:
    """Assemble a Scene from a geopack directory + a terrain mode.

    ``terrain_mode`` is one of ``flat``, ``dem`` (v1) or ``buildings`` (v2 --
    DEM + OSM building extrusion).
    """
    ortho_a = geopack.open_ortho("ortho_a")
    ortho_b = geopack.open_ortho("ortho_b")
    semantic = geopack.open_semantic()

    if terrain_mode == "flat":
        terrain = FlatTerrain(height=0.0)
    elif terrain_mode == "buildings":
        if semantic is None:
            raise ValueError("terrain_mode=buildings requires a semantic layer")
        terrain = BuildingTerrain(geopack.open_dem(), semantic, building_height=building_height)
    else:
        terrain = DemTerrain(geopack.open_dem())

    return Scene(
        ortho_a=ortho_a, ortho_b=ortho_b, terrain=terrain, semantic=semantic, crs=geopack.crs
    )


def classify(semantic_field: Field | None, east: float, north: float, size_m: float) -> str:
    """Terrain class at a patch centre, using the 05-metrics.md §5 rules.

    When no semantic layer is present every patch is ``background`` (an honest
    "unknown", matching rule 3 -- never fabricate a class).
    """
    if semantic_field is None:
        return "background"
    half = size_m / 2.0
    gsd = semantic_field.gsd
    c0 = int(np.floor((east - half - semantic_field.east_min) / gsd))
    r0 = int(np.floor((semantic_field.north_max - (north + half)) / gsd))
    c1 = int(np.ceil((east + half - semantic_field.east_min) / gsd))
    r1 = int(np.ceil((semantic_field.north_max - (north - half)) / gsd))
    h, w = semantic_field.data.shape[:2]
    c0, r0 = max(0, c0), max(0, r0)
    c1, r1 = min(w, c1), min(h, r1)
    if c1 <= c0 or r1 <= r0:
        return "background"
    window = semantic_field.data[r0:r1, c0:c1]
    total = window.size
    if total == 0:
        return "background"
    counts = np.bincount(window.ravel(), minlength=6)
    f = counts / total
    building = f[CLASS_BUILDING]
    road = f[CLASS_ROAD]
    water = f[CLASS_WATER]
    farmland = f[CLASS_FARMLAND]
    forest = f[CLASS_FOREST]
    if water > 0.50:
        return "water"
    if building > 0.15:
        return "urban"
    if farmland > 0.60:
        return "farmland"
    if forest > 0.60:
        return "forest"
    if road > 0.05 and building < 0.03:
        return "roads"
    if building >= 0.03:
        return "suburban"
    return "background"
