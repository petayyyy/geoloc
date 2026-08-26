"""Geopack access: manifest + georeferenced layers (ortho, DEM, validity)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import rasterio
import yaml


@dataclass
class GeoPack:
    dir: Path

    @property
    def manifest(self) -> dict:
        with open(self.dir / "manifest.yaml", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def layer_path(self, name: str) -> Path:
        return self.dir / self.manifest["layers"][name]["file"]

    def sample_dem(self, east: float, north: float) -> float:
        with rasterio.open(self.layer_path("dem")) as src:
            return float(src.sample([(east, north)]).__next__()[0])
