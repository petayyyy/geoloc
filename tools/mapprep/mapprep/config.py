"""Corridor build configuration (YAML).

Every threshold and gate here is configuration, not a code constant
(plan/prompts/P0-common.md rule 6). See configs/mapprep/maykop_corridor.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REQUIRED_TOP = {"mission_id", "crs", "bounds", "origin", "layers"}
_REQUIRED_BOUNDS = {"west", "south", "east", "north"}
_REQUIRED_ORIGIN = {"lat", "lon", "alt"}
_REQUIRED_LAYER = {"name", "provider", "target_gsd_m"}
_REQUIRED_DEM = {"target_gsd_m"}
_REQUIRED_SEMANTIC = {"gsd"}


class ConfigError(Exception):
    pass


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    _validate(config)
    return config


def _validate(config: dict[str, Any]) -> None:
    missing = _REQUIRED_TOP - set(config)
    if missing:
        raise ConfigError(f"missing top-level keys: {sorted(missing)}")
    missing = _REQUIRED_BOUNDS - set(config["bounds"])
    if missing:
        raise ConfigError(f"bounds missing keys: {sorted(missing)}")
    missing = _REQUIRED_ORIGIN - set(config["origin"])
    if missing:
        raise ConfigError(f"origin missing keys: {sorted(missing)}")
    west, east = config["bounds"]["west"], config["bounds"]["east"]
    south, north = config["bounds"]["south"], config["bounds"]["north"]
    if not (west < east and south < north):
        raise ConfigError("bounds must satisfy west < east, south < north")
    for layer in config["layers"]:
        missing = _REQUIRED_LAYER - set(layer)
        if missing:
            raise ConfigError(f"layer missing keys: {sorted(missing)}")
        if layer["target_gsd_m"] <= 0:
            raise ConfigError("target_gsd_m must be > 0")
        if not layer["name"].startswith("ortho_"):
            raise ConfigError("layer name must start with 'ortho_'")
    if len({layer["name"] for layer in config["layers"]}) != len(config["layers"]):
        raise ConfigError("duplicate layer names")
    if "dem" in config:
        missing = _REQUIRED_DEM - set(config["dem"])
        if missing:
            raise ConfigError(f"dem missing keys: {sorted(missing)}")
        if config["dem"]["target_gsd_m"] <= 0:
            raise ConfigError("dem.target_gsd_m must be > 0")
        from .dem import DEM_SOURCES

        for key in ("source",):
            value = config["dem"].get(key)
            if value is not None and value not in DEM_SOURCES:
                raise ConfigError(f"dem.{key} must be one of {sorted(DEM_SOURCES)}, got {value!r}")
        fallbacks = config["dem"].get("fallback_sources", [])
        if not isinstance(fallbacks, list):
            raise ConfigError("dem.fallback_sources must be a list")
        for value in fallbacks:
            if value not in DEM_SOURCES:
                raise ConfigError(
                    f"dem.fallback_sources entries must be one of {sorted(DEM_SOURCES)}, "
                    f"got {value!r}"
                )
    if "semantic" in config:
        missing = _REQUIRED_SEMANTIC - set(config["semantic"])
        if missing:
            raise ConfigError(f"semantic missing keys: {sorted(missing)}")
        if config["semantic"]["gsd"] <= 0:
            raise ConfigError("semantic.gsd must be > 0")


def cache_root(config: dict[str, Any]) -> Path:
    raw = config.get("cache", {}).get("root", "~/.cache/geoloc/tiles")
    return Path(raw).expanduser()


def overviews(config: dict[str, Any]) -> tuple[int, ...]:
    return tuple(config.get("mosaic", {}).get("overviews", [2, 4, 8]))
