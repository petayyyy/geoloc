"""Geopack manifest: write, read, validate.

The schema follows docs/plan/03-interfaces.md section 4. Extensions used by
T05 (and flagged in the plan): an optional per-layer `validity_file`, `license`,
`attribution` and `native_gsd`; `georef_bias` is a placeholder with null values
until T09 measures it. No commercial-use claims are ever written here while
ADR-008 stays open.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml

_SCHEMA_PATH = Path(__file__).parent / "schema" / "manifest.schema.json"

GEOREF_BIAS_PLACEHOLDER = {"east": None, "north": None, "sigma": None}


class ManifestValidationError(jsonschema.ValidationError):
    pass


def _load_schema() -> dict:
    with _SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_manifest(manifest: dict[str, Any]) -> None:
    try:
        jsonschema.validate(manifest, _load_schema())
    except jsonschema.ValidationError as exc:
        raise ManifestValidationError(exc.message) from exc


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    validate_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, default_flow_style=False)


def read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    validate_manifest(manifest)
    return manifest


def new_manifest(
    mission_id: str,
    crs: str,
    bounds: dict[str, float],
    origin: dict[str, float],
) -> dict[str, Any]:
    return {
        "version": 1,
        "mission_id": mission_id,
        "crs": crs,
        "bounds": dict(bounds),
        "origin": dict(origin),
        "layers": {},
    }


def add_ortho_layer(
    manifest: dict[str, Any],
    layer_name: str,
    result,
    provider,
    bounds_wgs84: dict,
    crs: str,
    capture_date: str | None = None,
    notes_extra: str = "",
) -> None:
    grid = result.grid
    manifest["layers"][layer_name] = {
        "file": result.ortho_path.name,
        "validity_file": result.mask_path.name,
        "provider": provider.id,
        "capture_date": capture_date,
        "gsd": result.mosaic_gsd_m,
        "native_gsd": result.native_gsd_m,
        "source_zoom": result.zoom,
        "crs": crs,
        "tiles": {
            "expected": result.tiles_expected,
            "fetched": result.tiles_fetched,
            "missing": len(result.missing_tiles),
        },
        "bounds_wgs84": dict(bounds_wgs84),
        "georef_bias": dict(GEOREF_BIAS_PLACEHOLDER),
        "license": provider.license,
        "attribution": provider.attribution,
        "notes": (
            f"mosaic grid origin ({grid.origin_east:.3f}, {grid.origin_north:.3f}) m, "
            f"{grid.width}x{grid.height} px"
            + ("; capture dates unknown for this provider" if capture_date is None else "")
            + ("; upsampling refused, mosaic at native GSD" if result.upsampling_refused else "")
            + (f"; {notes_extra}" if notes_extra else "")
        ),
    }
