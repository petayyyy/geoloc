"""Geopack manifest: write, read, validate.

The schema follows docs/plan/03-interfaces.md section 4. Extensions used by
T05/T06 (and flagged in the plan): an optional per-layer `validity_file`,
`license`, `attribution` and `native_gsd`; `georef_bias` is a placeholder with
null values until T09 measures it; `dem`/`semantic` layers carry their source,
vertical datum and class table. No commercial-use claims are ever written here
while ADR-008 stays open.

JSON Schema checks the *shape* of every layer; `validate_manifest` additionally
enforces the per-type required fields (ortho vs dem vs semantic) which a single
shared `$defs` cannot express without `oneOf` ambiguity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from .osm import CLASS_NAMES

_SCHEMA_PATH = Path(__file__).parent / "schema" / "manifest.schema.json"

GEOREF_BIAS_PLACEHOLDER = {"east": None, "north": None, "sigma": None}

_ORTHO_REQUIRED = {"provider"}
_DEM_REQUIRED = {"source", "source_datum", "vertical_datum", "geoid_model"}
_SEMANTIC_REQUIRED = {"source", "extract_date", "classes"}


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
    for name, layer in manifest.get("layers", {}).items():
        if name.startswith("ortho_"):
            required = _ORTHO_REQUIRED
        elif name == "dem":
            required = _DEM_REQUIRED
        elif name == "semantic":
            required = _SEMANTIC_REQUIRED
        else:
            raise ManifestValidationError(f"unknown layer name {name!r}")
        missing = required - set(layer)
        if missing:
            raise ManifestValidationError(
                f"layer {name!r} missing required keys: {sorted(missing)}"
            )
        if name == "semantic" and tuple(layer.get("classes", [])) != tuple(CLASS_NAMES):
            raise ManifestValidationError(
                f"layer 'semantic' classes must equal {list(CLASS_NAMES)}"
            )


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


def add_dem_layer(
    manifest: dict[str, Any],
    result,
    crs: str,
    *,
    bounds_wgs84: dict | None = None,
    license: str = "",
    attribution: str = "",
) -> None:
    grid = result.grid
    layer = {
        "file": result.dem_path.name,
        "source": "+".join(result.source_datasets) if result.source_datasets else "unknown",
        "gsd": result.target_gsd_m,
        "native_gsd": result.native_gsd_m,
        "crs": crs,
        "source_datum": result.source_datum,
        "vertical_datum": result.vertical_datum,
        "geoid_model": result.geoid_model,
        "source_tiles": result.source_tiles,
        "resampling": "bilinear",
        "georef_bias": dict(GEOREF_BIAS_PLACEHOLDER),
        "notes": (
            f"Copernicus DSM (buildings/vegetation included, not a DTM) from "
            f"{'+'.join(result.source_datasets) or 'unknown'}; {grid.width}x{grid.height} px "
            f"@ {result.target_gsd_m} m grid; native ~{result.native_gsd_m} m interpolated "
            f"to {result.target_gsd_m} m; vertical datum {result.source_datum} -> "
            f"{result.vertical_datum} via {result.geoid_model}; coverage "
            f"{result.valid_ratio:.1%}"
        ),
    }
    if bounds_wgs84:
        layer["bounds_wgs84"] = dict(bounds_wgs84)
    if license:
        layer["license"] = license
    if attribution:
        layer["attribution"] = attribution
    manifest["layers"]["dem"] = layer


def add_semantic_layer(
    manifest: dict[str, Any],
    result,
    crs: str,
    *,
    extract_date: str | None,
    source: str = "osm",
) -> None:
    grid = result.grid
    manifest["layers"]["semantic"] = {
        "file": result.semantic_path.name,
        "source": source,
        "extract_date": extract_date,
        "gsd": result.gsd_m,
        "crs": crs,
        "classes": list(CLASS_NAMES),
        "georef_bias": dict(GEOREF_BIAS_PLACEHOLDER),
        "notes": (
            f"{grid.width}x{grid.height} px class raster @ {result.gsd_m} m/px; "
            "road/water are stable classes, farmland varies by season; "
            "OSM coverage is uneven and must be treated as a hint, not truth"
        ),
    }
