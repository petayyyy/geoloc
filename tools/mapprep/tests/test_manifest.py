"""T05-U-04: manifest round-trip and schema validation."""

import jsonschema
import pytest
import yaml

from mapprep.manifest import (
    GEOREF_BIAS_PLACEHOLDER,
    ManifestValidationError,
    new_manifest,
    read_manifest,
    validate_manifest,
    write_manifest,
)


def _sample_manifest():
    manifest = new_manifest(
        "maykop-corridor-2026-08",
        "EPSG:32637",
        {
            "east_min": 572000.0,
            "east_max": 572600.0,
            "north_min": 4972000.0,
            "north_max": 4973400.0,
        },
        {"lat": 44.8285, "lon": 39.922, "alt": 80.0},
    )
    manifest["layers"]["ortho_a"] = {
        "file": "ortho_a.tif",
        "validity_file": "validity_a.tif",
        "provider": "esri_world_imagery",
        "capture_date": "2024-06",
        "gsd": 0.3,
        "native_gsd": 0.2118,
        "source_zoom": 19,
        "crs": "EPSG:32637",
        "tiles": {"expected": 338, "fetched": 338, "missing": 0},
        "georef_bias": dict(GEOREF_BIAS_PLACEHOLDER),
        "license": "internal dev only",
        "attribution": "Esri",
    }
    return manifest


def test_round_trip_preserves_content(tmp_path):
    manifest = _sample_manifest()
    path = tmp_path / "manifest.yaml"
    write_manifest(manifest, path)
    loaded = read_manifest(path)
    assert loaded == manifest


def test_georef_bias_placeholder_is_null(tmp_path):
    manifest = _sample_manifest()
    path = tmp_path / "manifest.yaml"
    write_manifest(manifest, path)
    loaded = read_manifest(path)
    bias = loaded["layers"]["ortho_a"]["georef_bias"]
    assert bias["east"] is None and bias["north"] is None and bias["sigma"] is None


def test_null_capture_date_is_honest(tmp_path):
    manifest = _sample_manifest()
    manifest["layers"]["ortho_a"]["capture_date"] = None
    path = tmp_path / "manifest.yaml"
    write_manifest(manifest, path)
    assert read_manifest(path) == manifest


def test_invalid_manifest_rejected(tmp_path):
    manifest = _sample_manifest()
    del manifest["mission_id"]
    with pytest.raises(ManifestValidationError):
        validate_manifest(manifest)


def test_invalid_layer_rejected(tmp_path):
    manifest = _sample_manifest()
    del manifest["layers"]["ortho_a"]["provider"]
    with pytest.raises(ManifestValidationError):
        validate_manifest(manifest)


def test_schema_rejects_unknown_top_level_keys():
    manifest = _sample_manifest()
    manifest["surprise"] = True
    with pytest.raises(jsonschema.ValidationError):
        validate_manifest(manifest)


def test_georef_bias_with_measured_values_valid():
    manifest = _sample_manifest()
    manifest["layers"]["ortho_a"]["georef_bias"] = {
        "east": 2.1,
        "north": -1.4,
        "sigma": 3.0,
    }
    validate_manifest(manifest)


def test_written_yaml_loads_with_strict_parser(tmp_path):
    manifest = _sample_manifest()
    path = tmp_path / "manifest.yaml"
    write_manifest(manifest, path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    assert raw["version"] == 1
    assert set(raw["bounds"]) == {"east_min", "east_max", "north_min", "north_max"}
