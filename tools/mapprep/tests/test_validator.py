"""T06-U-03: the package validator rejects an inconsistent package with a clear message."""

import numpy as np
import rasterio
from affine import Affine

from mapprep.validator import validate_package

CRS = "EPSG:32637"
BOUNDS = {
    "east_min": 572000.0,
    "east_max": 572100.0,
    "north_min": 4972000.0,
    "north_max": 4972100.0,
}


def _write_tiff(path, crs, bounds, gsd, dtype="uint8"):
    width = int(round((bounds["east_max"] - bounds["east_min"]) / gsd))
    height = int(round((bounds["north_max"] - bounds["north_min"]) / gsd))
    transform = Affine(gsd, 0, bounds["east_min"], 0, -gsd, bounds["north_max"])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as ds:
        ds.write(np.zeros((1, height, width), dtype=dtype))
    return path


def _manifest():
    return {"crs": CRS, "bounds": dict(BOUNDS), "layers": {}}


def test_consistent_package_passes(tmp_path):
    manifest = _manifest()
    manifest["layers"]["ortho_a"] = {
        "file": "ortho_a.tif",
        "gsd": 0.5,
    }
    manifest["layers"]["semantic"] = {
        "file": "semantic.tif",
        "gsd": 1.0,
    }
    _write_tiff(tmp_path / "ortho_a.tif", CRS, BOUNDS, 0.5)
    _write_tiff(tmp_path / "semantic.tif", CRS, BOUNDS, 1.0)
    assert validate_package(manifest, tmp_path) == []


def test_crs_mismatch_rejected_with_message(tmp_path):
    manifest = _manifest()
    manifest["layers"]["dem"] = {"file": "dem.tif", "gsd": 10.0}
    _write_tiff(tmp_path / "dem.tif", "EPSG:4326", BOUNDS, 10.0, dtype="float32")
    problems = validate_package(manifest, tmp_path)
    assert any("CRS" in problem for problem in problems)


def test_bounds_mismatch_rejected_with_message(tmp_path):
    manifest = _manifest()
    manifest["layers"]["semantic"] = {"file": "semantic.tif", "gsd": 1.0}
    tiny = {
        "east_min": 572000.0,
        "east_max": 572010.0,
        "north_min": 4972000.0,
        "north_max": 4972010.0,
    }
    _write_tiff(tmp_path / "semantic.tif", CRS, tiny, 1.0)
    problems = validate_package(manifest, tmp_path)
    assert any("cover" in problem for problem in problems)


def test_missing_file_rejected_with_message(tmp_path):
    manifest = _manifest()
    manifest["layers"]["dem"] = {"file": "dem.tif", "gsd": 10.0}
    problems = validate_package(manifest, tmp_path)
    assert any("missing file" in problem for problem in problems)


def test_gsd_mismatch_rejected_with_message(tmp_path):
    manifest = _manifest()
    manifest["layers"]["semantic"] = {"file": "semantic.tif", "gsd": 1.0}
    _write_tiff(tmp_path / "semantic.tif", CRS, BOUNDS, 2.0)
    problems = validate_package(manifest, tmp_path)
    assert any("gsd" in problem for problem in problems)
