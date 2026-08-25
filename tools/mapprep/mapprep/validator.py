"""Cross-layer consistency validator for a built geopack (T06-U-03).

The manifest claims a single CRS and a single corridor bounds; every raster
layer must agree. A mismatch (a layer in another CRS, a layer that does not
cover the corridor, a GSD that disagrees with the file) is rejected with a
message that names the layer and the two disagreeing values, so a pre-flight
check (T27) can fail loudly rather than silently mix layers.

This is stricter than verify.py's geotransform self-check: it validates the
package as a whole against the manifest, not a layer against its own grid.
"""

from __future__ import annotations

from pathlib import Path

import rasterio

_GSD_TOL = 1e-6


def _manifest_epsg(manifest: dict) -> int | None:
    crs = manifest.get("crs", "")
    if not crs.startswith("EPSG:"):
        return None
    try:
        return int(crs.split(":")[1])
    except (IndexError, ValueError):
        return None


def validate_package(manifest: dict, geopack_dir: Path) -> list[str]:
    """Return a list of problems; an empty list means the package is consistent."""
    problems: list[str] = []
    bounds = manifest["bounds"]
    expected_epsg = _manifest_epsg(manifest)

    for name, layer in manifest["layers"].items():
        file = layer.get("file")
        if not file:
            problems.append(f"{name}: layer has no file")
            continue
        path = geopack_dir / file
        if not path.exists():
            problems.append(f"{name}: missing file {file}")
            continue
        try:
            ds = rasterio.open(path)
        except rasterio.errors.RasterioError as exc:
            problems.append(f"{name}: cannot open {file}: {exc}")
            continue
        with ds:
            epsg = ds.crs.to_epsg() if ds.crs else None
            if expected_epsg is not None and epsg != expected_epsg:
                problems.append(
                    f"{name}: CRS {ds.crs.to_string() if ds.crs else None} does not "
                    f"match manifest CRS EPSG:{expected_epsg}"
                )
            gsd = layer.get("gsd")
            if gsd is not None:
                for axis, value in (("gsd", ds.transform.a), ("gsd_row", -ds.transform.e)):
                    if abs(gsd - value) > _GSD_TOL * max(1.0, abs(gsd)):
                        problems.append(
                            f"{name}: {axis} differs from file: manifest {gsd}, file {value}"
                        )
            tolerance = 2.0 * (gsd if gsd else 0.0)
            if (
                ds.bounds.left - tolerance > bounds["east_min"]
                or ds.bounds.right + tolerance < bounds["east_max"]
                or ds.bounds.bottom - tolerance > bounds["north_min"]
                or ds.bounds.top + tolerance < bounds["north_max"]
            ):
                problems.append(
                    f"{name}: layer bounds {ds.bounds} do not cover manifest corridor "
                    f"bounds within {tolerance:.3f} m"
                )
    return problems
