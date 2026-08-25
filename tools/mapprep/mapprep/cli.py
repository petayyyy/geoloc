"""Command line interface: build, import-cache, verify, inspect."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import dates as dates_mod
from .config import cache_root as config_cache_root
from .config import load_config, overviews
from .fetch import force_ipv4
from .manifest import add_ortho_layer, new_manifest, read_manifest, write_manifest
from .mosaic import build_layer, select_zoom
from .providers import get_provider
from .tilecache import import_cache
from .verify import (
    check_pyramid,
    cross_provider_offset,
    sample_tile_corner_gcps,
    verify_geotransform,
    write_gcps,
)


def _build(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    cache_root = config_cache_root(config)
    geopack_dir = Path(args.out)
    enrich_dates = bool(config.get("enrich_capture_dates", False))
    manifest = new_manifest(config["mission_id"], config["crs"], {}, config["origin"])
    bounds_wgs84 = config["bounds"]

    with force_ipv4():
        for layer_cfg in config["layers"]:
            provider = get_provider(layer_cfg["provider"])
            zoom = select_zoom(
                provider,
                bounds_wgs84,
                cache_root,
                layer_cfg.get("min_zoom", provider.min_zoom),
                layer_cfg.get("max_zoom", provider.max_zoom),
                args.offline,
            )
            print(
                f"[{layer_cfg['name']}] {provider.name}: source zoom {zoom}, "
                f"target gsd {layer_cfg['target_gsd_m']} m/px"
            )
            capture_date = None
            notes_extra = ""
            if enrich_dates:
                print(f"[{layer_cfg['name']}] probing capture dates ...")
                capture_date = dates_mod.enrich_capture_dates(
                    provider, bounds_wgs84, zoom, cache_root
                )
                if capture_date is None and provider.id == "esri_world_imagery":
                    release = dates_mod.esri_latest_release()
                    if release:
                        notes_extra = (
                            "provider release at build time: "
                            f"{release} (per-tile dates not exposed)"
                        )
                print(f"[{layer_cfg['name']}] capture date: {capture_date}")
            result = build_layer(
                provider,
                bounds_wgs84,
                layer_cfg["target_gsd_m"],
                zoom,
                cache_root,
                geopack_dir / f"{layer_cfg['name']}.tif",
                geopack_dir / f"validity_{layer_cfg['name'].split('_')[1]}.tif",
                config["crs"],
                offline=args.offline,
                overviews=overviews(config),
                enrich_dates=enrich_dates,
                cloud_polygons=_load_cloud_polygons(layer_cfg, Path(args.config).parent),
            )
            print(
                f"[{layer_cfg['name']}] mosaic {result.grid.width}x{result.grid.height} "
                f"@ {result.mosaic_gsd_m:.4f} m/px; tiles {result.tiles_fetched}/"
                f"{result.tiles_expected}; missing {len(result.missing_tiles)}; "
                f"seams marked {result.seam_count}"
            )
            add_ortho_layer(
                manifest,
                layer_cfg["name"],
                result,
                provider,
                bounds_wgs84,
                config["crs"],
                capture_date,
                notes_extra,
            )

    manifest["bounds"] = _manifest_bounds(manifest, geopack_dir)
    write_manifest(manifest, geopack_dir / "manifest.yaml")

    problems = verify_geotransform(manifest, geopack_dir)
    for problem in problems:
        print(f"VERIFY FAIL: {problem}")
    if problems:
        return 1

    gcps = sample_tile_corner_gcps(manifest, geopack_dir)
    write_gcps(gcps, geopack_dir / "gcp.csv")
    worst = max((g.error_px for g in gcps), default=0.0)
    print(f"georef self-check: {len(gcps)} tile-corner GCPs, max round-trip error {worst:.3e} px")

    cross = cross_provider_offset(manifest, geopack_dir)
    if not cross.get("skipped"):
        reliability = "" if cross.get("reliable") else " (low NCC, unreliable)"
        print(
            f"cross-provider NCC shift: {cross['shift_px']} px "
            f"({cross['offset_m']:.2f} m) between {cross['layers']}, "
            f"ncc {cross['ncc_score']:.3f} over {cross['windows']} windows{reliability}"
        )
    print(f"geopack written to {geopack_dir}")
    return 0


def _load_cloud_polygons(layer_cfg: dict, config_dir: Path):
    path = layer_cfg.get("cloud_polygons")
    if not path:
        return None
    from .mask import load_geojson_polygons

    return load_geojson_polygons(config_dir / path)


def _manifest_bounds(manifest: dict, geopack_dir: Path) -> dict:
    first = next(
        (layer for layer in manifest["layers"].values() if layer["file"].endswith(".tif")),
        None,
    )
    if first is None:
        raise RuntimeError("no ortho layers built")
    import rasterio

    with rasterio.open(geopack_dir / first["file"]) as ds:
        return {
            "east_min": ds.bounds.left,
            "east_max": ds.bounds.right,
            "north_min": ds.bounds.bottom,
            "north_max": ds.bounds.top,
        }


def _import_cache(args: argparse.Namespace) -> int:
    provider = get_provider(args.provider)
    cache_root = Path(args.cache).expanduser()
    count = import_cache(
        Path(args.src),
        provider,
        cache_root,
        layout=args.layout,
        z_offset=args.z_offset,
    )
    print(f"imported {count} tiles from {args.src} into {cache_root / provider.id}")
    return 0


def _verify(args: argparse.Namespace) -> int:
    geopack_dir = Path(args.geopack)
    manifest = read_manifest(geopack_dir / "manifest.yaml")
    problems = verify_geotransform(manifest, geopack_dir)
    for problem in problems:
        print(f"FAIL: {problem}")
    for layer_name, layer in manifest["layers"].items():
        if not layer_name.startswith("ortho_"):
            continue
        for _suffix, path in [("ortho", layer["file"]), ("mask", layer.get("validity_file"))]:
            if not path:
                continue
            pyramid = check_pyramid(geopack_dir / path)
            for problem in pyramid["problems"]:
                print(f"FAIL: {path}: {problem}")
            print(f"{path}: overviews {pyramid['overviews']}")
    gcps = sample_tile_corner_gcps(manifest, geopack_dir)
    worst = max((g.error_px for g in gcps), default=0.0)
    print(f"tile-corner GCPs: {len(gcps)}, max round-trip error {worst:.3e} px")
    if args.cross:
        cross = cross_provider_offset(manifest, geopack_dir)
        if cross.get("skipped"):
            print(f"cross-provider check skipped: {cross['reason']}")
        else:
            reliability = "" if cross.get("reliable") else " (low NCC, unreliable)"
            print(
                f"cross-provider NCC shift {cross['shift_px']} px "
                f"({cross['offset_m']:.2f} m), ncc {cross['ncc_score']:.3f} "
                f"over {cross['windows']} windows{reliability}"
            )
    return 1 if problems else 0


def _inspect(args: argparse.Namespace) -> int:
    manifest = read_manifest(Path(args.geopack) / "manifest.yaml")
    for key in ("mission_id", "crs", "bounds", "origin"):
        print(f"{key}: {manifest[key]}")
    for name, layer in manifest["layers"].items():
        print(
            f"{name}: {layer.get('provider')} gsd={layer.get('gsd')} "
            f"date={layer.get('capture_date')} file={layer.get('file')}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mapprep", description="geoloc T05 basemap tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="assemble a geopack for a corridor")
    p_build.add_argument("--config", required=True, help="corridor YAML config")
    p_build.add_argument("--out", required=True, help="output geopack directory")
    p_build.add_argument("--offline", action="store_true", help="never touch the network")
    p_build.set_defaults(func=_build)

    p_import = sub.add_parser("import-cache", help="import a pre-downloaded tile cache")
    p_import.add_argument("--src", required=True, help="source cache directory")
    p_import.add_argument("--provider", required=True, help="provider id")
    p_import.add_argument("--cache", required=True, help="normalized cache root")
    p_import.add_argument("--layout", choices=["sasplanet", "qgis_xyz"], default="sasplanet")
    p_import.add_argument("--z-offset", type=int, default=None)
    p_import.set_defaults(func=_import_cache)

    p_verify = sub.add_parser("verify", help="verify a built geopack")
    p_verify.add_argument("--geopack", required=True)
    p_verify.add_argument("--cross", action="store_true", help="cross-provider NCC shift")
    p_verify.set_defaults(func=_verify)

    p_inspect = sub.add_parser("inspect", help="print the manifest")
    p_inspect.add_argument("--geopack", required=True)
    p_inspect.set_defaults(func=_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
