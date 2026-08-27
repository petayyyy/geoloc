"""Command line interface: check | align | dsm | ortho | run."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from .align import (
    AlignmentResult,
    align_pose_graph,
    align_windowed,
    orient_up_from_cloud,
    pca_up_axis,
)
from .bagio import Capture
from .camera import Pinhole
from .dsm import DsmGrid, run_dsm
from .geo import GeoRef
from .ortho import DemField, run_ortho
from .rtkcheck import doppler_consistency, velocity_en


def load_config(path: Path, base: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["bag_dir"] = str((base / cfg["bag_dir"]).resolve())
    cfg["geopack_dir"] = str((base / cfg["geopack_dir"]).resolve())
    cfg["out_dir"] = str((base / cfg["out_dir"]).resolve())
    return cfg


def _cmd_check(cfg: dict) -> int:
    """Georeference self-check: projected RTK track vs its own Doppler velocity.

    Runs before anything else in `run`, because every downstream metre --
    the DSM, the ortho patch, the "odometry scale" -- inherits whatever
    geographic interpretation the fixes were read with.
    """
    cap = Capture(Path(cfg["bag_dir"]))
    geo = GeoRef.from_epsg(cfg["crs"])
    rtk_cfg = cfg.get("rtk", {})
    rtk = cap.read_rtk(bool(rtk_cfg.get("swap_latlon", False)))
    vel = cap.read_rtk_velocity()
    if len(vel) == 0:
        print("check: no rtk_velocity in this capture, skipping", file=sys.stderr)
        return 0
    utm = geo.lonlat_to_utm_many(rtk[:, 2], rtk[:, 1])
    v_en = velocity_en(vel[:, 1:4], str(rtk_cfg.get("velocity_frame", "ned")))
    v_at_fix = np.column_stack([np.interp(rtk[:, 0], vel[:, 0], v_en[:, k]) for k in range(2)])
    res = doppler_consistency(
        rtk[:, 0],
        utm,
        v_at_fix,
        baseline_s=float(rtk_cfg.get("doppler_baseline_s", 3.0)),
    )
    tol = float(rtk_cfg.get("doppler_ratio_tol", 0.02))
    print(
        f"check: doppler/position ratio={res.ratio:.4f} "
        f"(min {res.ratio_min:.4f} max {res.ratio_max:.4f}, n={res.n}, "
        f"baseline={res.baseline_s:.1f}s, tol={tol:.3f})"
    )
    if not res.ok(tol):
        print(
            "check: FAILED -- the projected RTK track and the receiver's own "
            "Doppler velocity disagree on distance. The usual cause is a wrong "
            "lat/lon interpretation (try flipping rtk.swap_latlon and re-running); "
            "a wrong `crs` or velocity_frame does it too. Do NOT absorb this into "
            "an alignment scale -- it is a georeference error, not odometry.",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_align(cfg: dict) -> int:
    cap = Capture(Path(cfg["bag_dir"]))
    geo = GeoRef.from_epsg(cfg["crs"])
    t0 = time.time()
    rtk = cap.read_rtk(bool(cfg["rtk"].get("swap_latlon", False)))
    odom = cap.read_odom()
    utm = geo.lonlat_to_utm_many(rtk[:, 2], rtk[:, 1])  # (lon, lat) after the swap
    rtk_utm = np.column_stack([rtk[:, 0], utm[:, 0], utm[:, 1], rtk[:, 3]])
    # vertical sign: PCA of the path gives the vertical axis but not its sign;
    # the lidar cloud lies below the drone, which resolves it
    up_pca = pca_up_axis(odom)
    up_odom = orient_up_from_cloud(odom, cap.iter_clouds(), up_pca)
    method = cfg["align"].get("method", "pose_graph")
    if method == "pose_graph":
        res = align_pose_graph(
            odom,
            rtk_utm,
            up_odom=up_odom,
            step_s=float(cfg["align"].get("step_s", 2.0)),
            rtk_weight=float(cfg["align"].get("rtk_weight", 1.0)),
            smooth_weight=float(cfg["align"].get("smooth_weight", 0.5)),
            rtk_huber_m=float(cfg["align"].get("rtk_huber_m", 2.0)),
            max_iter=int(cfg["align"].get("max_iter", 200)),
            estimate_scale=bool(cfg["align"].get("estimate_scale", False)),
        )
    elif method == "windowed":
        res = align_windowed(
            odom,
            rtk_utm,
            up_odom=up_odom,
            window_s=float(cfg["align"].get("window_s", 12.0)),
            step_s=float(cfg["align"].get("window_step_s", 4.0)),
            min_pairs=int(cfg["align"].get("min_pairs", 8)),
        )
    else:
        print(
            f"align: unknown method {method!r} (expected 'pose_graph' or 'windowed')",
            file=sys.stderr,
        )
        return 2
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    res.save(out / "align.yaml")
    print(
        f"align: {method}, {len(res.series.times)} nodes, "
        f"scale={res.series.scale:.4f}, "
        f"residual mean={res.residuals.mean():.2f} m max={res.residuals.max():.2f} m "
        f"({time.time() - t0:.1f}s)"
    )
    return 0


def _cmd_dsm(cfg: dict) -> int:
    cap = Capture(Path(cfg["bag_dir"]))
    out = Path(cfg["out_dir"])
    align = AlignmentResult.load(out / "align.yaml")
    odom = cap.read_odom()
    t0 = time.time()
    grid, align = run_dsm(
        align,
        odom,
        cap.iter_clouds(),
        cfg["dsm"],
        out,
        Path(cfg["geopack_dir"]) / "dem.tif",
        cfg["crs"],
    )
    print(
        f"dsm: {grid.width}x{grid.height} @ {grid.gsd} m, "
        f"coverage={np.mean(grid.count > 0) * 100:.1f}%, "
        f"z_shift={align.z_shift:+.2f} m ({time.time() - t0:.1f}s)"
    )
    return 0


def _cmd_ortho(cfg: dict) -> int:
    cap = Capture(Path(cfg["bag_dir"]))
    out = Path(cfg["out_dir"])
    align = AlignmentResult.load(out / "align_final.yaml")
    dsm = DsmGrid.load(out / "dsm.yaml", out / "dsm.tif", out / "dsm_confidence.tif")
    dem = DemField.open(Path(cfg["geopack_dir"]) / "dem.tif")
    cam = Pinhole.from_config(cfg["camera"], float(cfg["camera"].get("scale", 1.0)))
    odom = cap.read_odom()
    stamps, images = cap.read_images(out / "cache_images.npz")
    Rcl = cfg["camera"].get("Rcl")
    R_lidar_to_cam = np.array(Rcl, dtype=np.float64).reshape(3, 3) if Rcl else None
    t0 = time.time()
    stats = run_ortho(
        align, odom, stamps, images, cam, dsm, dem, cfg["ortho"], out, cfg["crs"], R_lidar_to_cam
    )
    cover = [s["lidar_coverage_ratio"] for s in stats["frames"]]
    print(
        f"ortho: {len(stats['frames'])} frames warped, "
        f"mean lidar coverage={np.mean(cover) * 100:.1f}% ({time.time() - t0:.1f}s)"
    )
    return 0


def _cmd_run(cfg: dict) -> int:
    rc = _cmd_check(cfg)
    if rc:
        return rc
    rc = _cmd_align(cfg)
    if rc:
        return rc
    rc = _cmd_dsm(cfg)
    if rc:
        return rc
    return _cmd_ortho(cfg)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="orthoproto", description="T14+T15 offline true-ortho prototype"
    )
    sub = parser.add_subparsers(dest="command")
    for name in ("run", "check", "align", "dsm", "ortho"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, help="capture config YAML")
    args = parser.parse_args(argv)
    cfg_path = Path(args.config).resolve()
    base = cfg_path.parents[2]  # configs/orthoproto/<name>.yaml -> repo root
    cfg = load_config(cfg_path, base)
    if args.command == "run":
        return _cmd_run(cfg)
    if args.command == "check":
        return _cmd_check(cfg)
    if args.command == "align":
        return _cmd_align(cfg)
    if args.command == "dsm":
        return _cmd_dsm(cfg)
    if args.command == "ortho":
        return _cmd_ortho(cfg)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
