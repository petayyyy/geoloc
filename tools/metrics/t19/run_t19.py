#!/usr/bin/env python3
"""Level-B runner: evaluate the T19 phase-correlation channel on a replayed
FAST-LIVO2 capture (T12 + T19 evaluation).

The runner sits between the offline true-ortho prototype (``orthoproto``) and
the metrics harness (``geoloc_metrics``):

    processed bag ──(orthoproto)──► true-ortho patches + UTM alignment
                                        │
    satellite basemap (geopack) ────────┤
    RTK (from the bag) ─────────────────┤
                                        ▼
                              run_t19  ──►  t19_match (C++, same code as the node)
                                        │
                                        ▼
                              fix + trajectory records  ──►  geoloc_metrics report

Ground truth is RTK. The prior for each patch is its aligned UTM centre (the
position orthoproto rendered it at, ~7-9 m from RTK on this capture). The
matcher searches a window centred on that prior and returns a shift; the fix is
``prior + shift``, and the error is ``|fix - RTK|``. That error includes the
basemap georeferencing bias (T09), which is why the harness reports both with
and without it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoloc_metrics.report import render_report
from geoloc_metrics.schema import (
    FIX_DTYPE,
    TRAJECTORY_DTYPE,
    Records,
    save_records,
)
from geoloc_metrics.summary import Summary
from geoloc_metrics.terrain import assign_terrain_class

DEFAULT_MATCH_BIN = str(Path(__file__).resolve().parent / "t19_match")


def _read_rtk_utm(bag_dir: Path, crs: str, swap_latlon: bool) -> np.ndarray:
    """(t_s, east, north) UTM trajectory from the bag's RTK topic."""
    from pyproj import Transformer
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    trans = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    rows = []
    with Reader(bag_dir) as reader:
        for conn, _t, raw in reader.messages():
            if conn.topic != "/dji_osdk_ros/rtk_position":
                continue
            m = ts.deserialize_cdr(raw, conn.msgtype)
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            lat, lon = (m.longitude, m.latitude) if swap_latlon else (m.latitude, m.longitude)
            rows.append((t, lon, lat))
    arr = np.asarray(rows, dtype=np.float64)
    east, north = trans.transform(arr[:, 1], arr[:, 2])
    return np.column_stack([arr[:, 0], np.asarray(east), np.asarray(north)])


def _interp_rtk(rtk_utm: np.ndarray, t: float) -> tuple[float, float]:
    e = np.interp(t, rtk_utm[:, 0], rtk_utm[:, 1])
    n = np.interp(t, rtk_utm[:, 0], rtk_utm[:, 2])
    return float(e), float(n)


def _read_geotiff(path: Path) -> tuple[np.ndarray, tuple]:
    """Return (array[H,W[,C]], affine(a,b,c,d,e,f)) for a GeoTIFF."""
    import rasterio

    with rasterio.open(path) as src:
        data = src.read()
        transform = src.transform
    # (bands, H, W) -> (H, W[, bands])
    if data.ndim == 3:
        data = data.transpose(1, 2, 0)
    return data, (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)


def _extract_window(
    base: np.ndarray, affine: tuple, center: tuple, gsd: float, radius_m: float
) -> tuple[np.ndarray, tuple[float, float]]:
    """Extract a square window at ``gsd`` centred on ``center`` (east, north).

    Returns (window, (origin_east, origin_north)) where origin is the top-left
    (north-west) corner. ``base`` is a north-up raster (H, W) or (H, W, C) with
    a GDAL affine ``(a, b, c, d, e, f)``.
    """
    a, _b, c, _d, e, f = affine
    h, w = base.shape[:2]
    size_px = int(round(2 * radius_m / gsd))
    half_m = radius_m
    e0, n0 = center
    shape = (size_px, size_px) + base.shape[2:]
    out = np.zeros(shape, dtype=base.dtype)
    for i in range(size_px):
        north = n0 + half_m - (i + 0.5) * gsd
        row = int((north - f) / e)
        if not (0 <= row < h):
            continue
        for j in range(size_px):
            east = e0 - half_m + (j + 0.5) * gsd
            col = int((east - c) / a)
            if 0 <= col < w:
                out[i, j] = base[row, col]
    origin = (e0 - half_m, n0 + half_m)
    return out, origin


def _to_gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        # GeoTIFF basemap is RGB; patches are RGB too.
        arr = arr[..., :3].astype(np.float64)
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    else:
        gray = arr.astype(np.float64)
    return gray / 255.0


def _run_matcher(
    match_bin: str,
    patch: np.ndarray,
    mapimg: np.ndarray,
    conf: np.ndarray,
    gsd: float,
    matcher_cfg: dict,
) -> dict:
    ph, pw = patch.shape
    mh, mw = mapimg.shape
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "p.bin").write_bytes(patch.astype(np.float64).tobytes())
        (td / "m.bin").write_bytes(mapimg.astype(np.float64).tobytes())
        (td / "c.bin").write_bytes(conf.astype(np.float64).tobytes())
        cmd = [
            match_bin,
            "--patch",
            str(td / "p.bin"),
            "--map",
            str(td / "m.bin"),
            "--conf",
            str(td / "c.bin"),
            "--pw",
            str(pw),
            "--ph",
            str(ph),
            "--mw",
            str(mw),
            "--mh",
            str(mh),
            "--gsd",
            str(gsd),
            "--grad-thresh",
            str(matcher_cfg.get("grad_thresh_rel", 0.0)),
            "--coarse-max",
            str(matcher_cfg.get("coarse_max_deg", 18.0)),
            "--coarse-step",
            str(matcher_cfg.get("coarse_step_deg", 6.0)),
            "--nrho",
            str(matcher_cfg.get("nrho", 64)),
            "--ntheta",
            str(matcher_cfg.get("ntheta", 512)),
            "--scale-tol",
            str(matcher_cfg.get("scale_check_tolerance", 0.10)),
            "--min-peak-ratio",
            str(matcher_cfg.get("min_peak_ratio", 1.6)),
            "--pos-sigma",
            str(matcher_cfg.get("position_sigma_m", 8.0)),
            "--yaw-sigma",
            str(matcher_cfg.get("yaw_sigma_deg", 1.5)),
            "--bias-sigma",
            str(matcher_cfg.get("basemap_bias_sigma_m", 3.0)),
        ]
        out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def _gate(result: dict, gate_cfg: dict) -> bool:
    return bool(
        result["success"]
        and not result["bad_scale"]
        and result["peak_ratio"] >= gate_cfg.get("min_peak_ratio", 1.6)
        and result["valid_fraction"] >= gate_cfg.get("min_covisibility", 0.20)
    )


def run(cfg: dict, match_bin: str = DEFAULT_MATCH_BIN) -> Path:
    base = Path(cfg["_base"])
    bag_dir = (base / cfg["bag_dir"]).resolve()
    ortho_dir = (base / cfg["ortho_dir"]).resolve()
    basemap_path = (base / cfg["basemap"]).resolve()
    semantic_path = (base / cfg["semantic"]).resolve()
    out_dir = (base / cfg["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    crs = cfg["crs"]
    rtk = _read_rtk_utm(bag_dir, crs, bool(cfg["rtk"].get("swap_latlon", False)))

    base_img, base_affine = _read_geotiff(basemap_path)
    semantic, semantic_affine = _read_geotiff(semantic_path)
    if semantic.ndim == 2:
        semantic = semantic.astype(np.uint8)
    else:
        semantic = semantic[..., 0].astype(np.uint8)

    stats = yaml.safe_load((ortho_dir / "ortho_stats.yaml").read_text())["frames"]
    idx_to_t = {int(s["idx"]): float(s["t"]) for s in stats}

    gsd = float(cfg["matcher"].get("gsd", 0.5))
    prior_radius = float(cfg["matcher"].get("prior_radius_m", 60.0))
    patch_radius = float(cfg.get("patch_radius_m", 130.0))
    # The matcher's FFT requires a power-of-two map window; round the search
    # radius up to the next power-of-two edge length.
    need_px = 2 * (patch_radius + prior_radius) / gsd
    win_px = 2 ** int(np.ceil(np.log2(need_px)))
    window_radius = win_px * gsd / 2.0

    fix_rows = []
    traj_rows = []
    patches_dir = ortho_dir / "patches"

    patch_files = sorted(patches_dir.glob("patch_*.tif"))
    patch_files = [p for p in patch_files if not p.name.endswith("_conf.tif")]

    for pfile in patch_files:
        idx = int(pfile.name.split("_")[1].split(".")[0])
        t = idx_to_t.get(idx)
        if t is None:
            continue
        patch_rgb, patch_affine = _read_geotiff(pfile)
        conf, _ = _read_geotiff(pfile.with_name(pfile.stem + "_conf.tif"))
        # patch centre = (east_min + radius, north_max - radius)
        a, _b, c, _d, _e, f = patch_affine
        east_min, north_max = c, f
        px_g = abs(a)
        half = patch_rgb.shape[1] * px_g / 2.0
        center_e = east_min + half
        center_n = north_max - half

        rtk_e, rtk_n = _interp_rtk(rtk, t)

        window, win_origin = _extract_window(
            base_img, base_affine, (center_e, center_n), gsd, window_radius
        )
        gray_patch = _to_gray(patch_rgb)
        gray_map = _to_gray(window)
        gray_conf = _to_gray(conf[..., 0] if conf.ndim == 3 else conf)

        result = _run_matcher(match_bin, gray_patch, gray_map, gray_conf, gsd, cfg["matcher"])

        # The matcher reports where the patch's top-left corner sits in the map
        # window. The fix (corrected centre) = window origin + shift + half-patch.
        est_east = win_origin[0] + result["shift_east_px"] * gsd + half
        est_north = win_origin[1] + result["shift_north_px"] * gsd - half

        accepted = _gate(result, cfg.get("gate", {}))

        terrain = assign_terrain_class(
            np.array([center_e]),
            np.array([center_n]),
            semantic,
            semantic_affine,
            window_radius_m=patch_radius,
        )[0]

        fix = np.empty(1, dtype=FIX_DTYPE)[0]
        fix["run_id"] = cfg.get("run_id", "amtown03_t19")
        fix["level"] = "B"
        fix["target"] = cfg.get("target", "x86")
        fix["t_s"] = t
        fix["attempt_index"] = idx
        fix["channel"] = 1
        fix["accepted"] = accepted
        fix["gt_east"] = rtk_e
        fix["gt_north"] = rtk_n
        fix["gt_yaw"] = 0.0
        fix["est_east"] = est_east
        fix["est_north"] = est_north
        fix["est_yaw"] = result["delta_yaw_fix_rad"]
        fix["cov_ee"] = result["cov"][0]
        fix["cov_en"] = result["cov"][1]
        fix["cov_ey"] = result["cov"][2]
        fix["cov_nn"] = result["cov"][3]
        fix["cov_ny"] = result["cov"][4]
        fix["cov_yy"] = result["cov"][5]
        fix["bias_east"] = float(cfg.get("bias", {}).get("east", 0.0))
        fix["bias_north"] = float(cfg.get("bias", {}).get("north", 0.0))
        fix["terrain"] = terrain
        fix["n_correspondences"] = result["n_correspondences"]
        fix["n_inliers"] = result["n_inliers"]
        fix["inlier_ratio"] = result["inlier_ratio"]
        fix["covisibility"] = result["covisibility"]
        fix["peak_ratio"] = result["peak_ratio"]
        fix["residual_rms_px"] = result["residual_rms_px"]
        fix["spatial_spread"] = result["spatial_spread"]
        fix["mean_confidence"] = result["mean_confidence"]
        fix["scale_check"] = result["scale_check"]
        fix["latency_ms"] = float(result.get("match_ms", 0.0))
        fix_rows.append(fix)

        # trajectory: aligned pose (prior) vs RTK, per patch frame
        traj = np.empty(1, dtype=TRAJECTORY_DTYPE)[0]
        traj["run_id"] = cfg.get("run_id", "amtown03_t19")
        traj["level"] = "B"
        traj["target"] = cfg.get("target", "x86")
        traj["t_s"] = t
        traj["gt_east"] = rtk_e
        traj["gt_north"] = rtk_n
        traj["gt_yaw"] = 0.0
        # The TRAJECTORY record is the pose estimate, i.e. the aligned prior --
        # not the matcher output. Writing est_* here made ATE_RMSE a measure of
        # the (rejected) match instead of the trajectory it is defined on.
        traj["est_east"] = center_e
        traj["est_north"] = center_n
        traj["est_yaw"] = 0.0
        traj["cov_ee"] = result["cov"][0]
        traj["cov_en"] = result["cov"][1]
        traj["cov_ey"] = result["cov"][2]
        traj["cov_nn"] = result["cov"][3]
        traj["cov_ny"] = result["cov"][4]
        traj["cov_yy"] = result["cov"][5]
        traj["path_m"] = 0.0
        traj["terrain"] = terrain
        traj_rows.append(traj)

    fixes = np.asarray(fix_rows, dtype=FIX_DTYPE) if fix_rows else np.empty(0, dtype=FIX_DTYPE)
    trajectory = (
        np.asarray(traj_rows, dtype=TRAJECTORY_DTYPE)
        if traj_rows
        else np.empty(0, dtype=TRAJECTORY_DTYPE)
    )
    records = Records(fixes=fixes, trajectory=trajectory)
    save_records(out_dir / "records", records)

    bias = (
        float(cfg.get("bias", {}).get("east", 0.0)),
        float(cfg.get("bias", {}).get("north", 0.0)),
    )
    summary = Summary.from_records(
        records,
        level="B",
        target=cfg.get("target", "x86"),
        git_sha=cfg.get("git_sha", ""),
        seed=cfg.get("seed"),
        bias=bias,
    )
    summary.write(out_dir / "summary.json")
    render_report(summary, records, out_dir / "report.html")
    print(
        f"n_fixes={len(fixes)} n_accepted={int(np.sum(fixes['accepted']))} "
        f"IFR={summary.fix_level['IFR']:.4f} A@20={summary.fix_level['A@20']:.3f}"
    )
    return out_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="T19 level-B runner")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    cfg_path = Path(args.config).resolve()
    base = cfg_path.parents[2]  # configs/metrics/<name>.yaml -> repo root
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_base"] = str(base)
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
