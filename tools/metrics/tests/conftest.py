import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from geoloc_metrics.schema import FIX_DTYPE, TRAJECTORY_DTYPE


def make_fixes(
    n,
    *,
    est_east,
    est_north,
    gt_east=None,
    gt_north=None,
    accepted=None,
    est_yaw=0.0,
    gt_yaw=0.0,
    cov_ee=1.0,
    cov_nn=1.0,
    cov_yy=0.01,
    bias_east=0.0,
    bias_north=0.0,
    terrain="urban",
    latency_ms=100.0,
    run_id="run",
    level="B",
    peak_ratio=2.0,
    n_inliers=100,
    channel=1,
):
    est_east = np.broadcast_to(np.asarray(est_east, dtype=np.float64), (n,))
    est_north = np.broadcast_to(np.asarray(est_north, dtype=np.float64), (n,))
    if gt_east is None:
        gt_east = np.zeros(n)
    if gt_north is None:
        gt_north = np.zeros(n)
    gt_east = np.broadcast_to(np.asarray(gt_east, dtype=np.float64), (n,))
    gt_north = np.broadcast_to(np.asarray(gt_north, dtype=np.float64), (n,))
    if accepted is None:
        accepted = np.ones(n, dtype=bool)
    accepted = np.broadcast_to(np.asarray(accepted, dtype=bool), (n,))

    rec = np.empty(n, dtype=FIX_DTYPE)
    rec["run_id"] = run_id
    rec["level"] = level
    rec["target"] = "x86"
    rec["t_s"] = np.arange(n, dtype=np.float64)
    rec["attempt_index"] = np.arange(n)
    rec["channel"] = channel
    rec["accepted"] = accepted
    rec["gt_east"] = gt_east
    rec["gt_north"] = gt_north
    rec["gt_yaw"] = np.broadcast_to(np.asarray(gt_yaw, dtype=np.float64), (n,))
    rec["est_east"] = est_east
    rec["est_north"] = est_north
    rec["est_yaw"] = np.broadcast_to(np.asarray(est_yaw, dtype=np.float64), (n,))
    rec["cov_ee"] = cov_ee
    rec["cov_en"] = 0.0
    rec["cov_ey"] = 0.0
    rec["cov_nn"] = cov_nn
    rec["cov_ny"] = 0.0
    rec["cov_yy"] = cov_yy
    rec["bias_east"] = bias_east
    rec["bias_north"] = bias_north
    rec["terrain"] = terrain
    rec["n_correspondences"] = n_inliers
    rec["n_inliers"] = n_inliers
    rec["inlier_ratio"] = 1.0
    rec["covisibility"] = 0.5
    rec["peak_ratio"] = peak_ratio
    rec["residual_rms_px"] = 1.0
    rec["spatial_spread"] = 0.5
    rec["mean_confidence"] = 0.5
    rec["scale_check"] = 1.0
    rec["latency_ms"] = latency_ms
    return rec


def make_trajectory(
    n, *, gt_east, gt_north, est_east=None, est_north=None, gt_yaw=0.0, est_yaw=0.0, path_m=None
):
    gt_east = np.broadcast_to(np.asarray(gt_east, dtype=np.float64), (n,))
    gt_north = np.broadcast_to(np.asarray(gt_north, dtype=np.float64), (n,))
    if est_east is None:
        est_east = gt_east.copy()
    if est_north is None:
        est_north = gt_north.copy()
    if path_m is None:
        path_m = np.arange(n, dtype=np.float64)
    rec = np.empty(n, dtype=TRAJECTORY_DTYPE)
    rec["run_id"] = "run"
    rec["level"] = "B"
    rec["target"] = "x86"
    rec["t_s"] = np.arange(n, dtype=np.float64)
    rec["gt_east"] = gt_east
    rec["gt_north"] = gt_north
    rec["gt_yaw"] = np.broadcast_to(np.asarray(gt_yaw, dtype=np.float64), (n,))
    rec["est_east"] = est_east
    rec["est_north"] = est_north
    rec["est_yaw"] = np.broadcast_to(np.asarray(est_yaw, dtype=np.float64), (n,))
    rec["cov_ee"] = 1.0
    rec["cov_en"] = 0.0
    rec["cov_ey"] = 0.0
    rec["cov_nn"] = 1.0
    rec["cov_ny"] = 0.0
    rec["cov_yy"] = 0.01
    rec["path_m"] = np.broadcast_to(np.asarray(path_m, dtype=np.float64), (n,))
    rec["terrain"] = "urban"
    return rec


@pytest.fixture
def rng():
    return np.random.default_rng(0)
