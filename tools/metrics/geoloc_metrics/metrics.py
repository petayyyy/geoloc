"""Every metric from ``docs/plan/testing/05-metrics.md``, implemented once.

The functions are pure and vectorized: they take numpy arrays and return
numbers. No file I/O, no ROS, no decision logic -- a metric only *describes*
what happened, it never decides whether a fix was valid. The integrity decision
(``accepted``) is an input column, never derived here.

The one correctness rule that everything else hangs off (task card T12 pitfall
#1): ``A@d`` and ``IFR`` are computed over **accepted** fixes, never over all
attempts. Availability is a separate metric (``acceptance_rate``).
"""

from __future__ import annotations

import numpy as np

from .schema import covariance_3x3

IFR_THRESHOLD_M = 50.0


# ---------------------------------------------------------------------------
# Derived errors
# ---------------------------------------------------------------------------


def position_error(east_err: np.ndarray, north_err: np.ndarray) -> np.ndarray:
    """Horizontal position error magnitude, ``sqrt(de^2 + dn^2)``."""
    return np.hypot(np.asarray(east_err, dtype=np.float64), np.asarray(north_err, dtype=np.float64))


def yaw_error_deg(est_yaw: np.ndarray, gt_yaw: np.ndarray) -> np.ndarray:
    """Smallest absolute yaw error in degrees, respecting the [-pi, pi] wrap."""
    d = np.angle(np.exp(1j * (np.asarray(est_yaw) - np.asarray(gt_yaw))))
    return np.abs(np.degrees(d))


# ---------------------------------------------------------------------------
# Fix-level metrics (05-metrics.md section 1)
# ---------------------------------------------------------------------------


def a_at_d(pos_err_accepted: np.ndarray, d: float) -> float:
    """Fraction of **accepted** fixes with position error <= ``d`` metres."""
    err = np.asarray(pos_err_accepted, dtype=np.float64)
    if err.size == 0:
        return float("nan")
    return float(np.mean(err <= d))


def ifr(pos_err_accepted: np.ndarray, threshold: float = IFR_THRESHOLD_M) -> float:
    """Integrity Failure Rate: fraction of **accepted** fixes with error > threshold."""
    err = np.asarray(pos_err_accepted, dtype=np.float64)
    if err.size == 0:
        return float("nan")
    return float(np.mean(err > threshold))


def acceptance_rate(n_accepted: int, n_attempts: int) -> float:
    """Accepted fixes / all attempts. Availability, kept separate from accuracy."""
    if n_attempts == 0:
        return float("nan")
    return n_accepted / n_attempts


def re_median_p95(yaw_err_accepted_deg: np.ndarray) -> tuple[float, float]:
    """Median and p95 of accepted-fix heading error (degrees)."""
    err = np.asarray(yaw_err_accepted_deg, dtype=np.float64)
    if err.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(err, 50)), float(np.percentile(err, 95))


def percentile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def latency_p50_p95(latency_ms: np.ndarray) -> tuple[float, float]:
    return percentile(latency_ms, 50), percentile(latency_ms, 95)


# ---------------------------------------------------------------------------
# Trajectory-level metrics (05-metrics.md section 2)
# ---------------------------------------------------------------------------


def ate_rmse(pos_err: np.ndarray) -> float:
    """Root-mean-square position error over the whole trajectory."""
    err = np.asarray(pos_err, dtype=np.float64)
    if err.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(err**2)))


def lateral_error(east_err: np.ndarray, north_err: np.ndarray, gt_yaw: np.ndarray) -> np.ndarray:
    """Cross-track error: the component of the position error perpendicular to
    the true heading (ENU yaw, CCW from East)."""
    e = np.asarray(east_err, dtype=np.float64)
    n = np.asarray(north_err, dtype=np.float64)
    yaw = np.asarray(gt_yaw, dtype=np.float64)
    return np.abs(-e * np.sin(yaw) + n * np.cos(yaw))


def lateral_p95(east_err: np.ndarray, north_err: np.ndarray, gt_yaw: np.ndarray) -> float:
    return percentile(lateral_error(east_err, north_err, gt_yaw), 95)


def convergence_distance(
    path_m: np.ndarray, sigma_yaw_deg: np.ndarray, threshold: float = 1.0
) -> float:
    """Path distance until yaw sigma first drops below ``threshold`` degrees."""
    p = np.asarray(path_m, dtype=np.float64)
    s = np.asarray(sigma_yaw_deg, dtype=np.float64)
    idx = np.flatnonzero(s < threshold)
    if idx.size == 0:
        return float("nan")
    return float(p[idx[0]])


def fixes_in_first_2km(accepted: np.ndarray, path_m: np.ndarray) -> int:
    """Number of accepted fixes with path distance < 2 km from start."""
    acc = np.asarray(accepted, dtype=bool)
    p = np.asarray(path_m, dtype=np.float64)
    return int(np.sum(acc & (p < 2000.0)))


def ttff(t_s: np.ndarray, accepted: np.ndarray, t0: float | None = None) -> float:
    """Time to first fix: seconds from the first sample to the first accepted fix."""
    t = np.asarray(t_s, dtype=np.float64)
    acc = np.asarray(accepted, dtype=bool)
    idx = np.flatnonzero(acc)
    if idx.size == 0:
        return float("nan")
    start = t[0] if t0 is None else t0
    return float(t[idx[0]] - start)


def max_gap_distance(path_m: np.ndarray, accepted: np.ndarray) -> float:
    """Maximum path distance between consecutive accepted fixes."""
    p = np.asarray(path_m, dtype=np.float64)
    acc = np.asarray(accepted, dtype=bool)
    idx = np.flatnonzero(acc)
    if idx.size < 2:
        return float("nan")
    gaps = np.diff(p[idx])
    return float(gaps.max()) if gaps.size else float("nan")


# ---------------------------------------------------------------------------
# Consistency metrics (05-metrics.md section 3)
# ---------------------------------------------------------------------------


def _nees_samples(
    error_east: np.ndarray,
    error_north: np.ndarray,
    error_yaw: np.ndarray,
    cov: np.ndarray,
    dof: int,
) -> np.ndarray:
    """Normalized estimation error squared per sample: e^T C^{-1} e, limited to
    the first ``dof`` state dimensions (position, optionally yaw)."""
    e = np.stack([np.asarray(error_east), np.asarray(error_north), np.asarray(error_yaw)], axis=1)
    e = e[:, :dof]
    c = cov[:, :dof, :dof]
    out = np.empty(len(e), dtype=np.float64)
    for i in range(len(e)):
        out[i] = float(e[i] @ np.linalg.inv(c[i]) @ e[i])
    return out


def nees_mean(
    error_east: np.ndarray,
    error_north: np.ndarray,
    error_yaw: np.ndarray,
    cov: np.ndarray,
    dof: int = 2,
) -> float:
    """Mean NEES over a Monte-Carlo set. Well-calibrated -> ``~dof``; understated
    covariance -> inflated (this is the T12-U-04 / T21 calibration signal)."""
    samples = _nees_samples(error_east, error_north, error_yaw, cov, dof)
    if samples.size == 0:
        return float("nan")
    return float(np.mean(samples))


def nis_mean(
    innov_east: np.ndarray,
    innov_north: np.ndarray,
    innov_yaw: np.ndarray,
    innov_cov: np.ndarray,
    dof: int = 2,
) -> float:
    """Mean normalized innovation squared over the fix sequence. ``innov`` is the
    difference between each fix and the constant-velocity prediction from the
    previous accepted fix; ``innov_cov`` is the sum of the two covariances."""
    return nees_mean(innov_east, innov_north, innov_yaw, innov_cov, dof)


def sigma_coverage(pos_err: np.ndarray, sigma_pos: np.ndarray, k: float) -> float:
    """Fraction of samples where the true error is within ``k`` sigma of the
    estimate. ``sigma_pos = sqrt(cov_ee + cov_nn)``."""
    err = np.asarray(pos_err, dtype=np.float64)
    s = np.asarray(sigma_pos, dtype=np.float64)
    if err.size == 0:
        return float("nan")
    return float(np.mean(err <= k * s))


def sigma_pos_from_cov(cov: np.ndarray) -> np.ndarray:
    return np.sqrt(cov[:, 0, 0] + cov[:, 1, 1])


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def fix_level_table(
    rec: np.ndarray, bias: tuple[float, float] = (0.0, 0.0), bias_mode: str = "with"
) -> dict:
    """All fix-level metrics for a set of fix records, as a flat dict.

    ``bias`` / ``bias_mode`` implement the dual bias accounting of T12: metrics
    are reported both with the measured basemap bias included ("with", the raw
    map error) and with it subtracted ("without", the system's own error).
    """
    accepted = rec["accepted"]
    b_e, b_n = (0.0, 0.0) if bias_mode == "with" else bias
    east_err = rec["est_east"] - rec["gt_east"] - b_e
    north_err = rec["est_north"] - rec["gt_north"] - b_n
    pos_err = position_error(east_err, north_err)
    yerr = yaw_error_deg(rec["est_yaw"], rec["gt_yaw"])

    acc_err = pos_err[accepted]
    acc_yerr = yerr[accepted]
    n_attempts = int(len(rec))
    n_accepted = int(np.sum(accepted))

    re_med, re_p95 = re_median_p95(acc_yerr)
    lat50, lat95 = latency_p50_p95(rec["latency_ms"])

    return {
        "n_attempts": n_attempts,
        "n_accepted": n_accepted,
        "acceptance_rate": acceptance_rate(n_accepted, n_attempts),
        "A@5": a_at_d(acc_err, 5.0),
        "A@10": a_at_d(acc_err, 10.0),
        "A@20": a_at_d(acc_err, 20.0),
        "A@50": a_at_d(acc_err, 50.0),
        "RE_med_deg": re_med,
        "RE_p95_deg": re_p95,
        "IFR": ifr(acc_err),
        "latency_p50_ms": lat50,
        "latency_p95_ms": lat95,
    }


def quality_distribution(rec: np.ndarray) -> dict:
    """p5/median/p95 of each quality attribute, over accepted fixes."""
    accepted = rec["accepted"]
    out = {}
    for field in (
        "n_inliers",
        "inlier_ratio",
        "covisibility",
        "peak_ratio",
        "residual_rms_px",
        "spatial_spread",
        "mean_confidence",
    ):
        vals = rec[field][accepted].astype(np.float64)
        out[field] = {
            "p5": percentile(vals, 5),
            "median": percentile(vals, 50),
            "p95": percentile(vals, 95),
        }
    return out


def trajectory_level_table(rec: np.ndarray) -> dict:
    """Trajectory-level metrics from a trajectory record table."""
    if rec is None or len(rec) == 0:
        return {}
    east_err = rec["est_east"] - rec["gt_east"]
    north_err = rec["est_north"] - rec["gt_north"]
    pos_err = position_error(east_err, north_err)
    yerr = yaw_error_deg(rec["est_yaw"], rec["gt_yaw"])
    cov = covariance_3x3(rec)
    sigma_yaw_deg = np.degrees(np.sqrt(np.clip(cov[:, 2, 2], 0, None)))
    return {
        "ATE_RMSE": ate_rmse(pos_err),
        "lateral_p95": lateral_p95(east_err, north_err, rec["gt_yaw"]),
        "yaw_error_converged_p95": percentile(yerr, 95),
        "convergence_distance": convergence_distance(rec["path_m"], sigma_yaw_deg),
        "max_gap_distance": max_gap_distance(rec["path_m"], np.ones(len(rec), dtype=bool)),
    }
