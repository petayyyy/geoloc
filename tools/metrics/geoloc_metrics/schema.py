"""The common record format the metrics harness consumes.

Both level A (OrthoSim) and level B (replay) runners emit these two tables, so
the harness computes every metric identically regardless of where the data came
from. The contract is deliberately plain (CSV of a fixed numpy structured
dtype): it must survive a hand-edit on the flight line and load without any
heavy dependency.

Two tables:

``fixes``      -- one row per matcher attempt (accepted or rejected).
``trajectory`` -- one row per pose sample (the full fused/estimated trajectory
                  vs ground truth), for trajectory-level metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Fix record: one row per matcher attempt.
# ---------------------------------------------------------------------------
#
# Covariance is the symmetric 3x3 over (east, north, yaw), stored row-major
# upper triangle: ee, en, ey, nn, ny, yy. Angles are radians, ENU yaw CCW from
# East, normalised to [-pi, pi] (the project convention). ``accepted`` is the
# integrity decision (geoloc_integrity, T22); the harness never decides it, it
# only reports metrics conditioned on it -- matching the rule that the matcher
# produces quality, integrity accepts.
FIX_FIELDS = [
    ("run_id", "U64"),
    ("level", "U1"),
    ("target", "U16"),
    ("t_s", "f8"),
    ("attempt_index", "i8"),
    ("channel", "i4"),
    ("accepted", "bool"),
    ("gt_east", "f8"),
    ("gt_north", "f8"),
    ("gt_yaw", "f8"),
    ("est_east", "f8"),
    ("est_north", "f8"),
    ("est_yaw", "f8"),
    ("cov_ee", "f8"),
    ("cov_en", "f8"),
    ("cov_ey", "f8"),
    ("cov_nn", "f8"),
    ("cov_ny", "f8"),
    ("cov_yy", "f8"),
    ("bias_east", "f8"),
    ("bias_north", "f8"),
    ("terrain", "U16"),
    ("n_correspondences", "i8"),
    ("n_inliers", "i8"),
    ("inlier_ratio", "f8"),
    ("covisibility", "f8"),
    ("peak_ratio", "f8"),
    ("residual_rms_px", "f8"),
    ("spatial_spread", "f8"),
    ("mean_confidence", "f8"),
    ("scale_check", "f8"),
    ("latency_ms", "f8"),
]

TRAJECTORY_FIELDS = [
    ("run_id", "U64"),
    ("level", "U1"),
    ("target", "U16"),
    ("t_s", "f8"),
    ("gt_east", "f8"),
    ("gt_north", "f8"),
    ("gt_yaw", "f8"),
    ("est_east", "f8"),
    ("est_north", "f8"),
    ("est_yaw", "f8"),
    ("cov_ee", "f8"),
    ("cov_en", "f8"),
    ("cov_ey", "f8"),
    ("cov_nn", "f8"),
    ("cov_ny", "f8"),
    ("cov_yy", "f8"),
    ("path_m", "f8"),
    ("terrain", "U16"),
]

FIX_DTYPE = np.dtype(FIX_FIELDS)
TRAJECTORY_DTYPE = np.dtype(TRAJECTORY_FIELDS)


@dataclass
class Records:
    """A loaded, validated run: fixes and (optionally) a trajectory."""

    fixes: np.ndarray
    trajectory: np.ndarray | None = None

    @property
    def run_id(self) -> str:
        if len(self.fixes) == 0:
            return ""
        return str(self.fixes["run_id"][0])


def empty_fixes() -> np.ndarray:
    return np.empty(0, dtype=FIX_DTYPE)


def empty_trajectory() -> np.ndarray:
    return np.empty(0, dtype=TRAJECTORY_DTYPE)


def make_fixes(rows: list[tuple]) -> np.ndarray:
    """Build a fix table from a list of tuples in FIX_FIELDS order."""
    out = np.empty(len(rows), dtype=FIX_DTYPE)
    for i, row in enumerate(rows):
        out[i] = row
    return out


def make_trajectory(rows: list[tuple]) -> np.ndarray:
    out = np.empty(len(rows), dtype=TRAJECTORY_DTYPE)
    for i, row in enumerate(rows):
        out[i] = row
    return out


def validate_fixes(fixes: np.ndarray) -> None:
    if fixes.dtype != FIX_DTYPE:
        raise ValueError(f"fixes dtype {fixes.dtype} does not match the fix contract")
    if len(fixes) and not np.isfinite(fixes["gt_east"]).all():
        raise ValueError("fixes contain non-finite ground truth east")
    if len(fixes) and not np.isfinite(fixes["gt_north"]).all():
        raise ValueError("fixes contain non-finite ground truth north")


def _fmt_for(fields: list[tuple[str, str]]) -> list[str]:
    """Per-field ``np.savetxt`` format strings derived from the dtype."""
    fmt = []
    for _, kind in fields:
        if kind == "f8":
            fmt.append("%.6f")
        elif kind == "i8":
            fmt.append("%d")
        elif kind == "bool":
            fmt.append("%d")
        else:  # unicode strings
            fmt.append("%s")
    return fmt


def _fields_names(fields: list[tuple[str, str]]) -> list[str]:
    return [name for name, _ in fields]


def _base_stem(path: Path) -> Path:
    """Strip a trailing ``.csv`` / ``.fixes.csv`` / ``.trajectory.csv`` suffix."""
    s = str(path)
    for suffix in (".trajectory.csv", ".fixes.csv", ".csv"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return Path(s)


def save_records(path: Path, records: Records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = _base_stem(path)
    fix_path = Path(str(stem) + ".fixes.csv")
    np.savetxt(
        fix_path,
        records.fixes,
        fmt=_fmt_for(FIX_FIELDS),
        delimiter=",",
        header=",".join(_fields_names(FIX_FIELDS)),
        comments="",
    )
    if records.trajectory is not None and len(records.trajectory):
        traj_path = Path(str(stem) + ".trajectory.csv")
        np.savetxt(
            traj_path,
            records.trajectory,
            fmt=_fmt_for(TRAJECTORY_FIELDS),
            delimiter=",",
            header=",".join(_fields_names(TRAJECTORY_FIELDS)),
            comments="",
        )


def load_records(path: Path) -> Records:
    """Load a run previously written by ``save_records`` (or an equivalent CSV)."""
    stem = _base_stem(path)
    fix_path = Path(str(stem) + ".fixes.csv")
    fixes = np.atleast_1d(np.genfromtxt(fix_path, delimiter=",", names=True, dtype=FIX_DTYPE))
    trajectory = None
    traj_path = Path(str(stem) + ".trajectory.csv")
    if traj_path.exists():
        trajectory = np.atleast_1d(
            np.genfromtxt(traj_path, delimiter=",", names=True, dtype=TRAJECTORY_DTYPE)
        )
    return Records(fixes=fixes, trajectory=trajectory)


def covariance_3x3(rec: np.ndarray) -> np.ndarray:
    """Reassemble the symmetric 3x3 covariance from the upper-triangle columns.

    Accepts a structured record (``cov_ee`` ... ``cov_yy``) and returns an
    ``(N, 3, 3)`` array over (east, north, yaw).
    """
    n = len(rec)
    cov = np.zeros((n, 3, 3), dtype=np.float64)
    cov[:, 0, 0] = rec["cov_ee"]
    cov[:, 1, 1] = rec["cov_nn"]
    cov[:, 2, 2] = rec["cov_yy"]
    cov[:, 0, 1] = cov[:, 1, 0] = rec["cov_en"]
    cov[:, 0, 2] = cov[:, 2, 0] = rec["cov_ey"]
    cov[:, 1, 2] = cov[:, 2, 1] = rec["cov_ny"]
    return cov
