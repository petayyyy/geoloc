"""Metric self-consistency check for a capture's RTK track (georeference guard).

A projected RTK track and the receiver's own Doppler velocity are two
*independent* metric statements about the same motion. If the geographic
interpretation of the fix messages is right, they must agree: over a short
baseline the distance travelled according to the projected positions and the
distance according to the integrated velocity are the same number.

They stop agreeing the moment latitude and longitude are mixed up, because
the metres-per-degree factors differ per axis *and* per latitude. Reading a
fix at 39.92 N / 44.83 E as 44.83 N / 39.92 E stretches northing by
~111128/85463 = 1.300 and squeezes easting by ~79067/111033 = 0.712 -- an
*anisotropic* distortion, so it survives every rigid or similarity fit and
shows up downstream as a bogus "odometry scale error" instead.

Deliberately short baselines: integrated velocity accumulates drift, so a
whole-capture comparison is a low-power test (the drift dilutes the ratio
towards 1). Sub-10 s baselines are where the two sources are both trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# rtk_velocity frame conventions: which vector component is which axis.
VELOCITY_FRAMES = {
    "ned": (1, 0),  # (east_index, north_index) -- x=North, y=East
    "enu": (0, 1),  # x=East, y=North
}


@dataclass(frozen=True)
class DopplerConsistency:
    """Result of :func:`doppler_consistency`."""

    ratio: float  # median |dpos| / |integral v dt| over the baselines
    ratio_min: float
    ratio_max: float
    n: int
    baseline_s: float

    def ok(self, tol: float) -> bool:
        return abs(self.ratio - 1.0) <= tol


def integrate_velocity(t: np.ndarray, vel_en: np.ndarray) -> np.ndarray:
    """Trapezoidal integral of a horizontal (east, north) velocity -> (N, 2) m."""
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(vel_en, dtype=np.float64)
    dt = np.diff(t)[:, None]
    step = dt * 0.5 * (v[:-1] + v[1:])
    return np.vstack([np.zeros((1, 2)), np.cumsum(step, axis=0)])


def velocity_en(vel_xyz: np.ndarray, frame: str) -> np.ndarray:
    """(N, 3) rtk_velocity vectors -> (N, 2) (east, north) in the given frame."""
    try:
        ie, in_ = VELOCITY_FRAMES[frame]
    except KeyError:
        raise ValueError(
            f"unknown rtk velocity frame {frame!r} (expected one of {sorted(VELOCITY_FRAMES)})"
        ) from None
    v = np.asarray(vel_xyz, dtype=np.float64)
    return np.column_stack([v[:, ie], v[:, in_]])


def doppler_consistency(
    t: np.ndarray,
    pos_en: np.ndarray,
    vel_en: np.ndarray,
    baseline_s: float = 3.0,
    min_displacement_m: float = 5.0,
) -> DopplerConsistency:
    """Compare projected-position displacement with Doppler-integrated displacement.

    Args:
        t: (N,) fix timestamps, seconds, ascending.
        pos_en: (N, 2) projected positions, metres (east, north).
        vel_en: (N, 2) horizontal velocity, m/s (east, north), same stamps.
        baseline_s: comparison baseline. Short on purpose (see module docstring).
        min_displacement_m: skip baselines the drone barely moved over --
            a stationary drone carries no scale information at all.

    Returns:
        A :class:`DopplerConsistency`; ``ratio`` is 1.0 for a correctly
        georeferenced track regardless of heading, speed or projection.
    """
    t = np.asarray(t, dtype=np.float64)
    pos = np.asarray(pos_en, dtype=np.float64)
    if len(t) != len(pos) or len(t) != len(vel_en):
        raise ValueError("doppler_consistency: t, pos_en and vel_en must be the same length")
    integ = integrate_velocity(t, vel_en)
    step = max(1, int(round(baseline_s / np.median(np.diff(t)))))
    if len(t) <= step:
        raise ValueError(
            f"doppler_consistency: baseline {baseline_s} s needs more than {step} fixes"
        )
    i = np.arange(0, len(t) - step)
    dp = np.linalg.norm(pos[i + step] - pos[i], axis=1)
    di = np.linalg.norm(integ[i + step] - integ[i], axis=1)
    keep = di >= min_displacement_m
    if not keep.any():
        raise ValueError("doppler_consistency: the drone never moved far enough to compare")
    r = dp[keep] / di[keep]
    return DopplerConsistency(
        ratio=float(np.median(r)),
        ratio_min=float(r.min()),
        ratio_max=float(r.max()),
        n=int(keep.sum()),
        baseline_s=float(baseline_s),
    )
