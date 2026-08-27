"""Dual bias accounting (T12, and 03-level-b section 5).

The satellite basemap has a georeferencing bias (measured in T09) -- typically
2-5 m CE90, sometimes more. That bias is the floor on the whole system's
accuracy: if we compare the matcher's estimate directly against RTK truth, we
measure the *map* error, not the *system* error.

So every metric is reported twice:

``with``    -- the raw error ``|est - gt|`` (includes the map bias);
``without`` -- the error after subtracting the measured bias, ``|est - bias - gt|``
               (the system's own error).

Both numbers go in the report. The difference between them is exactly the bias
contribution (T12-U-06): if the only error source is a constant basemap shift,
``with`` differs from ``without`` by exactly the bias vector.
"""

from __future__ import annotations

import numpy as np

DEFAULT_BIAS = (0.0, 0.0)


def shift_gt(
    gt_east: np.ndarray, gt_north: np.ndarray, bias: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Shift the ground truth by the basemap bias (used for the "without" view).

    If the map is offset by ``bias`` from truth, the matcher will report
    ``est ~= gt + bias``; comparing ``est`` against ``gt + bias`` isolates the
    system's own error from the map's error.
    """
    b_e, b_n = bias
    return gt_east + b_e, gt_north + b_n


def corrected_estimate(
    est_east: np.ndarray, est_north: np.ndarray, bias: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the measured bias from the estimate (equivalent to ``shift_gt``)."""
    b_e, b_n = bias
    return est_east - b_e, est_north - b_n


def dual_bias_tables(
    rec: np.ndarray, fix_metrics_fn, bias: tuple[float, float], terrain_breakdown_fn
) -> tuple[dict, dict]:
    """Return ``(with_bias, without_bias)`` metric tables, one per bias view.

    ``fix_metrics_fn`` and ``terrain_breakdown_fn`` take the same
    ``(rec, bias, bias_mode)`` signature; this helper calls both twice so the
    caller gets the full pair with a single call.
    """
    with_bias = {
        "fix_level": fix_metrics_fn(rec, bias=bias, bias_mode="with"),
        "by_terrain": terrain_breakdown_fn(rec, bias=bias, bias_mode="with"),
    }
    without_bias = {
        "fix_level": fix_metrics_fn(rec, bias=bias, bias_mode="without"),
        "by_terrain": terrain_breakdown_fn(rec, bias=bias, bias_mode="without"),
    }
    return with_bias, without_bias
