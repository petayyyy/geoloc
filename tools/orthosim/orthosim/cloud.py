"""Livox Avia point-cloud synthesis (T11).

Models the geometry that the downstream pipeline actually depends on:

- strict FOV 70.4 deg (horizontal) x 77.2 deg (vertical) -- the coverage-deficit
  test cases (A-DSM-01) come straight out of this number;
- ~240k points/s;
- range noise sigma ~2 cm;
- motion distortion (points are generated along the real trajectory over the
  accumulation interval, otherwise deskew is never exercised);
- water dropout (no returns where the semantic layer says water).

The exact non-repetitive rosette pattern is approximated by a uniform draw over
the FOV rectangle (documented as such in the T11 card); the pattern is not what
the tests measure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geopack import CLASS_WATER

AVIA_FOV_H_DEG = 70.4
AVIA_FOV_V_DEG = 77.2
AVIA_POINTS_PER_SEC = 240_000


@dataclass
class AviaSpec:
    fov_h_deg: float = AVIA_FOV_H_DEG
    fov_v_deg: float = AVIA_FOV_V_DEG
    points_per_sec: float = AVIA_POINTS_PER_SEC
    range_max_m: float = 190.0
    range_sigma_m: float = 0.02


def avia_rays(n: int, rng: np.random.Generator, spec: AviaSpec) -> np.ndarray:
    """``(n, 3)`` unit rays in the lidar body frame, optical axis along -Z.

    The FOV rectangle is parametrised by two half-angles about the nadir axis:
    ``tan(half_h)`` along +X and ``tan(half_v)`` along +Y. This reproduces the
    141 x 160 m footprint at 100 m AGL exactly (2*h*tan(half)).
    """
    fh = np.deg2rad(spec.fov_h_deg) / 2.0
    fv = np.deg2rad(spec.fov_v_deg) / 2.0
    tx = rng.uniform(-fh, fh, n)
    ty = rng.uniform(-fv, fv, n)
    d = np.stack([np.tan(tx), np.tan(ty), -np.ones(n)], axis=1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d


def _march_hit(C: np.ndarray, dirs: np.ndarray, z_fn, s_min: float, s_max: float, steps: int = 96):
    n = len(dirs)
    s_lo = np.full(n, s_min)
    s_hi = np.full(n, s_max)
    hit = np.zeros(n, dtype=bool)
    ds = (s_max - s_min) / steps
    for _ in range(steps):
        s_cur = s_lo + ds
        p = C + dirs * s_cur[:, None]
        zt = z_fn(p[:, 0], p[:, 1])
        below = np.isfinite(zt) & (p[:, 2] <= zt)
        newly = below & ~hit
        s_hi[newly] = s_cur[newly]
        hit |= below
        s_lo = np.where(hit, s_lo, s_cur)
    for _ in range(8):
        s_mid = 0.5 * (s_lo + s_hi)
        p = C + dirs * s_mid[:, None]
        zt = z_fn(p[:, 0], p[:, 1])
        below = np.isfinite(zt) & (p[:, 2] <= zt)
        s_hi = np.where(below, s_mid, s_hi)
        s_lo = np.where(below, s_lo, s_mid)
    return 0.5 * (s_lo + s_hi), hit


def synthesize_cloud(
    scene,
    trajectory,
    t0: float,
    duration: float,
    n_points: int,
    rng: np.random.Generator,
    spec: AviaSpec | None = None,
    *,
    motion_distortion: bool = True,
    drop_water: bool = True,
    reflectivity: float = 0.9,
    dirs_body: np.ndarray | None = None,
) -> np.ndarray:
    """Synthesize one accumulation interval of Avia points.

    Args:
        trajectory: callable ``t -> (C, R)`` returning the lidar position ``(3,)``
            and body->world rotation ``(3,3)`` at time ``t``.
        t0, duration: the accumulation interval ``[t0, t0 + duration]``.
        n_points: number of rays cast (equals the number of points when no
            dropout applies).
        reflectivity: 0..1, scales the per-ray return probability (water and
            dark/mirror surfaces are modelled as ``reflectivity -> 0``).
        dirs_body: optional ``(n_points, 3)`` override for the body-frame rays
            (used by tests to isolate a single ray direction).

    Returns:
        ``(M, 4)`` float64 array of ``(x, y, z, t_rel)`` in the world (UTM) frame,
        ``t_rel`` in ``[0, duration]``. ``M <= n_points``.
    """
    spec = spec or AviaSpec()
    z_fn = lambda e, n: scene.z(e, n)  # noqa: E731

    # Ray times: uniform over the interval when motion distortion is on, else all
    # at t0 (so a moving platform leaves a predictable v*dt smear, T11-U-03).
    t_rel = rng.uniform(0.0, duration, n_points) if motion_distortion else np.zeros(n_points)
    t_abs = t0 + t_rel

    # Body-frame rays; motion distortion applies the *pose at each point's time*.
    dirs_body = dirs_body if dirs_body is not None else avia_rays(n_points, rng, spec)

    # Gather poses. Trajectories are expected to be cheap callables; we sample
    # once per point (this is the honest model of a deskew test).
    dirs = np.empty_like(dirs_body)
    C_arr = np.empty((n_points, 3))
    if motion_distortion:
        for i in range(n_points):
            C, R = trajectory(float(t_abs[i]))
            C_arr[i] = C
            dirs[i] = dirs_body[i] @ R.T
    else:
        C0, R0 = trajectory(t0)
        C_arr[:] = C0
        dirs[:] = dirs_body @ R0.T

    agl = C_arr[0, 2] - np.nanmin(scene.z(C_arr[:, 0], C_arr[:, 1]))
    agl = max(30.0, float(agl)) if np.isfinite(agl) else 100.0
    s, hit = _march_hit(C_arr, dirs, z_fn, 0.05 * agl, 2.0 * agl)

    # Range gate + reflectivity/water dropout.
    keep = hit & (s <= spec.range_max_m) & (s > 0.0)
    if drop_water and scene.semantic is not None:
        p = C_arr + dirs * s[:, None]
        sem = scene.semantic.sample(p[:, 0], p[:, 1])
        water = np.isfinite(sem) & (sem > CLASS_WATER - 0.5) & (sem < CLASS_WATER + 0.5)
        keep &= ~water
    if reflectivity < 1.0:
        keep &= rng.random(n_points) < reflectivity

    idx = np.nonzero(keep)[0]
    points = C_arr[idx] + dirs[idx] * s[idx, None]

    # Range noise ~2 cm.
    if spec.range_sigma_m > 0.0:
        noise = rng.normal(0.0, spec.range_sigma_m, size=len(idx))
        points += dirs[idx] * noise[:, None]

    return np.column_stack([points, t_rel[idx]])


def flat_ground_hits(C: np.ndarray, dirs: np.ndarray, z_ground: float = 0.0) -> np.ndarray:
    """Analytic ground intersections for rays over a flat plane at ``z_ground``."""
    s = (z_ground - C[:, 2]) / dirs[:, 2]
    valid = dirs[:, 2] < 0.0
    s = np.where(valid, s, np.nan)
    return C + dirs * s[:, None]
