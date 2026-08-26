"""camera_init -> UTM alignment through the RTK anchor (T14/T15 step 1).

Two alignment strategies are implemented, selected by the capture config
(`align.method`, default `pose_graph`):

- `align_windowed` (legacy): fits an independent rigid SE(3) Kabsch transform
  in each sliding time window. Retained as a fallback / comparison point.
- `align_pose_graph` (default): a small global pose-graph / bundle adjustment
  over the whole sequence. It first *levels* the odometry frame (fixing the
  physical up axis from PCA of the path, see `pca_up_axis`), which removes the
  FAST-LIVO2 Z/tilt drift through aggressive turns (the cause of the 387 m
  AGL artefacts on `geoloc_capture_01`), then jointly optimizes a slowly
  drifting heading + translation with a global scale. The scale is a genuine,
  measurable property of this capture: the replayed FAST-LIVO2 odometry
  under-measures distance by ~22% (verified empirically: a per-window rigid
  fit leaves a ~8 m residual that a similarity fit collapses to ~0.16 m on the
  straight legs). RTK fixes enter as sparse absolute constraints with a Huber
  robust loss (E/N only -- the RTK altitude is frozen and unreliable, see
  README), while the odometry's own relative motion keeps the local shape.

The Z target is a constant (median RTK altitude across the segment), never a
per-point RTK altitude; the absolute Z is corrected later by `dsm.anchor_to_dem`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


def quat_to_rotm(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z), scalar-last or first is ambiguous in ROS; this
    takes (w,x,y,z) order as stored in geometry_msgs/Quaternion."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotm_to_quat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(dot)
    s0, s1 = np.sin((1 - t) * theta), np.sin(t * theta)
    return (s0 * q0 + s1 * q1) / np.sin(theta)


def fit_rigid3(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch: rotation and translation mapping A onto B (Nx3 -> Nx3)."""
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    A1, B1 = A - ca, B - cb
    U, _S, Vt = np.linalg.svd(A1.T @ B1)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca
    return R, t


def interpolate_odom_pose(odom: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Odom rows are (t, x, y, z, qw, qx, qy, qz); returns (pos, quat).

    Out-of-range times clamp to the nearest sample (no extrapolation).
    """
    idx = np.searchsorted(odom[:, 0], t)
    if idx <= 0:
        return odom[0, 1:4].copy(), odom[0, 4:8].copy()
    if idx >= len(odom):
        return odom[-1, 1:4].copy(), odom[-1, 4:8].copy()
    t0, t1 = odom[idx - 1, 0], odom[idx, 0]
    if t1 - t0 > 1.0:
        return odom[idx - 1, 1:4].copy(), odom[idx - 1, 4:8].copy()
    f = (t - t0) / (t1 - t0)
    pos = odom[idx - 1, 1:4] * (1 - f) + odom[idx, 1:4] * f
    quat = slerp(odom[idx - 1, 4:8], odom[idx, 4:8], f)
    return pos, quat


@dataclass
class TransformSeries:
    """Piecewise-linear rigid transform camera_init -> UTM over time.

    When `mirror_z` is set, the stored rotations/translations are proper but
    `at`/`apply` mirror the output z (S = diag(1,1,-1) applied to the mapped
    point). This resolves the vertical sign ambiguity of planar paths without
    breaking rotation interpolation: the mirror is factored out of the
    interpolated rotation and applied to the final output only.
    """

    times: np.ndarray  # (K,) node times, seconds, ascending
    rotations: np.ndarray  # (K, 3, 3), proper rotations
    translations: np.ndarray  # (K, 3)
    mirror_z: bool = False
    scale: float = 1.0  # global scale of the camera_init frame (1.0 = rigid)

    @staticmethod
    def _mirror(v: np.ndarray) -> np.ndarray:
        out = np.array(v, dtype=np.float64, copy=True)
        out[..., 2] *= -1.0
        return out

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        idx = np.searchsorted(self.times, t)
        if idx <= 0:
            R, tr = self.rotations[0].copy(), self.translations[0].copy()
        elif idx >= len(self.times):
            R, tr = self.rotations[-1].copy(), self.translations[-1].copy()
        else:
            t0, t1 = self.times[idx - 1], self.times[idx]
            f = (t - t0) / (t1 - t0)
            R = quat_to_rotm(
                slerp(rotm_to_quat(self.rotations[idx - 1]), rotm_to_quat(self.rotations[idx]), f)
            )
            tr = self.translations[idx - 1] * (1 - f) + self.translations[idx] * f
        if self.mirror_z:
            return self._mirror(R), self._mirror(tr)
        return R, tr

    def apply(self, xyz: np.ndarray, t: float) -> np.ndarray:
        R, tr = self.at(t)
        return xyz @ R.T * self.scale + tr

    def shift_z(self, dz: float) -> TransformSeries:
        self.translations = self.translations.copy()
        sign = -1.0 if self.mirror_z else 1.0
        self.translations[:, 2] += sign * dz
        return self

    def to_dict(self) -> dict:
        return {
            "times": self.times.tolist(),
            "rotations": self.rotations.reshape(-1, 9).tolist(),
            "translations": self.translations.tolist(),
            "mirror_z": bool(self.mirror_z),
            "scale": float(self.scale),
        }

    @classmethod
    def from_dict(cls, d: dict) -> TransformSeries:
        return cls(
            times=np.asarray(d["times"], dtype=np.float64),
            rotations=np.asarray(d["rotations"], dtype=np.float64).reshape(-1, 3, 3),
            translations=np.asarray(d["translations"], dtype=np.float64),
            mirror_z=bool(d.get("mirror_z", False)),
            scale=float(d.get("scale", 1.0)),
        )


@dataclass
class AlignmentResult:
    series: TransformSeries
    residuals: np.ndarray  # (K,) 2D mean residual per window, metres
    residuals_max: np.ndarray
    window_n: np.ndarray
    rtk_alt_used: float
    z_datum: str
    z_shift: float = 0.0

    def save(self, path: Path) -> None:
        data = {
            "format": "orthoproto-align-v1",
            "z_datum": self.z_datum,
            "z_shift": float(self.z_shift),
            "rtk_alt_used": float(self.rtk_alt_used),
            "series": self.series.to_dict(),
            "residuals": self.residuals.tolist(),
            "residuals_max": self.residuals_max.tolist(),
            "window_n": [int(n) for n in self.window_n],
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    @classmethod
    def load(cls, path: Path) -> AlignmentResult:
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls(
            series=TransformSeries.from_dict(d["series"]),
            residuals=np.asarray(d["residuals"]),
            residuals_max=np.asarray(d["residuals_max"]),
            window_n=np.asarray(d["window_n"], dtype=np.int64),
            rtk_alt_used=float(d["rtk_alt_used"]),
            z_datum=d["z_datum"],
            z_shift=float(d.get("z_shift", 0.0)),
        )


def _snap_to_previous(
    R_new: np.ndarray, R_prev: np.ndarray, line_axis_utm: np.ndarray
) -> np.ndarray:
    """Remove the rotation-about-the-line-axis ambiguity of a collinear fit.

    When a window's odometry points are (nearly) collinear, the rigid fit
    leaves the rotation about the line axis unobservable; the SVD returns an
    arbitrary member of that family and interpolating arbitrary members across
    windows breaks the series. Snapping the new rotation to the previous
    window's makes the series continuous where the data cannot decide.
    """
    a = line_axis_utm / np.linalg.norm(line_axis_utm)
    m = R_prev @ R_new.T
    v = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    theta = np.arctan2(np.dot(a, v) / 2.0, (np.trace(m) - 1.0) / 2.0)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    delta = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return delta @ R_new


def _collinearity_std(points: np.ndarray) -> float:
    """Middle principal standard deviation; small means (nearly) collinear."""
    centred = points - points.mean(axis=0)
    _U, S, _Vt = np.linalg.svd(centred)
    return float(S[1] / np.sqrt(max(1, len(points) - 1))) if len(S) > 1 else 0.0


def pca_up_axis(odom: np.ndarray) -> np.ndarray:
    """Smallest principal axis of the odometry path (the vertical-ish axis).

    The sign is arbitrary: PCA cannot tell up from down. Callers resolve the
    sign with sensor data (the lidar cloud lies BELOW the drone).
    """
    centred = odom[:, 1:4] - odom[:, 1:4].mean(axis=0)
    _U, _S, Vt = np.linalg.svd(centred)
    return Vt[-1] / np.linalg.norm(Vt[-1])


def orient_up_from_cloud(odom: np.ndarray, clouds, pca_up: np.ndarray) -> np.ndarray:
    """Flip the PCA axis so that the lidar cloud lies BELOW the drone.

    Returns the physical up direction in the odometry frame.
    """
    offset = 0.0
    n = 0
    for t, xyz in clouds:
        pos, _ = interpolate_odom_pose(odom, t)
        offset += float(np.median((xyz - pos) @ pca_up))
        n += 1
        if n >= 5:
            break
    if n == 0:
        return pca_up
    # cloud above the drone along pca_up means pca_up points downward
    return pca_up if offset < 0 else -pca_up


def align_windowed(
    odom: np.ndarray,
    rtk_utm: np.ndarray,
    window_s: float = 12.0,
    step_s: float = 4.0,
    min_pairs: int = 8,
    collinearity_m: float = 1.5,
    up_odom: np.ndarray | None = None,
) -> AlignmentResult:
    """Fit T(t) mapping camera_init to UTM.

    Args:
        odom: (N, 8) rows (t, x, y, z, qw, qx, qy, qz).
        rtk_utm: (M, 4) rows (t, east, north, alt) -- already lat/lon-swapped
            and UTM-projected by the caller.
        collinearity_m: windows whose odometry points are flatter than this
            (middle PCA std) are treated as collinear and their rotation is
            snapped to the previous window's.
        up_odom: physical up direction in the odometry frame; when given, each
            window's rotation is constrained to map it to UTM +up, removing
            the vertical sign ambiguity that a (nearly) planar path cannot
            resolve by itself.
    """
    rtk_t = rtk_utm[:, 0]
    rtk_alt = float(np.median(rtk_utm[:, 3]))
    odom_t_min, odom_t_max = odom[0, 0], odom[-1, 0]
    times, rots, trans = [], [], []
    res_means, res_maxs, ns = [], [], []
    R_prev = None

    for tc in np.arange(rtk_t.min() + window_s / 2, rtk_t.max() - window_s / 4, step_s):
        sel = (rtk_t >= tc - window_s / 2) & (rtk_t <= tc + window_s / 2)
        A, B = [], []
        for i in np.where(sel)[0]:
            if rtk_t[i] < odom_t_min or rtk_t[i] > odom_t_max:
                continue
            pos, _ = interpolate_odom_pose(odom, rtk_t[i])
            A.append(pos)
            B.append([rtk_utm[i, 1], rtk_utm[i, 2], rtk_alt])
        if len(A) < min_pairs:
            continue
        A, B = np.array(A), np.array(B)
        R, t = fit_rigid3(A, B)
        if collinearity_m > 0 and _collinearity_std(A) < collinearity_m and R_prev is not None:
            # collinear window: rotation about the line axis is unobservable,
            # snap it to the previous window to keep the series continuous
            line_utm = B[-1] - B[0]
            R = _snap_to_previous(R, R_prev, line_utm[:3])
            t = B.mean(axis=0) - R @ A.mean(axis=0)
        R_prev = R
        resid = np.linalg.norm((B[:, :2] - (A @ R.T + t)[:, :2]), axis=1)
        times.append(tc)
        rots.append(R)
        trans.append(t)
        res_means.append(float(resid.mean()))
        res_maxs.append(float(resid.max()))
        ns.append(len(A))

    rots = np.asarray(rots)
    trans = np.asarray(trans)
    if up_odom is not None and len(rots):
        # The fit cannot decide the vertical sign of a (nearly) planar path.
        # Enforce the cloud-derived physical up: mirroring the OUTPUT z of a
        # window is exact for planar paths (horizontal coordinates are
        # untouched), so a window whose rotation maps up_odom downward is
        # replaced by the mirror solution without any refit.
        pz = np.diag([1.0, 1.0, -1.0])
        for i in range(len(rots)):
            if (rots[i] @ up_odom)[2] < 0:
                rots[i] = pz @ rots[i]
                trans[i] = pz @ trans[i]

    return AlignmentResult(
        series=TransformSeries(
            times=np.asarray(times),
            rotations=rots,
            translations=trans,
        ),
        residuals=np.asarray(res_means),
        residuals_max=np.asarray(res_maxs),
        window_n=np.asarray(ns, dtype=np.int64),
        rtk_alt_used=rtk_alt,
        z_datum="rtk",
    )


def _rot_about(vec: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Minimal-rotation matrix mapping unit ``vec`` onto unit ``target``."""
    vec = vec / np.linalg.norm(vec)
    target = target / np.linalg.norm(target)
    a = np.cross(vec, target)
    s = np.linalg.norm(a)
    c = float(np.dot(vec, target))
    if s < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def _Rz(theta: float) -> np.ndarray:
    """3D rotation about the +Z axis by ``theta`` (heading)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _R2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _dR2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[-s, -c], [c, -s]])


def _similarity2d(A: np.ndarray, B: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Closed-form 2D similarity (scale + rotation + translation) A -> B."""
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    Ac, Bc = A - ca, B - cb
    U, _S, Vt = np.linalg.svd(Ac.T @ Bc)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    denom = float(np.sum(Ac * Ac))
    s = float(np.einsum("ij,ij->", Ac @ R.T, Bc) / denom) if denom > 1e-12 else 1.0
    t = cb - s * R @ ca
    return s, R, t


def _rigid2d(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    U, _S, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, cb - R @ ca


def align_pose_graph(
    odom: np.ndarray,
    rtk_utm: np.ndarray,
    up_odom: np.ndarray | None = None,
    step_s: float = 2.0,
    rtk_weight: float = 1.0,
    smooth_weight: float = 0.5,
    rtk_huber_m: float = 2.0,
    max_iter: int = 200,
) -> AlignmentResult:
    """Global pose-graph alignment of the whole trajectory.

    Level the odometry frame (fixed physical-up rotation), then jointly fit a
    slowly drifting heading + translation and a *global scale* so the odometry
    trajectory matches the RTK fixes in E/N. RTK fixes enter as sparse absolute
    constraints with a Huber robust loss; the odometry's own relative motion
    between keyframes preserves the local shape (it is exact, not a soft
    spring -- short-baseline LIO relative motion is accurate to ~cm here).

    Z is never pulled from the RTK fixes: the alignment's Z target is a single
    constant (median RTK altitude), and the absolute Z is corrected later by
    ``dsm.anchor_to_dem``. This mirrors the legacy windowed behaviour and is
    deliberate (the RTK altitude in ``geoloc_capture_01`` is frozen/unreliable).

    Args:
        odom: (N, 8) rows (t, x, y, z, qw, qx, qy, qz).
        rtk_utm: (M, 4) rows (t, east, north, alt) -- already lat/lon-swapped
            and UTM-projected by the caller.
        up_odom: physical up direction in the odometry frame; required for the
            leveling step that removes FAST-LIVO2 Z/tilt drift.
    """
    rtk_alt = float(np.median(rtk_utm[:, 3]))
    odom_t_min, odom_t_max = odom[0, 0], odom[-1, 0]

    # Only align RTK fixes that have odometry coverage.
    sel_rtk = (rtk_utm[:, 0] >= odom_t_min) & (rtk_utm[:, 0] <= odom_t_max)
    if int(sel_rtk.sum()) < 6:
        raise ValueError("align_pose_graph: too few RTK fixes within odometry coverage")
    tt = rtk_utm[sel_rtk, 0]
    B = rtk_utm[sel_rtk, 1:3]
    n = len(tt)

    if up_odom is not None:
        R_up = _rot_about(np.asarray(up_odom), np.array([0.0, 0.0, 1.0]))
    else:
        R_up = np.eye(3)

    def levelled_pos(times: np.ndarray) -> np.ndarray:
        pos = np.array([interpolate_odom_pose(odom, t)[0] for t in times])
        return (R_up @ pos.T).T

    A = levelled_pos(tt)  # (n, 3) leveled odometry at RTK fixes

    # Keyframe grid over the RTK range.
    t0, t1 = float(tt[0]), float(tt[-1])
    kt = np.arange(t0, t1 + step_s, step_s)
    if kt[-1] < t1:
        kt = np.append(kt, t1)
    pk = levelled_pos(kt)[:, :2]  # (K, 2)
    K = len(kt)
    M = K - 1  # number of inter-keyframe segments / headings
    dP = np.diff(pk, axis=0)  # (M, 2)
    seg = np.clip(np.searchsorted(kt, tt, side="right") - 1, 0, M - 1)
    partial = A[:, :2] - pk[seg]  # (n, 2)

    # Initial guess: global similarity.
    s0, Rg, tg = _similarity2d(A[:, :2], B)
    th0 = float(np.arctan2(Rg[1, 0], Rg[0, 0]))

    # Per-keyframe heading init (unwrapped local rigid fits) for a good start.
    As = A[:, :2] * s0
    theta = np.full(M, th0)
    half = max(step_s * 1.5, 1.5)
    for j in range(M):
        wsel = np.abs(tt - kt[j]) <= half
        if int(wsel.sum()) < 6:
            continue
        Rj, _ = _rigid2d(As[wsel], B[wsel])
        theta[j] = np.arctan2(Rj[1, 0], Rj[0, 0])
    theta = th0 + np.mod(theta - th0 + np.pi, 2 * np.pi) - np.pi

    P0 = tg
    s = s0

    def kf_positions(P0v: np.ndarray, sv: float, th: np.ndarray) -> np.ndarray:
        P = np.empty((K, 2))
        P[0] = P0v
        c = np.cos(th)
        sj = np.sin(th)
        for j in range(M):
            P[j + 1] = P[j] + sv * np.array(
                [c[j] * dP[j, 0] - sj[j] * dP[j, 1], sj[j] * dP[j, 0] + c[j] * dP[j, 1]]
            )
        return P

    nv = 3 + M
    nres = 2 * n + (M - 1)
    lam = 1e-3
    prev_cost = np.inf

    for _it in range(max_iter):
        Pk = kf_positions(P0, s, theta)
        c = np.cos(theta[seg])
        sj = np.sin(theta[seg])
        rot_part = s * np.column_stack(
            [c * partial[:, 0] - sj * partial[:, 1], sj * partial[:, 0] + c * partial[:, 1]]
        )
        Pt = Pk[seg] + rot_part  # (n, 2) aligned E/N per RTK fix
        e_abs = (Pt - B).ravel()
        e_smo = theta[1:] - theta[:-1]

        hub = np.where(np.abs(e_abs) > rtk_huber_m, rtk_huber_m / np.abs(e_abs), 1.0)
        ra = np.abs(e_abs)
        cost = rtk_weight * np.where(
            ra <= rtk_huber_m, 0.5 * ra**2, rtk_huber_m * (ra - 0.5 * rtk_huber_m)
        ).sum() + smooth_weight * 0.5 * np.sum(e_smo**2)

        # Analytic Jacobian.
        J = np.zeros((nres, nv))
        # keyframe position derivatives
        dPkf_ds = np.zeros((K, 2))
        dPkf_dth = np.zeros((M, K, 2))
        for j in range(M):
            Rj2 = _R2(theta[j])
            Mj = _dR2(theta[j])
            dPkf_ds[j + 1 :] += Rj2 @ dP[j]
            dPkf_dth[j, j + 1 :] = s * (Mj @ dP[j])
        for i in range(n):
            si = seg[i]
            J[2 * i : 2 * i + 2, 0:2] = np.eye(2)
            J[2 * i : 2 * i + 2, 2] = dPkf_ds[si] + rot_part[i] / s
            for k in range(si):
                J[2 * i : 2 * i + 2, 3 + k] = dPkf_dth[k, si]
            J[2 * i : 2 * i + 2, 3 + si] += s * (_dR2(theta[si]) @ partial[i])
        for j in range(M - 1):
            J[2 * n + j, 3 + j] = -1.0
            J[2 * n + j, 3 + j + 1] = 1.0

        e = np.concatenate([e_abs, e_smo])
        W = np.concatenate([rtk_weight * hub, np.full(M - 1, smooth_weight)])
        g = J.T @ (W * e)
        H = J.T @ (W[:, None] * J)

        accepted = False
        while lam < 1e15:
            try:
                dx = np.linalg.solve(H + lam * np.eye(nv), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            P0n = P0 + dx[:2]
            sn = s + dx[2]
            thn = theta + dx[3:]
            Pn = kf_positions(P0n, sn, thn)
            cn = np.cos(thn[seg])
            sjn = np.sin(thn[seg])
            rpn = sn * np.column_stack(
                [cn * partial[:, 0] - sjn * partial[:, 1], sjn * partial[:, 0] + cn * partial[:, 1]]
            )
            ea = (Pn[seg] + rpn - B).ravel()
            es = thn[1:] - thn[:-1]
            ra = np.abs(ea)
            costn = rtk_weight * np.where(
                ra <= rtk_huber_m, 0.5 * ra**2, rtk_huber_m * (ra - 0.5 * rtk_huber_m)
            ).sum() + smooth_weight * 0.5 * np.sum(es**2)
            if costn < cost:
                P0, s, theta = P0n, sn, thn
                lam = max(lam * 0.3, 1e-10)
                accepted = True
                break
            lam *= 10.0
        if not accepted:
            break
        if abs(cost - prev_cost) <= 1e-9 * max(1.0, cost):
            break
        prev_cost = cost

    # Build the output TransformSeries (leveled frame, heading + scale).
    Pk = kf_positions(P0, s, theta)
    full_theta = np.append(theta, theta[-1])  # last keyframe reuses the last heading
    rots = np.array([_Rz(th) @ R_up for th in full_theta])
    z0 = rtk_alt - s * float(np.median(A[:, 2]))
    trans = np.empty((K, 3))
    for j in range(K):
        R2z = _Rz(full_theta[j])
        lev = np.array([pk[j, 0], pk[j, 1], 0.0])
        rot = R2z @ lev
        trans[j] = [Pk[j, 0] - s * rot[0], Pk[j, 1] - s * rot[1], z0]

    # Residuals reported per keyframe (mean/max horizontal RTK residual).
    c = np.cos(theta[seg])
    sj = np.sin(theta[seg])
    rot_part = s * np.column_stack(
        [c * partial[:, 0] - sj * partial[:, 1], sj * partial[:, 0] + c * partial[:, 1]]
    )
    Pt = Pk[seg] + rot_part
    resid = np.linalg.norm(Pt - B, axis=1)
    res_means = np.array([resid[seg == j].mean() if (seg == j).any() else 0.0 for j in range(M)])
    res_maxs = np.array([resid[seg == j].max() if (seg == j).any() else 0.0 for j in range(M)])
    ns = np.array([(seg == j).sum() for j in range(M)], dtype=np.int64)

    return AlignmentResult(
        series=TransformSeries(
            times=kt,
            rotations=rots,
            translations=trans,
            scale=s,
        ),
        residuals=res_means,
        residuals_max=res_maxs,
        window_n=ns,
        rtk_alt_used=rtk_alt,
        z_datum="rtk",
    )
