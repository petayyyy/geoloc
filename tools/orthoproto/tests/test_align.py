"""align.py tests: rigid fit recovery and the windowed series on synthetic data."""

import numpy as np

from orthoproto.align import (
    TransformSeries,
    align_pose_graph,
    align_windowed,
    fit_rigid3,
    quat_to_rotm,
    rotm_to_quat,
)


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def test_fit_rigid3_exact():
    rng = np.random.default_rng(1)
    R = _rot(np.array([0.3, -0.2, 0.9]), 0.7)
    t = np.array([572800.0, 4964000.0, 1155.0])
    A = rng.normal(size=(100, 3))
    B = A @ R.T + t
    R2, t2 = fit_rigid3(A, B)
    assert np.allclose(R2, R, atol=1e-12)
    assert np.allclose(t2, t, atol=1e-9)


def test_quat_roundtrip():
    rng = np.random.default_rng(2)
    for _ in range(10):
        R = _rot(rng.normal(size=3), rng.uniform(-np.pi, np.pi))
        q = rotm_to_quat(R)
        R2 = quat_to_rotm(q)
        assert np.allclose(R2, R, atol=1e-12)


def _synthetic_path(rng: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RTK fixes (t, e, n, alt) on an out-and-back route with noise.

    The route carries a sinusoidal cross-track wiggle and vertical variation
    so that the windowed rigid fits are well-conditioned (a perfectly
    straight line leaves the rotation about the line axis unobservable; see
    test_windowed_alignment_collinear_snaps).
    """
    rtk = []
    t = 0.0
    # out leg: 40 s north at 12 m/s from (0,0), with a 40 m cross-track drift
    while t < 40.0:
        s = 12.0 * t
        e = 0.03 * s + 18.0 * np.sin(2 * np.pi * t / 40.0) + rng.normal(0, 0.05)
        n = s + rng.normal(0, 0.05)
        alt = 1155.6 + 6.0 * np.sin(2 * np.pi * t / 40.0) + rng.normal(0, 0.02)
        rtk.append((t, e, n, alt))
        t += rng.uniform(0.28, 0.32)
    # 180 deg turn (constant-speed semicircle) around the endpoint
    t_end = 40.0
    while t < 52.0:
        s = 12.0 * (t - t_end)
        e = 0.03 * 480.0 + 60.0 * np.sin(s / 60.0 * np.pi)
        n = 480.0 + 60.0 * (1.0 - np.cos(s / 60.0 * np.pi))
        rtk.append((t, e, n, 1155.6 + rng.normal(0, 0.02)))
        t += rng.uniform(0.28, 0.32)
    # return leg south at 12 m/s
    e0, n0 = rtk[-1][1], rtk[-1][2]
    t_ret = t
    while t < 64.0:
        s = 12.0 * (t - t_ret)
        e = e0 - 0.03 * s + rng.normal(0, 0.05)
        n = n0 - s + rng.normal(0, 0.05)
        rtk.append((t, e, n, 1155.6 + rng.normal(0, 0.02)))
        t += rng.uniform(0.28, 0.32)
    rtk = np.asarray(rtk)

    R_true = _rot(np.array([0.2, -0.1, 0.95]), 0.35)
    t_true = rtk[0, 1:4].copy()
    odom = []
    for i in range(0, len(rtk), 2):
        p = (rtk[i, 1:4] - t_true) @ R_true.T
        odom.append((rtk[i, 0], p[0], p[1], p[2], 1.0, 0.0, 0.0, 0.0))
    return rtk, np.asarray(odom), R_true, t_true


def test_windowed_alignment_recovers_transform():
    rng = np.random.default_rng(42)
    rtk, odom, R_true, t_true = _synthetic_path(rng)
    res = align_windowed(odom, rtk, window_s=12.0, step_s=4.0, min_pairs=8)

    # straight legs: residual below RTK noise level
    straight_windows = (res.series.times < 34.0) | (res.series.times > 56.0)
    assert res.residuals[straight_windows].mean() < 0.15

    # the series maps odom poses back to UTM within ~0.3 m on straight legs
    errors = []
    for i in range(0, len(odom), 10):
        t = odom[i, 0]
        if 34.0 < t < 56.0:
            continue
        p_utm = res.series.apply(odom[i, 1:4][None, :], t)[0]
        p_true = rtk[np.argmin(np.abs(rtk[:, 0] - t)), 1:3]
        errors.append(np.linalg.norm(p_utm[:2] - p_true))
    assert np.mean(errors) < 0.3


def test_windowed_alignment_collinear_snaps():
    """A perfectly straight flight must not produce wild interpolated transforms."""
    rng = np.random.default_rng(43)
    rtk = []
    t = 0.0
    while t < 60.0:
        s = 12.0 * t
        rtk.append((t, 0.05 * s + rng.normal(0, 0.05), s + rng.normal(0, 0.05), 1155.6))
        t += rng.uniform(0.28, 0.32)
    rtk = np.asarray(rtk)
    R_true = _rot(np.array([0.2, -0.1, 0.95]), 0.35)
    t_true = rtk[0, 1:4].copy()
    odom = []
    for i in range(0, len(rtk), 2):
        p = (rtk[i, 1:4] - t_true) @ R_true.T
        odom.append((rtk[i, 0], p[0], p[1], p[2], 1.0, 0.0, 0.0, 0.0))
    odom = np.asarray(odom)
    res = align_windowed(odom, rtk, window_s=12.0, step_s=4.0, min_pairs=8)
    assert len(res.series.times) >= 5
    for i in range(0, len(odom), 5):
        p_utm = res.series.apply(odom[i, 1:4][None, :], odom[i, 0])[0]
        p_true = rtk[np.argmin(np.abs(rtk[:, 0] - odom[i, 0])), 1:3]
        assert np.linalg.norm(p_utm[:2] - p_true) < 0.3


def test_transform_series_interpolation():
    ts = TransformSeries(
        times=np.array([0.0, 10.0]),
        rotations=np.stack([np.eye(3), _rot(np.array([0.0, 0.0, 1.0]), 0.5)]),
        translations=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
    )
    R, t = ts.at(5.0)
    assert np.allclose(t, [5.0, 0.0, 0.0])
    R, t = ts.at(-3.0)
    assert np.allclose(R, np.eye(3))
    R, t = ts.at(100.0)
    assert np.allclose(t, [10.0, 0.0, 0.0])


def _synthetic_scaled_path(
    rng, scale=1.2, heading=0.4, outliers=False, up_odom=None
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Odometry on a leveled out-and-back route + a known similarity to RTK.

    The odometry frame's physical up is ``up_odom`` (default +Z, i.e. already
    leveled). RTK is the *leveled* odometry mapped by ``rtk = scale * R(heading)
    @ leveled + t`` plus small noise (and, when ``outliers``, a few gross 50 m
    RTK outliers), with a frozen altitude.
    """
    from orthoproto.align import _rot_about

    up_odom = np.array([0.0, 0.0, 1.0]) if up_odom is None else np.asarray(up_odom)
    up_odom = up_odom / np.linalg.norm(up_odom)
    R_up = _rot_about(up_odom, np.array([0.0, 0.0, 1.0]))

    pts = []
    t = 0.0
    while t < 40.0:
        d = 12.0 * t
        e = 0.03 * d + 18.0 * np.sin(2 * np.pi * t / 40.0)
        n = d
        pts.append((t, e, n))
        t += 0.3
    t_end = 40.0
    while t < 52.0:
        d = 12.0 * (t - t_end)
        e = 0.03 * 480.0 + 60.0 * np.sin(d / 60.0 * np.pi)
        n = 480.0 + 60.0 * (1.0 - np.cos(d / 60.0 * np.pi))
        pts.append((t, e, n))
        t += 0.3
    e0, n0 = pts[-1][1], pts[-1][2]
    t_ret = t
    while t < 64.0:
        d = 12.0 * (t - t_ret)
        e = e0 - 0.03 * d
        n = n0 - d
        pts.append((t, e, n))
        t += 0.3
    pts = np.asarray(pts)  # (M, 3) t, e, n (physical, altitude 80)

    R = _rot(np.array([0.0, 0.0, 1.0]), heading)
    t_true = np.array([pts[0, 1], pts[0, 2], 80.0])

    odom = []
    rtk = []
    for i in range(len(pts)):
        phys = np.array([pts[i, 1], pts[i, 2], 80.0])
        rel = phys - t_true
        p_odom = R_up.T @ rel  # odometry in the (tilted) odometry frame
        odom.append((pts[i, 0], p_odom[0], p_odom[1], p_odom[2], 1.0, 0.0, 0.0, 0.0))

        q = scale * (R @ rel) + t_true
        q = q + rng.normal(0, 0.1, 3)
        if outliers and i % 25 == 17:
            q = q + np.array([50.0, -40.0, 0.0])
        rtk.append((pts[i, 0], q[0], q[1], q[2]))
    return np.asarray(odom), np.asarray(rtk), scale, R, t_true


def test_pose_graph_recovers_similarity():
    rng = np.random.default_rng(100)
    odom, rtk, scale, R, t_true = _synthetic_scaled_path(rng, scale=1.2, heading=0.4)
    res = align_pose_graph(odom, rtk, up_odom=np.array([0.0, 0.0, 1.0]), step_s=4.0)

    assert abs(res.series.scale - scale) < 0.02, f"scale {res.series.scale} vs {scale}"
    assert res.residuals.mean() < 0.5, f"mean residual {res.residuals.mean()}"

    # the series maps odometry poses back onto the RTK fixes within ~0.5 m
    errors = []
    for i in range(0, len(odom), 10):
        p_utm = res.series.apply(odom[i, 1:4][None, :], odom[i, 0])[0]
        p_true = rtk[np.argmin(np.abs(rtk[:, 0] - odom[i, 0])), 1:3]
        errors.append(np.linalg.norm(p_utm[:2] - p_true))
    assert np.mean(errors) < 0.5, f"mean mapping error {np.mean(errors)}"


def test_pose_graph_robust_to_rtk_outliers():
    rng = np.random.default_rng(101)
    odom, rtk, scale, R, t_true = _synthetic_scaled_path(
        rng, scale=1.2, heading=0.4, outliers=True
    )
    res = align_pose_graph(odom, rtk, up_odom=np.array([0.0, 0.0, 1.0]), step_s=4.0)

    # Huber loss must reject the 50 m outliers: scale stays close and the
    # bulk of the mapped positions stay near their (non-outlier) RTK fixes.
    assert abs(res.series.scale - scale) < 0.05, f"scale {res.series.scale} vs {scale}"
    errs = []
    for i in range(0, len(odom), 5):
        p_utm = res.series.apply(odom[i, 1:4][None, :], odom[i, 0])[0]
        p_true = rtk[np.argmin(np.abs(rtk[:, 0] - odom[i, 0])), 1:3]
        errs.append(np.linalg.norm(p_utm[:2] - p_true))
    errs = np.asarray(errs)
    assert np.median(errs) < 0.5, f"median mapping error {np.median(errs)}"
    assert np.percentile(errs, 75) < 2.0, f"75th pct mapping error {np.percentile(errs, 75)}"


def test_pose_graph_leveling_stabilizes_z():
    # A tilted odometry frame must not leak its Z drift into the aligned Z:
    # after leveling, the aligned Z sits near the constant RTK altitude.
    rng = np.random.default_rng(102)
    up_odom = np.array([0.3, -0.2, 0.9])
    odom, rtk, scale, R, t_true = _synthetic_scaled_path(
        rng, scale=1.0, heading=0.3, up_odom=up_odom
    )
    res = align_pose_graph(odom, rtk, up_odom=up_odom, step_s=4.0)

    zs = []
    for i in range(0, len(odom), 10):
        p_utm = res.series.apply(odom[i, 1:4][None, :], odom[i, 0])[0]
        zs.append(p_utm[2])
    zs = np.asarray(zs)
    assert zs.max() - zs.min() < 1.0, f"aligned Z spread {zs.max() - zs.min()}"


def test_transform_series_scale_applies():
    ts = TransformSeries(
        times=np.array([0.0]),
        rotations=np.array([np.eye(3)]),
        translations=np.array([[10.0, 20.0, 30.0]]),
        scale=2.0,
    )
    out = ts.apply(np.array([[1.0, 0.0, 0.0]]), 0.0)
    assert np.allclose(out, [[12.0, 20.0, 30.0]])

