"""camera.py tests: projection round trips through the radtan model."""

import numpy as np

from orthoproto.camera import Pinhole


def _cam() -> Pinhole:
    return Pinhole(
        fx=363.47, fy=363.21, cx=295.63, cy=261.46,
        k1=-0.052, k2=0.1168, p1=0.0015, p2=0.00013,
    )


def test_project_unproject_roundtrip():
    cam = _cam()
    rng = np.random.default_rng(3)
    px = np.column_stack([rng.uniform(0, 611, 500), rng.uniform(0, 511, 500)])
    dirs = cam.cam2world(px)
    px2 = cam.world2cam(dirs * 100.0)
    rms = float(np.sqrt(np.mean(np.sum((px2 - px) ** 2, axis=1))))
    assert rms < 1e-3, f"round-trip RMS {rms} px"


def test_distorted_projection_matches_opencv_model():
    cam = Pinhole(fx=400.0, fy=400.0, cx=300.0, cy=250.0, k1=0.1, k2=0.02, p1=0.001, p2=-0.0005)
    # point straight ahead with lateral offset: distortion must move it outward
    p0 = np.array([[0.0, 0.0, 1.0]])
    uv0 = cam.world2cam(p0)
    assert np.allclose(uv0, [[300.0, 250.0]], atol=1e-9)
    p1 = np.array([[0.2, 0.0, 1.0]])
    uv1 = cam.world2cam(p1)
    r_norm = uv1[0, 0] - 300.0
    assert r_norm > 80.0  # 0.2*400=80 undistorted; k1>0 pushes outward


def test_undistort_inverts_distort():
    cam = _cam()
    rng = np.random.default_rng(5)
    xn = rng.uniform(-0.6, 0.6, 1000)
    yn = rng.uniform(-0.6, 0.6, 1000)
    xd, yd = cam.distort(xn, yn)
    xn2, yn2 = cam.undistort(xd, yd)
    assert np.max(np.abs(xn2 - xn)) < 1e-8
    assert np.max(np.abs(yn2 - yn)) < 1e-8
