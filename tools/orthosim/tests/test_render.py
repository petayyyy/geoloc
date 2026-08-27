"""render / pair / augment tests: T10-U-01..05."""

import numpy as np
import pytest

from orthosim.augment import (
    contrast,
    gamma,
    gaussian_noise,
    haze,
    jpeg,
    motion_blur,
    season_shift,
    snow,
    vignette,
    white_balance,
)
from orthosim.camera import Pinhole
from orthosim.dsm import FlatTerrain
from orthosim.pairs import PairSpec, generate_pair
from orthosim.render import (
    CrossProviderError,
    rectify,
    render_map_window,
    render_perspective,
    render_true_ortho,
)
from orthosim.synthetic import SyntheticScene

R_NADIR = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
CAM = Pinhole(fx=400.0, fy=400.0, cx=100.0, cy=100.0)


def _texture(east: np.ndarray, north: np.ndarray) -> np.ndarray:
    r = 128.0 + 96.0 * np.sin(2 * np.pi * east / 9.0) * np.cos(2 * np.pi * north / 7.0)
    g = 128.0 + 96.0 * np.sin(2 * np.pi * east / 11.0 + 1.0)
    b = 128.0 + 96.0 * np.cos(2 * np.pi * north / 13.0 + 2.0)
    return np.stack([r, g, b], axis=-1)


def _analytic_scene() -> SyntheticScene:
    return SyntheticScene(texture_a=_texture, texture_b=_texture, terrain=FlatTerrain(0.0))


def test_gt_correctness_t10_u_01():
    # The query rendered at the specified GT pose must be exactly the texture
    # content at those map coordinates -- so an ideal same-source match recovers
    # SE(2) to sub-pixel error (0.1 px) trivially.
    scene = _analytic_scene()
    gt_east, gt_north, yaw = 5.25, -3.75, 0.3
    gsd, size = 0.5, 100
    query = render_true_ortho(scene, "a", gt_east, gt_north, yaw, gsd, size, aa=1)

    # Reconstruct the exact (east, north) grid render_true_ortho samples.
    cols = np.arange(size, dtype=np.float64)
    rows = np.arange(size, dtype=np.float64)
    u = (cols + 0.5 - size / 2) * gsd
    v = -(rows + 0.5 - size / 2) * gsd
    uu, vv = np.meshgrid(u, v)
    c, s = np.cos(yaw), np.sin(yaw)
    east = gt_east + c * uu - s * vv
    north = gt_north + s * uu + c * vv
    truth = _texture(east, north)

    assert np.allclose(query, truth, atol=1e-9)


def test_warp_round_trip_t10_u_02():
    # The camera projection model round-trips: world2cam(cam2world(px)) == px
    # to within 0.1 px (the "warp -> inverse warp" property, T10-U-02).
    rng = np.random.default_rng(7)
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    C = np.array([572800.0, 4964000.0, 200.0])
    px = rng.uniform(0, 200, size=(500, 2))

    dirs_cam = CAM.cam2world(px)
    dirs_world = dirs_cam @ R.T
    p = C + dirs_world * 75.0
    xyz_cam = (p - C) @ R
    px_back = CAM.world2cam(xyz_cam)
    assert np.max(np.abs(px_back - px)) < 0.1


def test_pipeline_round_trip_consistency():
    # Forward perspective render then backward rectification reproduces the
    # true-ortho patch (flat nadir scene), to interpolation tolerance.
    scene = _analytic_scene()
    C = np.array([0.0, 0.0, 100.0])
    frame = render_perspective(scene, "a", CAM, R_NADIR, C, width=200, height=200)
    res = rectify(frame, scene, CAM, R_NADIR, C, gsd=0.5, size=100)
    truth = render_true_ortho(scene, "a", 0.0, 0.0, 0.0, 0.5, 100, aa=1)

    err = np.abs(res["rgb"] - truth)
    mask = res["confidence"] > 0
    assert mask.mean() > 0.8
    assert err[mask].mean() < 6.0, f"round-trip mean error {err[mask].mean():.2f}"


def test_cross_provider_guard_t10_u_04():
    scene = _analytic_scene()
    spec = PairSpec(
        id=0,
        gt_east=0.0,
        gt_north=0.0,
        gt_yaw=0.0,
        prior_east=0.0,
        prior_north=0.0,
        prior_yaw=0.0,
        query_provider="a",
        map_provider="a",
    )
    with pytest.raises(CrossProviderError):
        generate_pair(scene, spec, np.random.default_rng(0), 0.5, 64, 128)


def test_augmentations_identity_t10_u_05():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 255, size=(32, 32, 3))
    identity_cases = {
        "gamma": lambda x: gamma(x, 1.0),
        "white_balance": lambda x: white_balance(x, [1.0, 1.0, 1.0]),
        "contrast": lambda x: contrast(x, 1.0),
        "haze": lambda x: haze(x, 0.0),
        "gaussian_noise": lambda x: gaussian_noise(x, rng, 0.0),
        "motion_blur": lambda x: motion_blur(x, 1),
        "vignette": lambda x: vignette(x, 0.0),
        "jpeg": lambda x: jpeg(x, 100),
        "snow": lambda x: snow(x, rng, 0.0),
        "season_shift": lambda x: season_shift(x, 0.0),
    }
    for name, fn in identity_cases.items():
        assert np.allclose(fn(img), img, atol=1e-9), f"{name} identity failed"

    changing = {
        "gamma": lambda x: gamma(x, 1.4),
        "white_balance": lambda x: white_balance(x, [1.3, 0.8, 1.1]),
        "contrast": lambda x: contrast(x, 1.5),
        "haze": lambda x: haze(x, 0.3),
        "gaussian_noise": lambda x: gaussian_noise(x, rng, 8.0),
        "motion_blur": lambda x: motion_blur(x, 3),
        "vignette": lambda x: vignette(x, 0.5),
        "jpeg": lambda x: jpeg(x, 60),
        "snow": lambda x: snow(x, rng, 0.5),
        "season_shift": lambda x: season_shift(x, 0.5),
    }
    for name, fn in changing.items():
        assert not np.allclose(fn(img), img), f"{name} should change the image"


def test_determinism_t10_u_03():
    # Two runs with the same seed produce byte-identical query/map arrays.
    scene = _analytic_scene()
    spec = PairSpec(
        id=0,
        gt_east=1.0,
        gt_north=-2.0,
        gt_yaw=0.1,
        prior_east=3.0,
        prior_north=5.0,
        prior_yaw=0.05,
        query_provider="a",
        map_provider="b",
        augment={"gaussian_noise": 4.0, "gamma": 1.1},
    )
    a = generate_pair(scene, spec, np.random.default_rng(7), 0.5, 64, 128)
    b = generate_pair(scene, spec, np.random.default_rng(7), 0.5, 64, 128)
    assert np.array_equal(a.query, b.query)
    assert np.array_equal(a.map_window, b.map_window)


def test_map_window_covers_gt():
    # The map window must be large enough to contain the query at the GT offset.
    scene = _analytic_scene()
    query = render_true_ortho(scene, "b", 4.0, 6.0, 0.0, 0.5, 64, aa=1)
    window = render_map_window(scene, "a", 0.0, 0.0, 0.0, 0.5, 256, aa=1)
    assert query.shape == (64, 64, 3)
    assert window.shape == (256, 256, 3)
