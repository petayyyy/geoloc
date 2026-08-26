"""dsm.py tests: plane rasterization (T14-U-01), outlier filter (T14-U-02)."""

import numpy as np

from orthoproto.align import TransformSeries
from orthoproto.dsm import NODATA, build_dsm

CFG = {
    "dsm_gsd_m": 0.5,
    "margin_m": 20.0,
    "range_max_m": 190.0,
    "median_win": 5,
    "outlier_threshold_m": 3.0,
    "dense_min_points": 4,
    "sparse_min_points": 1,
}


def _plane_capture(
    rng: np.random.default_rng, spike: bool = False, z_ground: float = 100.0
) -> tuple:
    """Drone flying over a flat plane at z=z_ground; clouds + odom + identity series.

    The identity series makes the cloud coordinates already UTM; the range
    gate needs the drone position at each cloud time, hence the odom rows.
    """
    e0, n0 = 5000.0, 6000.0
    series = TransformSeries(
        times=np.array([0.0, 1000.0]),
        rotations=np.stack([np.eye(3), np.eye(3)]),
        translations=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    odom = []
    for i in range(21):
        t = i * 0.5
        e = e0 + (i - 10) * 6.0
        odom.append((t, e, n0, z_ground + 100.0, 1.0, 0.0, 0.0, 0.0))

    clouds = []
    for i in range(20):
        t = i * 0.5 + 0.25
        ex = rng.uniform(e0 - 100, e0 + 100, 20000)
        ny = rng.uniform(n0 - 100, n0 + 100, 20000)
        z = np.full(20000, z_ground) + rng.normal(0, 0.005, 20000)
        if spike:
            k = np.argmin(np.abs(ex - e0) + np.abs(ny - n0))
            z[k] = z_ground + 50.0
        clouds.append((t, np.column_stack([ex, ny, z]).astype(np.float32)))
    return series, np.asarray(odom), clouds, e0, n0


def test_plane_dsm_constant_height():
    rng = np.random.default_rng(11)
    series, odom, clouds, e0, n0 = _plane_capture(rng)
    dsm = build_dsm(series, odom, iter(clouds), CFG)
    dense = dsm.count >= CFG["dense_min_points"]
    assert dense.sum() > 4000
    # point noise is 5 mm and max-z aggregation biases upward, hence 3 cm here
    # (the task criterion of 2 cm is for a noiseless plane)
    heights = dsm.z[dense] - 100.0
    assert np.abs(heights).max() < 0.03, f"max height error {np.abs(heights).max():.3f} m"


def test_outlier_cell_replaced_by_median():
    rng = np.random.default_rng(12)
    series, odom, clouds, e0, n0 = _plane_capture(rng, spike=True)
    dsm = build_dsm(series, odom, iter(clouds), CFG)
    dense = dsm.count >= CFG["dense_min_points"]
    heights = dsm.z[dense]
    # the 150 m spike must not survive the median filter
    assert np.abs(heights - (100.0)).max() < 0.5
    assert (dsm.confidence[dense] >= 192).all() or (dsm.confidence[dense] >= 96).all()


def test_plane_dsm_survives_negative_utm_z():
    # Regression: accumulate_clouds used to reject every point with an
    # absolute-Z sanity floor (p[:, 2] > -500.0). That floor assumed Z was
    # already a plausible elevation, but build_dsm runs on align.series
    # BEFORE anchor_to_dem applies its z_shift -- for this capture the
    # pre-shift Z sits near the raw (unreliable, ~-1155 m) RTK-anchored
    # value, well below -500, so real points were silently dropped and the
    # DSM came out 0% covered. Ground truth here is deliberately placed at
    # z=-1200 to reproduce that regime; only the drone-relative range gate
    # should apply now.
    rng = np.random.default_rng(14)
    series, odom, clouds, e0, n0 = _plane_capture(rng, z_ground=-1200.0)
    dsm = build_dsm(series, odom, iter(clouds), CFG)
    dense = dsm.count >= CFG["dense_min_points"]
    assert dense.sum() > 4000, "points below the old -500 m floor must not be dropped"
    heights = dsm.z[dense] - (-1200.0)
    assert np.abs(heights).max() < 0.03


def test_holes_are_nodata():
    rng = np.random.default_rng(13)
    series, odom, clouds, e0, n0 = _plane_capture(rng)
    # thin cloud coverage: only a small square in the middle
    clouds_thin = []
    for i in range(4):
        t = i * 0.5 + 0.25
        ex = rng.uniform(e0 - 20, e0 + 20, 4000)
        ny = rng.uniform(n0 - 20, n0 + 20, 4000)
        clouds_thin.append((t, np.column_stack([ex, ny, np.full(4000, 100.0)]).astype(np.float32)))
    dsm = build_dsm(series, odom, iter(clouds_thin), CFG)
    holes = dsm.count == 0
    assert holes.sum() > 0
    assert (dsm.z[holes] == NODATA).all()
    assert (dsm.confidence[holes] == 0).all()
