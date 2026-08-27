"""Regression tests for the georeference self-check (rtkcheck).

The bug these guard against, found 2026-08-27: `geoloc_capture_01` was read
with `rtk.swap_latlon: true`, i.e. its (correctly labelled) 39.92 N / 44.83 E
fixes were interpreted as 44.83 N / 39.92 E. That is an *anisotropic*
distortion of the track -- northing x1.300, easting x0.712 -- so it survives
every rigid and similarity fit; downstream it surfaced as a fitted alignment
"scale" of 1.2261 and was written up as a 22% odometry scale error.

A projected track compared against the receiver's own Doppler velocity catches
it immediately, because velocity is a metric statement that no map projection
touches.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyproj import Transformer

from orthoproto.rtkcheck import doppler_consistency, integrate_velocity, velocity_en

# The real site of geoloc_capture_01 (MARS-LVIG AMtown03, Ararat plain, Armenia).
SITE_LAT = 39.9205
SITE_LON = 44.8280
UTM_TRUE = "EPSG:32638"  # zone for lon 44.83
UTM_SWAPPED = "EPSG:32637"  # zone the swapped reading (lon 39.92) falls in
SEED = 20260827


def _synthetic_track(seed: int = SEED, n: int = 200, hz: float = 3.32, straight_heading_deg=None):
    """A deterministic 12 m/s track with a turn, as (t, lat, lon, v_east, v_north).

    Built velocity-first and integrated on the ellipsoid, so the positions and
    the velocities are consistent by construction -- exactly the property the
    check is testing for.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / hz
    speed = 12.0
    if straight_heading_deg is None:
        heading = np.deg2rad(25.0) + np.deg2rad(60.0) * np.tanh((t - t[-1] / 2) / 6.0)
    else:
        heading = np.full(n, np.deg2rad(straight_heading_deg))
    v_east = speed * np.sin(heading)
    v_north = speed * np.cos(heading)
    disp = integrate_velocity(t, np.column_stack([v_east, v_north]))
    # metres -> degrees about the site (local tangent plane; the check only
    # ever looks at short baselines, where this is exact to well under 1 mm)
    m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * np.deg2rad(SITE_LAT))
    m_per_deg_lon = 111412.84 * np.cos(np.deg2rad(SITE_LAT)) - 93.5 * np.cos(
        3 * np.deg2rad(SITE_LAT)
    )
    lat = SITE_LAT + disp[:, 1] / m_per_deg_lat
    lon = SITE_LON + disp[:, 0] / m_per_deg_lon
    # a little receiver noise, so the test is not a noise-free special case
    lat = lat + rng.normal(0.0, 0.01, n) / m_per_deg_lat
    lon = lon + rng.normal(0.0, 0.01, n) / m_per_deg_lon
    return t, lat, lon, np.column_stack([v_east, v_north])


def _project(lat, lon, epsg):
    tr = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    east, north = tr.transform(lon, lat)
    return np.column_stack([east, north])


def test_correct_georeference_is_metrically_self_consistent():
    t, lat, lon, v_en = _synthetic_track()
    res = doppler_consistency(t, _project(lat, lon, UTM_TRUE), v_en)
    assert res.ok(0.02)
    assert res.ratio == pytest.approx(1.0, abs=2e-3)


def test_swapped_latlon_is_caught():
    """The exact 2026-08-27 bug: read `latitude` as lon and `longitude` as lat."""
    t, lat, lon, v_en = _synthetic_track()
    swapped = _project(lon, lat, UTM_SWAPPED)  # lat<->lon, and the zone follows
    res = doppler_consistency(t, swapped, v_en)
    assert not res.ok(0.02)
    # The distortion is anisotropic, so the ratio depends on heading and its
    # extremes are the two per-axis factors: a due-north leg is squeezed to
    # ~0.712 and a due-east leg stretched to ~1.300. On the real capture,
    # whose track runs mostly east-west, the median came out 1.2277 -- which
    # is the number the old pose-graph fitted as a 1.2261 "odometry scale".
    assert 0.69 < res.ratio_min < 0.73
    assert 1.28 < res.ratio_max < 1.32


@pytest.mark.parametrize(
    "heading_deg, expected",
    [(90.0, 1.300), (0.0, 0.712)],  # due east stretched, due north squeezed
)
def test_swap_distortion_is_the_metres_per_degree_ratio(heading_deg, expected):
    """Pin the two factors: they are pure geodesy, not a fitted quantity."""
    t, lat, lon, v_en = _synthetic_track(straight_heading_deg=heading_deg)
    res = doppler_consistency(t, _project(lon, lat, UTM_SWAPPED), v_en)
    assert res.ratio == pytest.approx(expected, abs=5e-3)


def test_swap_survives_a_similarity_fit_but_not_this_check():
    """Why a fitted scale could never have caught it: the distortion is anisotropic.

    A best-fit *similarity* (uniform scale + rotation) still leaves metres of
    residual, so the residual alone looks like ordinary odometry drift.
    """
    t, lat, lon, v_en = _synthetic_track()
    true_en = _project(lat, lon, UTM_TRUE)
    swapped = _project(lon, lat, UTM_SWAPPED)

    def similarity_residual(a, b):
        ac, bc = a - a.mean(0), b - b.mean(0)
        u, sv, vt = np.linalg.svd(ac.T @ bc)
        d = np.eye(2)
        if np.linalg.det(u @ vt) < 0:
            d[1, 1] = -1
        rot = (u @ d @ vt).T
        scale = (sv * np.diag(d)).sum() / (ac**2).sum()
        return np.sqrt((((bc - scale * (rot @ ac.T).T) ** 2).sum(1)).mean())

    assert similarity_residual(true_en, swapped) > 5.0
    assert doppler_consistency(t, true_en, v_en).ok(0.02)


def test_ratio_is_invariant_to_baseline_over_short_windows():
    t, lat, lon, v_en = _synthetic_track()
    pos = _project(lat, lon, UTM_TRUE)
    ratios = [doppler_consistency(t, pos, v_en, baseline_s=b).ratio for b in (1.0, 3.0, 9.0)]
    assert max(ratios) - min(ratios) < 5e-3


def test_velocity_frame_selection():
    v = np.array([[1.0, 2.0, 3.0]])
    assert velocity_en(v, "ned").tolist() == [[2.0, 1.0]]
    assert velocity_en(v, "enu").tolist() == [[1.0, 2.0]]
    with pytest.raises(ValueError, match="unknown rtk velocity frame"):
        velocity_en(v, "body")


def test_stationary_track_refuses_rather_than_guessing():
    t = np.arange(60) / 3.0
    pos = np.zeros((60, 2))
    with pytest.raises(ValueError, match="never moved far enough"):
        doppler_consistency(t, pos, np.zeros((60, 2)))
