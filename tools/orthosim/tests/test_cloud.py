"""Avia cloud synthesis tests: T11-U-01..04."""

from __future__ import annotations

import numpy as np
import rasterio.transform

from orthosim.cloud import (
    AVIA_POINTS_PER_SEC,
    AviaSpec,
    avia_rays,
    flat_ground_hits,
    synthesize_cloud,
)
from orthosim.dsm import FlatTerrain
from orthosim.geopack import CLASS_WATER, Field
from orthosim.synthetic import SyntheticScene


def _flat_scene(semantic: Field | None = None) -> SyntheticScene:
    def tex(east, north):
        v = np.full_like(np.asarray(east, dtype=np.float64), 100.0)
        return np.stack([v, v, v], axis=-1)

    return SyntheticScene(texture_a=tex, texture_b=tex, terrain=FlatTerrain(0.0), semantic=semantic)


def test_fov_footprint_t11_u_01():
    spec = AviaSpec()
    fh = np.deg2rad(spec.fov_h_deg) / 2.0
    fv = np.deg2rad(spec.fov_v_deg) / 2.0
    # The four extreme rays at the FOV corners.
    dirs = []
    for tx in (-fh, fh):
        for ty in (-fv, fv):
            d = np.array([np.tan(tx), np.tan(ty), -1.0])
            dirs.append(d / np.linalg.norm(d))
    C = np.tile(np.array([0.0, 0.0, 100.0]), (4, 1))
    hits = flat_ground_hits(C, np.array(dirs), z_ground=0.0)

    width = hits[:, 0].max() - hits[:, 0].min()
    height = hits[:, 1].max() - hits[:, 1].min()
    expected_w = 2 * 100.0 * np.tan(fh)
    expected_h = 2 * 100.0 * np.tan(fv)
    assert abs(width - expected_w) / expected_w < 0.02, width
    assert abs(height - expected_h) / expected_h < 0.02, height
    assert abs(expected_w - 141.27) < 1.0
    assert abs(expected_h - 159.72) < 1.0


def test_rays_stay_inside_fov():
    rng = np.random.default_rng(0)
    spec = AviaSpec()
    d = avia_rays(20000, rng, spec)
    fh = np.deg2rad(spec.fov_h_deg) / 2.0
    fv = np.deg2rad(spec.fov_v_deg) / 2.0
    assert np.all(np.abs(d[:, 0] / -d[:, 2]) <= np.tan(fh) + 1e-9)
    assert np.all(np.abs(d[:, 1] / -d[:, 2]) <= np.tan(fv) + 1e-9)


def test_density_t11_u_02():
    scene = _flat_scene()
    spec = AviaSpec()
    rng = np.random.default_rng(1)
    duration = 0.1
    n = int(spec.points_per_sec * duration)  # 24000

    def trajectory(t):
        return np.array([0.0, 0.0, 100.0]), np.eye(3)

    cloud = synthesize_cloud(
        scene,
        trajectory,
        0.0,
        duration,
        n,
        rng,
        spec,
        motion_distortion=False,
        drop_water=False,
        reflectivity=1.0,
    )
    rate = len(cloud) / duration
    assert abs(rate - AVIA_POINTS_PER_SEC) / AVIA_POINTS_PER_SEC < 0.05


def test_motion_distortion_t11_u_03():
    scene = _flat_scene()
    spec = AviaSpec()
    nadir = np.tile(np.array([0.0, 0.0, -1.0]), (4000, 1))
    v, duration = 10.0, 0.1

    def trajectory(t):
        return np.array([v * t, 0.0, 100.0]), np.eye(3)

    smeared = synthesize_cloud(
        scene,
        trajectory,
        0.0,
        duration,
        4000,
        np.random.default_rng(2),
        spec,
        motion_distortion=True,
        drop_water=False,
        reflectivity=1.0,
        dirs_body=nadir,
    )
    static = synthesize_cloud(
        scene,
        trajectory,
        0.0,
        duration,
        4000,
        np.random.default_rng(2),
        spec,
        motion_distortion=False,
        drop_water=False,
        reflectivity=1.0,
        dirs_body=nadir,
    )
    spread = smeared[:, 0].max() - smeared[:, 0].min()
    assert abs(spread - v * duration) < 0.05, spread
    assert static[:, 0].max() - static[:, 0].min() < 1e-9


def test_water_dropout_t11_u_04():
    water = Field(
        data=np.full((200, 200), CLASS_WATER, dtype=np.float64),
        transform=rasterio.transform.from_origin(-100.0, 100.0, 1.0, 1.0),
        nodata=None,
    )
    scene = _flat_scene(semantic=water)
    spec = AviaSpec()

    def trajectory(t):
        return np.array([0.0, 0.0, 100.0]), np.eye(3)

    dropped = synthesize_cloud(
        scene,
        trajectory,
        0.0,
        0.1,
        5000,
        np.random.default_rng(3),
        spec,
        motion_distortion=False,
        drop_water=True,
        reflectivity=1.0,
    )
    kept = synthesize_cloud(
        scene,
        trajectory,
        0.0,
        0.1,
        5000,
        np.random.default_rng(3),
        spec,
        motion_distortion=False,
        drop_water=False,
        reflectivity=1.0,
    )
    assert len(dropped) == 0
    assert len(kept) > 4000
