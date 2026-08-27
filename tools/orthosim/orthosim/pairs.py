"""Pair generation: one query patch + one map window + exact ground truth.

GT is *specified* here (the pose at which the query is rendered), never
measured -- and it is fixed before augmentation and never recomputed after
(P5-data-sim rule). Geometry perturbations (attitude/AGL) would change the
pipeline input, so they belong in ``query_mode=pipeline`` only, never here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .augment import apply_augmentations
from .camera import normalize_angle
from .render import (
    ensure_cross_provider,
    rectify,
    render_map_window,
    render_perspective,
    render_true_ortho,
)


@dataclass
class PairSpec:
    id: int
    gt_east: float
    gt_north: float
    gt_yaw: float
    prior_east: float
    prior_north: float
    prior_yaw: float
    query_provider: str
    map_provider: str
    augment: dict = field(default_factory=dict)
    terrain_class: str = "background"

    @property
    def delta(self) -> tuple[float, float, float]:
        return (
            self.gt_east - self.prior_east,
            self.gt_north - self.prior_north,
            normalize_angle(self.gt_yaw - self.prior_yaw),
        )


@dataclass
class RenderedPair:
    spec: PairSpec
    query: np.ndarray
    map_window: np.ndarray


def generate_pair(
    scene,
    spec: PairSpec,
    rng: np.random.Generator,
    gsd: float,
    patch_size: int,
    window_size: int,
    aa: int = 2,
    query_mode: str = "direct",
    cam=None,
    R_cam_utm: np.ndarray | None = None,
) -> RenderedPair:
    """Render one pair, enforcing the cross-provider rule (T10-U-04)."""
    ensure_cross_provider(spec.query_provider, spec.map_provider)

    if query_mode == "pipeline":
        # Full path: perspective frame -> rectify. Exercises the same math as
        # T14/T15 without importing it. Camera params are required.
        if cam is None or R_cam_utm is None:
            raise ValueError("query_mode=pipeline requires cam and R_cam_utm")
        agl = _ground_agl(scene, spec.gt_east, spec.gt_north)
        C = np.array([spec.gt_east, spec.gt_north, agl + 100.0])
        frame = render_perspective(scene, spec.query_provider, cam, R_cam_utm, C, 612, 512)
        query = rectify(frame, scene, cam, R_cam_utm, C, gsd, patch_size)["rgb"]
    else:
        query = render_true_ortho(
            scene,
            spec.query_provider,
            spec.gt_east,
            spec.gt_north,
            spec.gt_yaw,
            gsd,
            patch_size,
            aa,
        )

    query = apply_augmentations(query, spec.augment, rng)

    map_window = render_map_window(
        scene,
        spec.map_provider,
        spec.prior_east,
        spec.prior_north,
        spec.prior_yaw,
        gsd,
        window_size,
        aa=1,
    )
    return RenderedPair(spec=spec, query=query, map_window=map_window)


def _ground_agl(scene, east: float, north: float) -> float:
    z = scene.z(np.array([east]), np.array([north]))[0]
    return 100.0 if not np.isfinite(z) else float(z)
