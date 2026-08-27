"""OrthoSim rendering: the geometric core of T10/T11.

Everything is expressed in the geopack's UTM (east, north, up-metres) frame and
follows the project raster conventions (pixel (0,0) top-left, centres at
(col+0.5, row+0.5), row increases southward).

Two complementary paths are provided:

- ``render_true_ortho`` / ``render_map_window`` -- sample the ortho *directly*
  at the ground grid (this is what a perfect true-ortho rectifier produces and
  what the matcher consumes). Independent of any camera/DSM code.
- ``render_perspective`` / ``rectify`` -- the forward perspective projection and
  its inverse through the DSM. This exercises the same math as the T14/T15
  pipeline *without importing it*, and gives T10-U-02 (warp round-trip) a real
  forward+inverse pair to test.
"""

from __future__ import annotations

import numpy as np

from .camera import Pinhole
from .scene import Scene


class CrossProviderError(RuntimeError):
    """Raised when a pair would be rendered from the same source (self-deception)."""


def _patch_centres(east_c: float, north_c: float, yaw: float, gsd: float, size: int):
    """Pixel-centre (east, north) grids for a ``size x size`` patch.

    ``yaw`` rotates the patch's up-axis away from map north, CCW from east.
    """
    cols = np.arange(size, dtype=np.float64)
    rows = np.arange(size, dtype=np.float64)
    u = (cols + 0.5 - size / 2.0) * gsd  # patch-frame east
    v = -(rows + 0.5 - size / 2.0) * gsd  # patch-frame north
    uu, vv = np.meshgrid(u, v)
    c, s = np.cos(yaw), np.sin(yaw)
    east = east_c + c * uu - s * vv
    north = north_c + s * uu + c * vv
    return east, north


def _box_average_sampled(sampled: np.ndarray, aa: int, size: int) -> np.ndarray:
    """Fold ``aa``-supersampled channels (size*aa, size*aa, C) into (size, size, C)."""
    if aa <= 1:
        return sampled
    flat = sampled.reshape(size, aa, size, aa, -1)
    return flat.mean(axis=(1, 3))


def render_true_ortho(
    scene: Scene,
    provider: str,
    east_c: float,
    north_c: float,
    yaw: float,
    gsd: float,
    size: int,
    aa: int = 2,
) -> np.ndarray:
    """Direct true-ortho query patch: ortho ``provider`` sampled at ground truth.

    Returns an ``(size, size, C)`` float64 array. This is the idealised rectified
    patch -- for a good orthophoto it is exactly what the T15 rectifier should
    produce, and it is the input the matcher is tested against.
    """
    if aa <= 1:
        east, north = _patch_centres(east_c, north_c, yaw, gsd, size)
        rgb = scene.sample(provider, east.ravel(), north.ravel())
        return rgb.reshape(size, size, -1)
    east, north = _patch_centres(east_c, north_c, yaw, gsd / aa, size * aa)
    rgb = scene.sample(provider, east.ravel(), north.ravel()).reshape(size * aa, size * aa, -1)
    return _box_average_sampled(rgb, aa, size)


def render_map_window(
    scene: Scene,
    provider: str,
    east_c: float,
    north_c: float,
    yaw: float,
    gsd: float,
    size: int,
    aa: int = 1,
) -> np.ndarray:
    """Map window (the search region) sampled from the map provider."""
    return render_true_ortho(scene, provider, east_c, north_c, yaw, gsd, size, aa=aa)


def _ray_march_terrain(
    C: np.ndarray,
    dirs: np.ndarray,
    scene: Scene,
    s_min: float,
    s_max: float,
    steps: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """March rays from ``C`` along ``dirs`` until they hit the terrain.

    Returns ``(s, hit)``: the distance along each ray and a boolean hit mask.
    """
    n = len(dirs)
    s_lo = np.full(n, s_min)
    s_hi = np.full(n, s_max)
    hit = np.zeros(n, dtype=bool)
    ds = (s_max - s_min) / steps
    for _ in range(steps):
        s_cur = s_lo + ds
        p = C + dirs * s_cur[:, None]
        zt = scene.z(p[:, 0], p[:, 1])
        below = np.isfinite(zt) & (p[:, 2] <= zt)
        newly = below & ~hit
        s_hi[newly] = s_cur[newly]
        hit |= below
        s_lo = np.where(hit, s_lo, s_cur)
    for _ in range(8):
        s_mid = 0.5 * (s_lo + s_hi)
        p = C + dirs * s_mid[:, None]
        zt = scene.z(p[:, 0], p[:, 1])
        below = np.isfinite(zt) & (p[:, 2] <= zt)
        s_hi = np.where(below, s_mid, s_hi)
        s_lo = np.where(below, s_lo, s_mid)
    return 0.5 * (s_lo + s_hi), hit


def _agl(scene: Scene, C: np.ndarray, floor: float = 30.0) -> float:
    z_ground = scene.z(np.array([C[0]]), np.array([C[1]]))[0]
    if not np.isfinite(z_ground):
        return floor
    return max(floor, float(C[2] - z_ground))


def render_perspective(
    scene: Scene,
    provider: str,
    cam: Pinhole,
    R_cam_utm: np.ndarray,
    C: np.ndarray,
    width: int,
    height: int,
    steps: int = 64,
) -> np.ndarray:
    """Forward-render a perspective camera frame from the ortho + terrain.

    Returns an ``(height, width, 3)`` float64 image; unseen pixels are 0.
    """
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    px = np.column_stack([u.ravel(), v.ravel()])
    dirs_cam = cam.cam2world(px)
    dirs_world = dirs_cam @ R_cam_utm.T
    agl = _agl(scene, C)
    s, hit = _ray_march_terrain(C, dirs_world, scene, 0.05 * agl, 1.6 * agl, steps)
    p = C + dirs_world * s[:, None]
    rgb = scene.sample(provider, p[:, 0], p[:, 1]).reshape(-1, 3)
    rgb = np.where(hit[:, None], rgb, 0.0)
    return rgb.reshape(height, width, 3)


def rectify(
    frame: np.ndarray,
    scene: Scene,
    cam: Pinhole,
    R_cam_utm: np.ndarray,
    C: np.ndarray,
    gsd: float,
    size: int,
    steps: int = 48,
) -> dict:
    """Backward-projection rectifier: perspective frame -> true-ortho patch.

    Mirrors the T15 backward projection (ray through the DSM, sample the frame
    through the full pinhole model) but is self-contained. Returns the patch
    rgb plus a confidence mask.
    """
    east, north = _patch_centres(C[0], C[1], 0.0, gsd, size)
    agl = _agl(scene, C)
    z = scene.z(east.ravel(), north.ravel())
    z = np.where(np.isfinite(z), z, C[2] - agl)
    p_hit = np.column_stack([east.ravel(), north.ravel(), z])
    uv = None
    for _ in range(3):
        xyz_cam = (p_hit - C) @ R_cam_utm
        uv = cam.world2cam(xyz_cam)
        dirs = cam.cam2world(uv) @ R_cam_utm.T
        s, hit = _ray_march_terrain(C, dirs, scene, 0.05 * agl, 1.6 * agl, steps)
        p_final = C + dirs * s[:, None]
        zt = scene.z(p_final[:, 0], p_final[:, 1])
        z_new = np.where(np.isfinite(zt), zt, p_final[:, 2])
        p_hit = np.column_stack([p_final[:, 0], p_final[:, 1], z_new])

    h_img, w_img = frame.shape[:2]
    in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < w_img) & (uv[:, 1] >= 0) & (uv[:, 1] < h_img)
    # bilinear sample the frame at uv (frame is (H, W, 3))
    u = uv[:, 0]
    v = uv[:, 1]
    c0 = np.floor(u).astype(np.int64)
    r0 = np.floor(v).astype(np.int64)
    fu, fv = u - c0, v - r0
    valid = in_frame & (c0 >= 0) & (c0 < w_img) & (r0 >= 0) & (r0 < h_img)
    c1 = np.clip(c0 + 1, 0, w_img - 1)
    r1 = np.clip(r0 + 1, 0, h_img - 1)
    c0 = np.clip(c0, 0, w_img - 1)
    r0 = np.clip(r0, 0, h_img - 1)
    p00 = frame[r0, c0].astype(np.float64)
    p10 = frame[r0, c1].astype(np.float64)
    p01 = frame[r1, c0].astype(np.float64)
    p11 = frame[r1, c1].astype(np.float64)
    rgb = (p00 * (1 - fu)[:, None] + p10 * fu[:, None]) * (1 - fv)[:, None] + (
        p01 * (1 - fu)[:, None] + p11 * fu[:, None]
    ) * fv[:, None]
    rgb = np.where(valid[:, None], rgb, 0.0)
    return {
        "rgb": rgb.reshape(size, size, 3),
        "confidence": valid.reshape(size, size).astype(np.float64),
    }


def ensure_cross_provider(query_provider: str, map_provider: str) -> None:
    """Blocking self-check (T10-U-04): query and map must come from different sources."""
    if query_provider == map_provider:
        raise CrossProviderError(
            f"self-deception guard: query provider {query_provider!r} == "
            f"map provider {map_provider!r}; OrthoSim refuses to render a pair "
            "from the same source"
        )
