"""ortho.py tests: flat-scene nadir warp reproduces the ground truth (T15-U-02)."""

import numpy as np
import rasterio.transform

from orthoproto.camera import Pinhole
from orthoproto.dsm import NODATA, DsmGrid
from orthoproto.ortho import DemField, terrain_height, warp_frame

CFG = {
    "patch_gsd_m": 0.5,
    "patch_radius_m": 55.0,
    "ray_steps": 48,
    "agl_floor_m": 30.0,
}

W, H = 612, 512
FX, FY, CX, CY = 400.0, 400.0, 306.0, 256.0
# nadir camera: camera (x right, y down, z forward-down) -> ENU (east, north, up)
R_NADIR = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def _flat_world(gsd: float, half: float) -> DsmGrid:
    n = int(2 * half / gsd)
    z = np.zeros((n, n), dtype=np.float32)
    return DsmGrid(
        gsd=gsd,
        east_min=-half,
        north_max=half,
        height=n,
        width=n,
        z=z,
        count=np.full((n, n), 10, dtype=np.uint32),
        dispersion=np.zeros((n, n), dtype=np.float32),
        confidence=np.full((n, n), 255, dtype=np.uint8),
    )


def _flat_dem() -> DemField:
    n = 400
    z = np.zeros((n, n), dtype=np.float64)
    return DemField(
        z=z,
        transform=rasterio.transform.from_origin(-100.0, 100.0, 0.5, 0.5),
        nodata=-9999.0,
    )


def _texture(east: np.ndarray, north: np.ndarray) -> np.ndarray:
    r = 128.0 + 96.0 * np.sin(2 * np.pi * east / 9.0) * np.cos(2 * np.pi * north / 7.0)
    g = 128.0 + 96.0 * np.sin(2 * np.pi * east / 11.0 + 1.0)
    b = 128.0 + 96.0 * np.cos(2 * np.pi * north / 13.0 + 2.0)
    return np.stack([r, g, b], axis=-1)


def _render_image(cam: Pinhole, C: np.ndarray, height: float) -> np.ndarray:
    """Forward-render the analytic texture into the camera (the inverse path)."""
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    px = np.column_stack([u.ravel(), v.ravel()])
    dirs = cam.cam2world(px)  # camera frame
    dirs_utm = dirs @ R_NADIR
    s = height / -dirs_utm[:, 2]
    p = C + dirs_utm * s[:, None]
    return _texture(p[:, 0], p[:, 1]).reshape(H, W, 3)


def test_nadir_flat_scene_reproduces_ground_truth():
    cam = Pinhole(fx=FX, fy=FY, cx=CX, cy=CY)
    dsm = _flat_world(0.5, 90.0)
    dem = _flat_dem()
    height = 100.0
    C = np.array([0.0, 0.0, height])
    img = _render_image(cam, C, height)

    res = warp_frame(img, cam, R_NADIR, C, dsm, dem, CFG)

    east_g, north_g = np.meshgrid(res["east"], res["north"])
    truth = _texture(east_g, north_g)
    err = np.abs(res["rgb"] - truth)
    # center region: exact to interpolation tolerance; edges of the frame fall
    # outside the patch and carry zero confidence there
    mask = res["confidence"] > 0
    assert mask.mean() > 0.5
    assert err[mask].mean() < 6.0, f"mean RGB error {err[mask].mean():.2f}"
    assert res["confidence"][mask].max() == 255


def test_ray_direction_round_trip():
    # Regression: warp_frame's world->camera step is `(p - C) @ R_cam_utm`
    # (equivalent to R_cam_utm.T @ (p - C) for a row vector -- see the
    # identity `v @ M == (M.T @ v.T).T`). The camera->world step used to be
    # `(R_cam_utm.T @ cam2world(uv).T).T`, i.e. it applied R_cam_utm.T too --
    # the SAME map as world->camera, not its inverse. Composing them was
    # (R_cam_utm.T)**2, not the identity, so a real terrain hit's own pixel
    # reprojected through this "round trip" came back at a wildly different
    # uv instead of the same one, and iterating that in warp_frame diverged
    # every ray to nan/inf within 2-3 rounds (verified empirically against
    # the real capture: 1208 genuine hits in one 520x520 patch collapsed to
    # 0 after just one more iteration).
    cam = Pinhole(fx=FX, fy=FY, cx=CX, cy=CY, k1=-0.052, k2=0.1168, p1=0.0015, p2=0.00013)
    rng = np.random.default_rng(7)
    R_cam_utm, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    C = np.array([572800.0, 4964000.0, 200.0])
    uv = np.array([[58.5, 344.7]])

    dirs_cam = cam.cam2world(uv)
    dirs_world = dirs_cam @ R_cam_utm.T
    p = C + dirs_world[0] * 75.0
    xyz_cam_check = (p - C) @ R_cam_utm
    assert np.allclose(xyz_cam_check, 75.0 * dirs_cam[0], atol=1e-9)
    uv_check = cam.world2cam(xyz_cam_check[None, :])
    assert np.allclose(uv_check, uv, atol=1e-6)


def test_terrain_height_rejects_cells_touching_nodata():
    # Regression: bilinear-interpolating across a DSM nodata boundary used to
    # blend the -9999 sentinel into a finite-looking but wildly wrong height
    # (a `vals == NODATA` check after the blend never catches this, since a
    # mix of real height and the sentinel is neither). Half the grid is real
    # ground at z=150, the other half is nodata; the interpolated cells that
    # straddle the boundary must fall back to the (flat, z=0) DEM instead of
    # returning a poisoned blend.
    gsd = 1.0
    n = 20
    z = np.full((n, n), 150.0, dtype=np.float32)
    z[:, n // 2 :] = NODATA
    dsm = DsmGrid(
        gsd=gsd,
        east_min=0.0,
        north_max=float(n),
        height=n,
        width=n,
        z=z,
        count=np.where(z != NODATA, 10, 0).astype(np.uint32),
        dispersion=np.zeros((n, n), dtype=np.float32),
        confidence=np.where(z != NODATA, 255, 0).astype(np.uint8),
    )
    dem = _flat_dem()

    # Sample squarely on the boundary column, where bilinear interpolation
    # mixes a real (150) and a nodata (-9999) corner.
    east = np.array([n / 2.0])
    north = np.array([n / 2.0])
    zval, lidar = terrain_height(east, north, dsm, dem)

    assert lidar[0] == 0.0, "a cell touching nodata must not be reported as lidar-covered"
    assert abs(zval[0]) < 1e-6, "must fall back to the (flat, z=0) DEM, not a blended DSM value"


def test_dem_bilinear_rejects_cells_touching_nodata():
    n = 10
    z = np.full((n, n), 100.0, dtype=np.float64)
    z[:, n // 2 :] = -9999.0
    dem = DemField(
        z=z, transform=rasterio.transform.from_origin(0.0, float(n), 1.0, 1.0), nodata=-9999.0
    )
    vals = dem.bilinear(np.array([n / 2.0]), np.array([n / 2.0]))
    assert np.isnan(vals[0]), "a cell touching nodata must not return a blended finite value"


def test_camera_at_patch_centre_projection_symmetry():
    """Patch grid centred on the nadir point: frame centre maps to patch centre."""
    cam = Pinhole(fx=FX, fy=FY, cx=CX, cy=CY)
    dsm = _flat_world(0.5, 90.0)
    dem = _flat_dem()
    C = np.array([0.0, 0.0, 100.0])
    img = _render_image(cam, C, 100.0)
    res = warp_frame(img, cam, R_NADIR, C, dsm, dem, CFG)
    n = len(res["east"])
    centre = res["confidence"][n // 2 - 4 : n // 2 + 5, n // 2 - 4 : n // 2 + 5]
    assert centre.mean() > 200.0
