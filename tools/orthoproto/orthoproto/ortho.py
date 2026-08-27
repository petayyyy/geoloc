"""True-ortho patches from /rgb_img (T15, offline).

Backward projection: for every output grid pixel the ray is marched through
the lidar DSM (Copernicus DEM as fallback outside lidar coverage) and the
source frame is sampled through the full pinhole+radtan model -- the frames
are NOT undistorted, the distortion lives in the projection math.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import yaml
from PIL import Image

from .align import AlignmentResult, interpolate_odom_pose
from .camera import Pinhole
from .dsm import NODATA, DsmGrid


@dataclass
class DemField:
    """Copernicus DEM layer, in-memory with bilinear sampling."""

    z: np.ndarray
    transform: object
    nodata: float

    @classmethod
    def open(cls, path: Path) -> DemField:
        with rasterio.open(path) as src:
            return cls(z=src.read(1).astype(np.float64), transform=src.transform, nodata=src.nodata)

    def bilinear(self, east: np.ndarray, north: np.ndarray) -> np.ndarray:
        col = (east - self.transform.c) / self.transform.a
        row = (north - self.transform.f) / self.transform.e
        c0 = np.floor(col).astype(np.int64)
        r0 = np.floor(row).astype(np.int64)
        fc, fr = col - c0, row - r0
        h, w = self.z.shape
        z = np.full(east.shape, np.nan, dtype=np.float64)
        valid = (c0 >= 0) & (c0 < w) & (r0 >= 0) & (r0 < h)
        c1 = np.clip(c0 + 1, 0, w - 1)
        r1 = np.clip(r0 + 1, 0, h - 1)
        c0 = np.clip(c0, 0, w - 1)
        r0 = np.clip(r0, 0, h - 1)
        z00 = self.z[r0, c0].astype(np.float64)
        z10 = self.z[r0, c1].astype(np.float64)
        z01 = self.z[r1, c0].astype(np.float64)
        z11 = self.z[r1, c1].astype(np.float64)
        # Any nodata corner must invalidate the whole interpolated cell -- a
        # post-blend `vals == nodata` check does not, since blending a real
        # height with a nodata sentinel (e.g. -9999) produces some other
        # finite-looking number, not the sentinel itself.
        any_nodata = (
            (z00 == self.nodata)
            | (z10 == self.nodata)
            | (z01 == self.nodata)
            | (z11 == self.nodata)
        )
        vals = (z00 * (1 - fc) + z10 * fc) * (1 - fr) + (z01 * (1 - fc) + z11 * fc) * fr
        vals[any_nodata | ~valid] = np.nan
        z[valid] = vals[valid]
        return z


def terrain_height(
    east: np.ndarray, north: np.ndarray, dsm: DsmGrid, dem: DemField
) -> tuple[np.ndarray, np.ndarray]:
    """Height at (east, north): DSM where covered, DEM fallback elsewhere.
    Second return is a lidar-coverage flag (1.0 DSM, 0.0 DEM only)."""
    col = (east - dsm.east_min) / dsm.gsd - 0.5
    row = (dsm.north_max - north) / dsm.gsd - 0.5
    c0 = np.floor(col).astype(np.int64)
    r0 = np.floor(row).astype(np.int64)
    fc, fr = col - c0, row - r0
    h, w = dsm.z.shape
    inside = (c0 >= 0) & (c0 < w) & (r0 >= 0) & (r0 < h)
    c1 = np.clip(c0 + 1, 0, w - 1)
    r1 = np.clip(r0 + 1, 0, h - 1)
    c0 = np.clip(c0, 0, w - 1)
    r0 = np.clip(r0, 0, h - 1)
    z00 = dsm.z[r0, c0].astype(np.float64)
    z10 = dsm.z[r0, c1].astype(np.float64)
    z01 = dsm.z[r1, c0].astype(np.float64)
    z11 = dsm.z[r1, c1].astype(np.float64)
    # Same nodata-blending hazard as DemField.bilinear: a cell with any
    # nodata corner must be rejected outright, not accepted because the
    # blended value happens not to equal NODATA exactly.
    any_nodata = (z00 == NODATA) | (z10 == NODATA) | (z01 == NODATA) | (z11 == NODATA)
    dsm_vals = (z00 * (1 - fc) + z10 * fc) * (1 - fr) + (z01 * (1 - fc) + z11 * fc) * fr
    lidar = inside & ~any_nodata & ~np.isnan(dsm_vals)
    dsm_vals = np.where(lidar, dsm_vals, np.nan)

    dem_vals = dem.bilinear(east, north)
    z = np.where(np.isnan(dsm_vals), dem_vals, dsm_vals)
    return z, lidar.astype(np.float32)


def sample_image_bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear sample of an (H, W, 3) uint8 image at (u, v); 0 outside."""
    h, w = img.shape[:2]
    c0 = np.floor(u).astype(np.int64)
    r0 = np.floor(v).astype(np.int64)
    fu, fv = u - c0, v - r0
    valid = (c0 >= 0) & (c0 < w) & (r0 >= 0) & (r0 < h)
    c1 = np.clip(c0 + 1, 0, w - 1)
    r1 = np.clip(r0 + 1, 0, h - 1)
    c0 = np.clip(c0, 0, w - 1)
    r0 = np.clip(r0, 0, h - 1)
    p00 = img[r0, c0].astype(np.float64)
    p10 = img[r0, c1].astype(np.float64)
    p01 = img[r1, c0].astype(np.float64)
    p11 = img[r1, c1].astype(np.float64)
    out = ((p00 * (1 - fu[..., None]) + p10 * fu[..., None]) * (1 - fv[..., None]) +
           (p01 * (1 - fu[..., None]) + p11 * fu[..., None]) * fv[..., None])
    return np.where(valid[..., None], out, 0.0).astype(np.float64)


def box_blur(img: np.ndarray, passes: int) -> np.ndarray:
    """Separable 3-tap box blur; two passes approximate a small gaussian (AA)."""
    out = img.astype(np.float64)
    k = np.array([1.0, 2.0, 1.0]) / 4.0
    for _ in range(passes):
        out = np.pad(out, ((1, 1), (0, 0), (0, 0)), mode="edge")
        out = (out[:-2] * k[0] + out[1:-1] * k[1] + out[2:] * k[2])
        out = np.pad(out, ((0, 0), (1, 1), (0, 0)), mode="edge")
        out = out[:, :-2] * k[0] + out[:, 1:-1] * k[1] + out[:, 2:] * k[2]
    return out


def _ratio(flag: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of the usable pixels where `flag` holds; 0.0 if none are usable."""
    return float(np.mean(flag[mask])) if mask.any() else 0.0


def sample_density(uv: np.ndarray) -> np.ndarray:
    """Linear camera pixels backing one ortho pixel, per pixel of the grid.

    `uv` is the (h, w, 2) field of source-image coordinates each ortho pixel
    samples from. The Jacobian determinant of that map is the camera-pixel
    *area* one ortho pixel draws on; its square root is the linear figure, so
    1.0 means "one camera pixel per ortho pixel" and 0.1 means a single camera
    pixel is stretched across ten -- the ray-marching smear.

    Central differences inside, one-sided at the border; non-finite entries
    (rays that missed) give 0.0 rather than a guess.
    """
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 3 or uv.shape[2] != 2:
        raise ValueError(f"sample_density expects (h, w, 2), got {uv.shape}")
    safe = np.where(np.isfinite(uv), uv, 0.0)
    du_dc, du_dr = np.gradient(safe[:, :, 0], axis=1), np.gradient(safe[:, :, 0], axis=0)
    dv_dc, dv_dr = np.gradient(safe[:, :, 1], axis=1), np.gradient(safe[:, :, 1], axis=0)
    det = np.abs(du_dc * dv_dr - dv_dc * du_dr)
    density = np.sqrt(det)
    return np.where(np.isfinite(uv).all(axis=2), density, 0.0)


def warp_frame(
    img: np.ndarray,
    cam: Pinhole,
    R_cam_utm: np.ndarray,
    C: np.ndarray,
    dsm: DsmGrid,
    dem: DemField,
    cfg: dict,
) -> dict:
    """Backward-project one frame into a UTM patch; returns arrays + metadata."""
    gsd = float(cfg.get("patch_gsd_m", 0.5))
    radius = float(cfg.get("patch_radius_m", 130.0))
    agl_floor = float(cfg.get("agl_floor_m", 30.0))
    steps = int(cfg.get("ray_steps", 48))
    min_sample_density = float(cfg.get("min_sample_density", 0.0))

    n = int(round(2 * radius / gsd))
    east = C[0] + (np.arange(n) - n / 2 + 0.5) * gsd
    north = C[1] - (np.arange(n) - n / 2 + 0.5) * gsd
    east_g, north_g = np.meshgrid(east, north)
    h, w = east_g.shape

    z_g, lidar_g = terrain_height(east_g, north_g, dsm, dem)
    z_cam, _ = terrain_height(np.array([C[0]]), np.array([C[1]]), dsm, dem)
    agl = max(agl_floor, float(C[2] - z_cam[0]))
    if not np.isfinite(agl):
        agl = 100.0

    # Ray direction depends on the hit pixel, the pixel depends on the hit
    # point: alternate marching and re-projection (converges in 2-3 rounds
    # for a near-nadir camera).

    def march(dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (s_final, hit) by ray marching against the terrain field."""
        s_lo = np.full(len(dirs), 0.15 * agl)
        s_hi = np.full(len(dirs), 1.6 * agl)
        hit = np.zeros(len(dirs), dtype=bool)
        ds = (1.6 * agl - 0.15 * agl) / steps
        for _ in range(steps):
            s_cur = s_lo + ds
            p = C + dirs * s_cur[:, None]
            zt, _ = terrain_height(p[:, 0], p[:, 1], dsm, dem)
            below = np.isfinite(zt) & (p[:, 2] <= zt)
            newly = below & ~hit
            s_hi[newly] = s_cur[newly]
            hit |= below
            s_lo = np.where(hit, s_lo, s_cur)
        for _ in range(6):
            s_mid = 0.5 * (s_lo + s_hi)
            p = C + dirs * s_mid[:, None]
            zt, _ = terrain_height(p[:, 0], p[:, 1], dsm, dem)
            below = np.isfinite(zt) & (p[:, 2] <= zt)
            s_hi = np.where(below, s_mid, s_hi)
            s_lo = np.where(below, s_lo, s_mid)
        return 0.5 * (s_lo + s_hi), hit

    z_guess = np.where(np.isfinite(z_g), z_g, C[2] - agl)
    p_hit = np.column_stack([east_g.ravel(), north_g.ravel(), z_guess.ravel()])
    for _ in range(3):
        xyz_cam = (p_hit - C) @ R_cam_utm
        uv = cam.world2cam(xyz_cam)
        # camera -> world is the inverse of world -> camera above (`@ R_cam_utm`),
        # so this needs R_cam_utm itself, not its transpose. `v @ R_cam_utm.T`
        # applies R_cam_utm to the row-vector v (see the `xyz_cam` line's
        # identity, `v @ M == (M.T @ v.T).T`). The transposed form here used
        # to apply R_cam_utm.T twice over a round trip -- not the identity --
        # which discarded every ray direction (see test_ray_direction_round_trip).
        dirs = cam.cam2world(uv) @ R_cam_utm.T
        s_final, hit = march(dirs)
        p_final = C + dirs * s_final[:, None]
        zt_final, lidar_final = terrain_height(p_final[:, 0], p_final[:, 1], dsm, dem)
        z_new = np.where(np.isfinite(zt_final), zt_final, p_final[:, 2])
        p_hit = np.column_stack([p_final[:, 0], p_final[:, 1], z_new])

    h_img, w_img = img.shape[:2]
    in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < w_img) & (uv[:, 1] >= 0) & (uv[:, 1] < h_img)
    rgb = sample_image_bilinear(img, uv[:, 0], uv[:, 1])

    usable = hit & np.isfinite(zt_final) & in_frame
    conf = np.zeros(len(p_final), dtype=np.float64)
    conf[usable] = 255.0 * lidar_final[usable] + 64.0 * (1.0 - lidar_final[usable])

    # Geometric quality, independent of where the height came from. The base
    # confidence above only says *which terrain source* was used (255 lidar,
    # 64 DEM), so a ray that grazes the ground -- smearing one camera pixel
    # across a long streak of ortho pixels -- scores exactly as well as a
    # near-nadir one. That is what puts the "comet" streaks in the patch
    # periphery at full confidence, and it makes the confidence layer useless
    # as a gate for matching. `sample_density` measures the actual thing:
    # linear camera pixels backing one ortho pixel, from the Jacobian of the
    # ortho->image map. Below 1.0 the ortho is upsampled from a single camera
    # pixel and carries no independent information.
    density = sample_density(uv.reshape(h, w, 2))
    if min_sample_density > 0.0:
        conf = conf.reshape(h, w) * np.clip(density / min_sample_density, 0.0, 1.0)
    else:
        conf = conf.reshape(h, w)

    rgb = rgb.reshape(h, w, 3)
    return {
        "rgb": rgb,
        "confidence": conf,
        "east": east,
        "north": north,
        "east_min": float(east[0] - gsd / 2),
        "north_max": float(north[0] + gsd / 2),
        "gsd": gsd,
        "agl": agl,
        # Terrain-source coverage, deliberately measured before the geometric
        # term so the two stay separable: this stays "how much of the patch
        # had lidar under it", not "how much was also well-sampled".
        "lidar_coverage_ratio": _ratio(lidar_final.reshape(h, w) > 0.5, usable.reshape(h, w)),
        "sample_density_ratio": _ratio(density >= 1.0, usable.reshape(h, w)),
    }


def run_ortho(
    align: AlignmentResult,
    odom: np.ndarray,
    img_stamps: np.ndarray,
    images: np.ndarray,
    cam: Pinhole,
    dsm: DsmGrid,
    dem: DemField,
    cfg: dict,
    out_dir: Path,
    crs: str,
    R_lidar_to_cam: np.ndarray | None = None,
) -> dict:
    """R_lidar_to_cam ("Rcl" in the FAST-LIVO2 extrinsic_calib config): the
    fixed rotation from the lidar/IMU body frame (what `odom`'s orientation
    is expressed in) to the camera's own optical frame. The camera is
    mounted at a real, non-trivial angle to the lidar on this rig -- Rcl is
    far from identity -- so treating the odometry orientation as the
    camera's own orientation (the previous default) pointed the projected
    ray bundle in a physically wrong direction: verified on the real
    capture, every rendered patch showed a torn "two wings with a gap"
    pattern (the direction where the misoriented bundle grazes/misses the
    ground) instead of a solid ground footprint, on frames whose AGL and
    alignment were otherwise sane. Passing Rcl fixed it (idx 40 in this
    capture: lidar_coverage_ratio 0.0 -> 0.54, confidence mean 3.9 -> 138).
    """
    R_lidar_to_cam = np.eye(3) if R_lidar_to_cam is None else R_lidar_to_cam
    patch_every = int(cfg.get("patch_every_n", 40))
    aa_passes = int(cfg.get("aa_blur_passes", 2))
    img_offset = float(cfg.get("img_time_offset_s", -0.1))
    # Frames where the alignment's odometry -> UTM transform has drifted (e.g.
    # FAST-LIVO2 Z drift through an aggressive turn) compute an implausible
    # AGL and poison the mosaic blend with a wrong pose; skip them rather than
    # accumulate garbage. None disables the filter.
    agl_max_m = cfg.get("agl_max_m")
    agl_max_m = float(agl_max_m) if agl_max_m is not None else None
    # Optional frame cap for fast iteration (None = all frames). Renders only
    # the first ``max_frames`` frames; keep it unset for a full run.
    max_frames = cfg.get("max_frames")
    max_frames = int(max_frames) if max_frames is not None else None

    (out_dir / "patches").mkdir(parents=True, exist_ok=True)
    stats = []

    # global mosaic grid: trajectory extent + one patch radius
    radius = float(cfg.get("patch_radius_m", 130.0))
    gsd = float(cfg.get("patch_gsd_m", 0.5))
    traj = np.array(
        [align.series.apply(odom[i, 1:4][None, :], odom[i, 0])[0] for i in range(len(odom))]
    )
    e_min = np.floor((traj[:, 0].min() - radius) / gsd) * gsd
    e_max = np.ceil((traj[:, 0].max() + radius) / gsd) * gsd
    n_min = np.floor((traj[:, 1].min() - radius) / gsd) * gsd
    n_max = np.ceil((traj[:, 1].max() + radius) / gsd) * gsd
    m_h = int(round((n_max - n_min) / gsd))
    m_w = int(round((e_max - e_min) / gsd))
    acc = np.zeros((m_h, m_w, 3), dtype=np.float64)
    wgt = np.zeros((m_h, m_w), dtype=np.float64)

    n_skipped = 0
    n_frames = len(images) if max_frames is None else min(max_frames, len(images))
    for idx in range(n_frames):
        t_pose = img_stamps[idx] + img_offset
        pos, quat = interpolate_odom_pose(odom, t_pose)
        R_align, _ = align.series.at(t_pose)
        C = align.series.apply(pos[None, :], t_pose)[0]
        R_body = _quat_matrix(quat)
        R_cam_init = R_body @ R_lidar_to_cam.T
        R_cam_utm = R_align @ R_cam_init

        if agl_max_m is not None:
            z_cam, _ = terrain_height(np.array([C[0]]), np.array([C[1]]), dsm, dem)
            agl_precheck = C[2] - z_cam[0]
            if not np.isfinite(agl_precheck) or agl_precheck > agl_max_m:
                n_skipped += 1
                stats.append(
                    {
                        "idx": idx,
                        "t": float(img_stamps[idx]),
                        "agl": float(agl_precheck),
                        "lidar_coverage_ratio": 0.0,
                        "skipped": True,
                    }
                )
                continue

        img = box_blur(images[idx], aa_passes) if aa_passes else images[idx].astype(np.float64)
        res = warp_frame(img, cam, R_cam_utm, C, dsm, dem, cfg)

        # accumulate into the mosaic (confidence-squared weighting)
        r0 = int(round((n_max - res["north_max"]) / gsd))
        c0 = int(round((res["east_min"] - e_min) / gsd))
        ph, pw = res["rgb"].shape[:2]
        w = res["confidence"] ** 2
        acc[r0 : r0 + ph, c0 : c0 + pw] += res["rgb"] * w[..., None]
        wgt[r0 : r0 + ph, c0 : c0 + pw] += w

        if idx % patch_every == 0 or idx == n_frames - 1:
            _save_patch(out_dir, idx, res, crs)

        stats.append(
            {
                "idx": idx,
                "t": float(img_stamps[idx]),
                "agl": res["agl"],
                "lidar_coverage_ratio": res["lidar_coverage_ratio"],
            }
        )
    if agl_max_m is not None:
        print(f"ortho: skipped {n_skipped}/{n_frames} frames with agl > {agl_max_m} m")

    _save_mosaic(out_dir, acc, wgt, e_min, n_max, gsd, stats, crs)
    return {"frames": stats}


def _save_mosaic(
    out_dir: Path,
    acc: np.ndarray,
    wgt: np.ndarray,
    east_min: float,
    north_max: float,
    gsd: float,
    stats: list,
    crs: str,
) -> None:
    rgb = np.where(wgt[..., None] > 0, acc / np.maximum(wgt[..., None], 1e-12), 0.0)
    conf = np.where(wgt > 0, np.sqrt(np.minimum(wgt / max(1, len(stats)), 1.0)) * 255.0, 0.0)
    h, w = rgb.shape[:2]
    transform = rasterio.transform.from_origin(east_min, north_max, gsd, gsd)
    profile = dict(
        driver="GTiff",
        height=h,
        width=w,
        count=3,
        dtype="uint8",
        crs=crs,
        transform=transform,
    )
    rgb8 = np.clip(rgb, 0, 255).astype(np.uint8)
    with rasterio.open(out_dir / "ortho_mosaic.tif", "w", **profile) as dst:
        dst.write(rgb8.transpose(2, 0, 1))
    with rasterio.open(
        out_dir / "ortho_mosaic_conf.tif", "w", **{**profile, "count": 1, "dtype": "uint8"}
    ) as dst:
        dst.write(conf.astype(np.uint8), 1)
    Image.fromarray(rgb8).save(out_dir / "ortho_mosaic.png")
    with open(out_dir / "ortho_stats.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"frames": stats}, f, sort_keys=False)


def _quat_matrix(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _save_patch(out_dir: Path, idx: int, res: dict, crs: str) -> None:
    gsd = res["gsd"]
    transform = rasterio.transform.from_origin(res["east_min"], res["north_max"], gsd, gsd)
    h, w = res["rgb"].shape[:2]
    profile = dict(
        driver="GTiff",
        height=h,
        width=w,
        count=3,
        dtype="uint8",
        crs=crs,
        transform=transform,
    )
    with rasterio.open(out_dir / "patches" / f"patch_{idx:04d}.tif", "w", **profile) as dst:
        dst.write(np.clip(res["rgb"], 0, 255).astype(np.uint8).transpose(2, 0, 1))
    with rasterio.open(
        out_dir / "patches" / f"patch_{idx:04d}_conf.tif",
        "w",
        **{**profile, "count": 1, "dtype": "uint8"},
    ) as dst:
        dst.write(res["confidence"].astype(np.uint8), 1)
    rgb8 = np.clip(res["rgb"], 0, 255).astype(np.uint8)
    Image.fromarray(rgb8).save(out_dir / "patches" / f"patch_{idx:04d}.png")
