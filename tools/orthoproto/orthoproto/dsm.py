"""DSM from /cloud_registered (T14, offline).

Clouds arrive already registered into the FAST-LIVO2 `camera_init` frame
(deskew is FAST-LIVO2's job), so the only transform needed is the
RTK-anchored T(t) from `align`. Rasterization: per-cell max height plus point
count and dispersion; outlier filter by 5x5 median; holes stay nodata -- a
cell with no returns is *unknown*, never zero (project rule 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import yaml
from PIL import Image

from .align import AlignmentResult, TransformSeries, interpolate_odom_pose

NODATA = -9999.0


@dataclass
class DsmGrid:
    gsd: float
    east_min: float
    north_max: float
    height: int
    width: int
    z: np.ndarray  # (H, W) float32, NODATA where no coverage
    count: np.ndarray  # (H, W) uint32 point observations per cell
    dispersion: np.ndarray  # (H, W) float32 height std per cell
    confidence: np.ndarray  # (H, W) uint8

    def pixel_centre(self, row: int, col: int) -> tuple[float, float]:
        return (
            self.east_min + (col + 0.5) * self.gsd,
            self.north_max - (row + 0.5) * self.gsd,
        )

    def save(self, path: Path, z_path: Path, conf_path: Path) -> None:
        transform = rasterio.transform.from_origin(
            self.east_min, self.north_max, self.gsd, self.gsd
        )
        profile = dict(
            driver="GTiff",
            height=self.height,
            width=self.width,
            count=1,
            dtype="float32",
            crs="EPSG:32637",
            transform=transform,
            nodata=NODATA,
        )
        with rasterio.open(z_path, "w", **profile) as dst:
            dst.write(self.z.astype(np.float32), 1)
        with rasterio.open(conf_path, "w", **{**profile, "dtype": "uint8", "nodata": 255}) as dst:
            dst.write(self.confidence, 1)
        meta = {
            "format": "orthoproto-dsm-v1",
            "gsd": self.gsd,
            "east_min": self.east_min,
            "north_max": self.north_max,
            "height": self.height,
            "width": self.width,
            "coverage_ratio": float(np.mean(self.count > 0)),
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, sort_keys=False)

    @classmethod
    def load(cls, path: Path, z_path: Path, conf_path: Path) -> DsmGrid:
        with open(path, encoding="utf-8") as f:
            meta = yaml.safe_load(f)
        with rasterio.open(z_path) as src:
            z = src.read(1).astype(np.float32)
        with rasterio.open(conf_path) as src:
            conf = src.read(1).astype(np.uint8)
        count = np.zeros_like(conf, dtype=np.uint32)
        dispersion = np.zeros_like(conf, dtype=np.float32)
        return cls(
            gsd=float(meta["gsd"]),
            east_min=float(meta["east_min"]),
            north_max=float(meta["north_max"]),
            height=int(meta["height"]),
            width=int(meta["width"]),
            z=z,
            count=count,
            dispersion=dispersion,
            confidence=conf,
        )

    def preview_png(self, path: Path) -> None:
        z = np.where(self.count > 0, self.z, np.nan)
        lo, hi = np.nanpercentile(z, 2), np.nanpercentile(z, 98)
        if hi - lo < 1e-6:
            hi = lo + 1.0
        img = np.clip((z - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(path)


def accumulate_clouds(
    series: TransformSeries,
    odom: np.ndarray,
    clouds,
    gsd: float,
    margin_m: float,
    range_max_m: float,
    odom_bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Stream clouds, transform by T(t), aggregate per-cell stats.

    Returns (z_max, count, z2_sum, obs_weights, east_min, north_max).
    """
    east_min, east_max, north_min, north_max = odom_bounds
    east_min = np.floor((east_min - margin_m) / gsd) * gsd
    east_max = np.ceil((east_max + margin_m) / gsd) * gsd
    north_min = np.floor((north_min - margin_m) / gsd) * gsd
    north_max = np.ceil((north_max + margin_m) / gsd) * gsd
    width = int(round((east_max - east_min) / gsd))
    height = int(round((north_max - north_min) / gsd))

    z_max = np.full((height, width), -np.inf, dtype=np.float32)
    count = np.zeros((height, width), dtype=np.uint32)
    z2_sum = np.zeros((height, width), dtype=np.float64)
    z_max_flat = z_max.ravel()
    count_flat = count.ravel()
    z2_flat = z2_sum.ravel()

    for t, xyz in clouds:
        R, tr = series.at(t)
        p = xyz.astype(np.float64) @ R.T + tr
        # range gate: keep points within range_max of the drone position.
        # No absolute-Z sanity floor here on purpose: align.series is not
        # DEM-shifted yet at this point (anchor_to_dem runs after this pass),
        # so raw Z sits near the RTK altitude used as the fit's Z target,
        # which for this capture is the known-unreliable frozen ~1155.6 m
        # (see z_datum="dem" in the capture config) -- an absolute floor here
        # would reject real points for a datum reason, not a data-quality one.
        # The drone-relative distance gate is the meaningful physical filter.
        drone, _ = interpolate_odom_pose(odom, t)
        drone_utm = R @ drone + tr
        d2 = (p - drone_utm) ** 2
        keep = d2.sum(axis=1) <= range_max_m**2
        p = p[keep]
        if len(p) == 0:
            continue
        rows = ((north_max - p[:, 1]) / gsd).astype(np.int64)
        cols = ((p[:, 0] - east_min) / gsd).astype(np.int64)
        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        rows, cols = rows[valid], cols[valid]
        z = p[valid, 2].astype(np.float32)
        if len(z) == 0:
            continue
        flat = rows * width + cols
        np.maximum.at(z_max_flat, flat, z)
        np.add.at(count_flat, flat, 1)
        np.add.at(z2_flat, flat, z.astype(np.float64) ** 2)

    return z_max, count, z2_sum, np.zeros_like(count, dtype=np.float32), east_min, north_max


def build_dsm(
    series: TransformSeries,
    odom: np.ndarray,
    clouds,
    cfg: dict,
) -> DsmGrid:
    gsd = float(cfg.get("dsm_gsd_m", 0.5))
    margin = float(cfg.get("margin_m", 150.0))
    range_max = float(cfg.get("range_max_m", 190.0))
    median_win = int(cfg.get("median_win", 5))
    outlier_z = float(cfg.get("outlier_threshold_m", 3.0))
    dense_min = int(cfg.get("dense_min_points", 4))
    sparse_min = int(cfg.get("sparse_min_points", 1))

    # grid bounds from the odom trajectory mapped into UTM
    traj = np.array(
        [series.apply(odom[i, 1:4][None, :], odom[i, 0])[0] for i in range(len(odom))]
    )
    bounds = (traj[:, 0].min(), traj[:, 0].max(), traj[:, 1].min(), traj[:, 1].max())

    z_max, count, z2_sum, _obs, east_min, north_max = accumulate_clouds(
        series, odom, clouds, gsd, margin, range_max, bounds
    )

    # median filter over valid cells (outlier suppression), scipy-free
    import warnings

    h, w = count.shape
    pad = median_win // 2
    z_pad = np.pad(z_max, pad, mode="constant", constant_values=np.nan)
    cnt_pad = np.pad(count, pad, mode="constant", constant_values=0)
    win = np.lib.stride_tricks.sliding_window_view(
        z_pad, (median_win, median_win)
    ).reshape(h, w, -1)
    wcnt = np.lib.stride_tricks.sliding_window_view(
        cnt_pad, (median_win, median_win)
    ).reshape(h, w, -1)
    win = np.where(wcnt > 0, win, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(win, axis=2).astype(np.float32)

    valid = count > 0
    outlier = np.abs(z_max - median) > outlier_z
    z_filt = np.where(valid & outlier, median, z_max)
    z_filt[~valid] = NODATA

    # confidence: density graded, dispersion and outliers penalize
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.where(count > 0, z2_sum / count - (z_max ** 2), 0.0)
    dispersion = np.sqrt(np.clip(var, 0, None)).astype(np.float32)

    confidence = np.zeros((h, w), dtype=np.uint8)
    confidence[count >= dense_min] = 255
    confidence[(count >= sparse_min) & (count < dense_min)] = 128
    confidence[(count >= dense_min) & (dispersion > 2.0)] = 192
    confidence[valid & outlier] = 96

    return DsmGrid(
        gsd=gsd,
        east_min=float(east_min),
        north_max=float(north_max),
        height=h,
        width=w,
        z=z_filt,
        count=count,
        dispersion=dispersion,
        confidence=confidence,
    )


def anchor_to_dem(dsm: DsmGrid, dem_tif: Path, cfg: dict) -> float:
    """Vertical offset aligning the lidar DSM ground with the geopack DEM.

    The DJI RTK altitude in this capture is unreliable (see README), so the
    DSM is shifted so its dense ground cells match Copernicus GLO-30
    (ellipsoid) over the same area. Returns the shift to ADD to DSM heights.
    """
    dense_min = int(cfg.get("dense_min_points", 4))
    mask = dsm.count >= dense_min
    if int(mask.sum()) < 100:
        return 0.0
    z = dsm.z[mask]
    lo, hi = np.percentile(z, [5, 20])
    ground = np.median(z[(z >= lo) & (z <= hi)])

    rows, cols = np.where(mask)
    east = dsm.east_min + (cols + 0.5) * dsm.gsd
    north = dsm.north_max - (rows + 0.5) * dsm.gsd
    with rasterio.open(dem_tif) as src:
        dem_vals = np.array([v[0] for v in src.sample(list(zip(east, north)))])
    dem_vals = dem_vals[~np.isnan(dem_vals)]
    if len(dem_vals) < 50:
        return 0.0
    dem_med = float(np.median(dem_vals))
    shift = dem_med - float(ground)
    return shift


def run_dsm(
    align: AlignmentResult,
    odom: np.ndarray,
    clouds,
    cfg: dict,
    out_dir: Path,
    dem_tif: Path,
) -> tuple[DsmGrid, AlignmentResult]:
    dsm = build_dsm(align.series, odom, clouds, cfg)
    z_datum = cfg.get("z_datum", "dem")
    shift = 0.0
    if z_datum == "dem":
        shift = anchor_to_dem(dsm, dem_tif, cfg)
        dsm.z = np.where(dsm.count > 0, dsm.z + shift, NODATA)
        align.z_shift = shift
        align.series.shift_z(shift)
    align.z_datum = z_datum

    out_dir.mkdir(parents=True, exist_ok=True)
    dsm.save(out_dir / "dsm.yaml", out_dir / "dsm.tif", out_dir / "dsm_confidence.tif")
    dsm.preview_png(out_dir / "dsm_preview.png")
    align.save(out_dir / "align_final.yaml")
    return dsm, align
