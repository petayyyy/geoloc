# orthoproto — T14+T15 offline prototype

Offline counterpart of the `geoloc_ortho` runtime node: takes a recorded
FAST-LIVO2 capture and produces true-ortho patches in the geopack CRS,
without any ROS2 install (bag reading is pure-Python via `rosbags`).

Pipeline (`orthoproto run --config <capture.yaml>`, or step by step with
`align` / `dsm` / `ortho`):

1. **align** — fits a piecewise-rigid `camera_init -> UTM` transform by
   windowed Kabsch alignment between FAST-LIVO2 odometry and the RTK ground
   truth (`align.py`).
2. **dsm** — accumulates `/cloud_registered` through that transform into a
   gridded height-above-ellipsoid surface (`dsm.py`).
3. **ortho** — backward-projects each `/rgb_img` frame onto the DSM (falling
   back to the geopack's Copernicus DEM outside lidar coverage) to produce
   true-ortho patches and a mosaic (`ortho.py`).

## Data quirks of the `geoloc_capture_01` capture

These are capture-specific facts, not general FAST-LIVO2/DJI behaviour —
recorded here because they're config (`configs/orthoproto/geoloc_capture_01.yaml`),
not code, per the project's "thresholds are config" rule.

- **RTK/GPS lat/lon are swapped.** On `/dji_osdk_ros/rtk_position` and
  `/dji_osdk_ros/gps_position`, `NavSatFix.latitude` actually holds the
  longitude value (~39.9x) and `.longitude` holds the latitude value
  (~44.8x) — confirmed against the real geographic location (Maykop,
  44.8°N/39.9°E). `bagio.Capture.read_rtk(swap_latlon=True)` and the
  `rtk.swap_latlon` config flag handle this; never assume the fields are
  correctly labelled in a new capture without checking.
- **RTK altitude is frozen/unreliable** (~1155.6 m constant through the
  capture — not a plausible ellipsoidal height for this region). Because of
  this, `align_windowed` fits a *robust constant* altitude (the median RTK
  altitude across the whole capture) as every window's Z target rather than
  each point's own reported altitude — the horizontal (E/N) fit is what's
  trusted from RTK. The resulting absolute Z is deliberately wrong until
  `dsm.anchor_to_dem` shifts the whole DSM (and the alignment series, via
  `TransformSeries.shift_z`) to match the geopack's Copernicus DEM instead.
  Config: `dsm.z_datum: dem` (recommended for this capture) vs `rtk`.
  **Consequence for `accumulate_clouds`:** it must gate points only on
  distance from the drone, never on an absolute Z value — the pre-shift
  frame legitimately has real points around -1000 to -1600 m. An earlier
  version had a `p[:, 2] > -500.0` sanity floor here that silently dropped
  100% of points for this reason (fixed 2026-08-26; see `test_dsm.py`'s
  `test_plane_dsm_survives_negative_utm_z` regression test).
- **`/rgb_img` is `bgr8`**, not `rgb8`. `bagio.decode_image_rgb` converts it
  once at read time so everything downstream (patches, mosaic, any future
  colour-based matching against the RGB satellite basemap) is RGB.
- **Camera model**: FAST-LIVO2's vikit `Pinhole` with `d0..d3 = (k1, k2, p1,
  p2)` distortion, at the VIO image scale (0.25x of the full sensor,
  `camera_MARS_LVIG_AM.yaml`). `/rgb_img` is the raw (distorted) VIO frame —
  `camera.py`'s projection goes through the full distortion model rather
  than assuming undistorted input.

## Bugs fixed 2026-08-26 (found running the pipeline on real data for the first time)

- **`dsm.py` `accumulate_clouds`**: an absolute-Z sanity floor (`p[:, 2] >
  -500.0`) was applied before `anchor_to_dem` ever ran, so it silently
  rejected 100% of points (real pre-shift Z sat near the frozen ~-1155 m RTK
  anchor). Dropped; the drone-relative distance gate is the real filter.
- **`bagio.py` `read_images`**: `/rgb_img` is `bgr8`; channels were never
  reversed, so every saved patch/mosaic had red and blue swapped.
- **`ortho.py` `terrain_height` / `DemField.bilinear`**: bilinear
  interpolation blended a real height with the `-9999` nodata sentinel at
  any DSM/DEM coverage boundary, producing a finite-looking but wildly wrong
  height (observed: -9914 m against a real ~150 m ground) instead of being
  rejected. Fixed by invalidating the whole interpolated cell when *any* of
  the 4 corner samples is nodata, not just checking the blended result.
- **`ortho.py` `warp_frame`** (the most severe of the four): the
  camera-frame -> world-frame ray direction used `R_cam_utm.T`, the *same*
  rotation as the world -> camera step just above it (`(p - C) @
  R_cam_utm`, which is `R_cam_utm.T @ (p - C)` for a row vector) instead of
  its inverse. Composing them was `(R_cam_utm.T)**2`, not the identity, so
  even a geometrically correct terrain hit reprojected to a wildly
  different pixel, and iterating that in `warp_frame`'s 3-round refinement
  diverged every ray to nan/inf (verified on the real capture: 1208 genuine
  hit-and-in-frame pixels in one 520x520 patch fell to 0 after just one
  more iteration). The existing test fixture (`R_NADIR`, a diagonal matrix)
  couldn't catch this because a diagonal matrix is its own transpose --
  `test_ray_direction_round_trip` uses a random general rotation instead.

## Known open items (not yet fixed)

- Windowed RTK alignment residuals on the real capture run ~7-9 m mean
  (up to ~17 m max) — plausibly genuine RTK/GPS quality in this segment
  (fix status wasn't captured alongside `rtk_position`), but `fit_rigid3`
  is a plain least-squares Kabsch fit with no outlier rejection, so a
  RANSAC/IRLS pass is worth trying before trusting this as ground truth for
  T09 bias measurement.
- The first ~30% of a capture's clouds/frames (before the first alignment
  window's centre) get the *nearest* window's transform rather than a real
  local fit (`TransformSeries.at` clamps, it doesn't extrapolate) — honest
  but lower-quality at the leading edge of a capture.
- **`ortho_mosaic.tif` is smeared for this capture** (verified 2026-08-26,
  after all four bugs above were fixed): individual patches are sharp and
  correctly geo-referenced where the alignment is good (e.g.
  `patches/patch_0200.png`, real buildings/road/vegetation, crisp), but
  `run_ortho`'s computed AGL climbs from a sane ~68 m at the start of the
  capture to 387 m by the end (`ortho_stats.yaml`; 204/396 frames > 150 m
  AGL) -- odometry Z-drift during the capture's 180-degree turn (see
  `align.py`'s own docstring), which the constant-per-window RTK-altitude
  target in `align_windowed` doesn't correct. Roughly half the frames blend
  into the mosaic from a substantially wrong camera pose. Not a code bug in
  the pipeline itself -- a data/alignment-robustness limitation of this
  particular short, turn-heavy capture. Fixing it properly is T09/T17
  territory (bias measurement, more robust alignment); a cheap interim
  mitigation would be excluding high-AGL frames from `_save_mosaic`'s
  accumulation.
