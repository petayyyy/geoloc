# orthoproto — T14+T15 offline prototype

Offline counterpart of the `geoloc_ortho` runtime node: takes a recorded
FAST-LIVO2 capture and produces true-ortho patches in the geopack CRS,
without any ROS2 install (bag reading is pure-Python via `rosbags`).

Pipeline (`orthoproto run --config <capture.yaml>`, or step by step with
`align` / `dsm` / `ortho`):

1. **align** — fits a `camera_init -> UTM` transform. Default is a global
   pose-graph (`align_pose_graph`, `align.method: pose_graph`): levels the
   odometry frame (fixes Z/tilt drift), then jointly fits a slowly drifting
   heading + translation with a *global scale* and a Huber robust loss on the
   RTK fixes. The legacy independent windowed Kabsch fit (`align_windowed`,
   `align.method: windowed`) is retained for comparison (`align.py`).
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
  this, both `align_windowed` and `align_pose_graph` fit a *robust constant*
  altitude (the median RTK altitude across the whole capture) as the Z target
  rather than each point's own reported altitude — the horizontal (E/N) fit is
  what's trusted from RTK. The resulting absolute Z is deliberately wrong until
  `dsm.anchor_to_dem` shifts the whole DSM (and the alignment series, via
  `TransformSeries.shift_z`) to match the geopack's Copernicus DEM instead.
  Config: `dsm.z_datum: dem` (recommended for this capture) vs `rtk`.
  **Consequence for `accumulate_clouds`:** it must gate points only on
  distance from the drone, never on an absolute Z value — the pre-shift
  frame legitimately has real points around -1000 to -1600 m. An earlier
  version had a `p[:, 2] > -500.0` sanity floor here that silently dropped
  100% of points for this reason (fixed 2026-08-26; see `test_dsm.py`'s
  `test_plane_dsm_survives_negative_utm_z` regression test).
- **The replayed FAST-LIVO2 odometry under-measures distance by ~22%.** The
  odometry's own ground speed reads ~11.7 m/s while the RTK track advances at
  ~14.1 m/s (verified directly on `/aft_mapped_to_init` vs `/rtk_position`,
  and by fitting: a rigid per-window Kabsch fit leaves a ~7-9 m residual that
  a *similarity* fit collapses to ~0.16 m on the straight legs). This is why
  the pose-graph alignment estimates a `scale` (~1.227 for this capture)
  instead of a rigid SE(3): without it, even a perfect rigid fit is wrong by
  metres. The scale is stored on `TransformSeries.scale` and applied in
  `apply()` (both positions and lidar clouds share the same camera_init
  frame).
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
- **`ortho.py` `run_ortho`** (found 2026-08-27, while investigating why an
  AGL-based mosaic filter alone didn't fix the smear below): the camera's
  orientation was taken directly from the odometry (lidar/IMU body frame)
  quaternion, `R_cam_init = _quat_matrix(quat)`, with no camera-to-body
  extrinsic correction. This rig's `Rcl` (in FAST-LIVO2's
  `MARS_LVIG_AMtown.yaml` `extrin_calib`) is far from identity, so every
  rendered ray bundle pointed in a physically wrong direction -- every
  patch showed a torn "two wings with a gap" pattern instead of a solid
  footprint, even on frames with sane AGL and good alignment. Fixed by
  passing `Rcl` through from the capture config (`camera.Rcl`) into
  `run_ortho(..., R_lidar_to_cam=Rcl)`, composing `R_cam_init = R_body @
  Rcl.T`. Verified on the real capture (frame idx 40): lidar_coverage_ratio
  0.0 -> 0.54, confidence mean 3.9 -> 138.
- **`ortho.py` `warp_frame`** (the most severe of the first four bugs): the
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

- Windowed RTK alignment residuals on the real capture used to run ~7-9 m mean
  (up to ~17 m max). This was **not** primarily RTK noise: it was the
  odometry's ~22% scale error (see "Data quirks" above) that a rigid fit
  cannot absorb. The pose-graph alignment estimates that scale, dropping the
  mean horizontal residual to ~0.8 m on this capture (max ~5 m, concentrated
  on the short, distorted turn and the post-turn return leg where the
  odometry scale drifts slightly — the pose-graph still uses a *single global*
  scale, so a residual few metres remain there). Genuine RTK fix quality
  (fix status was never captured) is still worth measuring separately for T09.
- The first ~30% of a capture's clouds/frames (before the first alignment
  window's centre) get the *nearest* window's transform rather than a real
  local fit (`TransformSeries.at` clamps, it doesn't extrapolate) — honest
  but lower-quality at the leading edge of a capture.
- **`ortho_mosaic.tif` was smeared for this capture** (first noticed
  2026-08-26, after the first four bugs above were fixed): `run_ortho`'s
  computed AGL climbed from a sane ~68 m at the start of the capture to
  387 m by the end (`ortho_stats.yaml`; 204/396 frames > 150 m AGL). Root
  cause was the odometry frame's tilt + Z drift through the capture's
  180-degree turn (see `align.py`'s own docstring), which the constant
  per-window RTK-altitude target in `align_windowed` didn't correct.
  **Fixed 2026-08-27** by the pose-graph alignment: leveling the odometry
  frame (via `pca_up_axis` + `orient_up_from_cloud`, exactly as before) plus
  the joint global fit brings the computed AGL to ~63-97 m (mean ~81 m,
  matching AMtown.kml's ~80 m) with **0 frames** over `agl_max_m` — the
  `agl_max_m` filter is now a safety net, not the thing keeping the mosaic
  from tearing. The earlier cheap mitigation (`ortho.agl_max_m` = 100 m) is
  still in the config but no longer skips any frame.
- The odometry's turn is *reflected* relative to the RTK track (the LIO turns
  in the opposite rotational sense through the 180-degree return), so the
  pose-graph's heading sweeps ~360 degrees through the turn to map it. This
  is correct for the *positions* (residual ~0.8 m) but means the alignment
  rotation is not physically meaningful during the ~10 s turn apex; frames
  there are lower-confidence by construction.
