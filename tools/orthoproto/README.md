# orthoproto — T14+T15 offline prototype

Offline counterpart of the `geoloc_ortho` runtime node: takes a recorded
FAST-LIVO2 capture and produces true-ortho patches in the geopack CRS,
without any ROS2 install (bag reading is pure-Python via `rosbags`).

Pipeline (`orthoproto run --config <capture.yaml>`, or step by step with
`align` / `dsm` / `ortho`):

0. **check** — georeference self-check (`rtkcheck.py`): compares the
   projected RTK track against the receiver's own Doppler velocity. Run it
   first; `run` does. See "The 2026-08-27 georeference bug" below for why.
1. **align** — fits a `camera_init -> UTM` transform. Default is a global
   pose-graph (`align_pose_graph`, `align.method: pose_graph`): levels the
   odometry frame, then jointly fits a slowly drifting heading + translation
   with a Huber robust loss on the RTK fixes. Rigid by default — a global
   scale exists but is opt-in (`align.estimate_scale`). The legacy independent windowed Kabsch fit (`align_windowed`,
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

- **The site is the Ararat plain, ARMENIA — 39.92 N / 44.83 E, UTM 38N**
  (`EPSG:32638`), terrain ~1050 m above the ellipsoid. This capture is
  MARS-LVIG `AMtown03`, flown 2022-07-18; "AM" is the country code, matching
  `HKisland`/`HKairport` in the same dataset. It is *not* a Maykop flight, and
  `data/missions/maykop-corridor-2026-08.geopack` (44.83 N / 39.92 E, UTM 37N,
  terrain 143-159 m) covers a different place ~430 km away. Build the right
  basemap once, with network:
  `python -m mapprep build --config configs/mapprep/amtown_armenia.yaml --out data/missions/amtown-armenia-2026-08.geopack`.
- **RTK/GPS lat/lon are NOT swapped.** `NavSatFix.latitude` holds ~39.92 (the
  latitude) and `.longitude` holds ~44.83 (the longitude), exactly as
  labelled; `AMtown.kml` agrees once read per the KML spec (`lon,lat,alt`).
  `rtk.swap_latlon` is `false` for this capture and `orthoproto check` guards
  it.
- **RTK altitude is FINE, not frozen.** It reads 1155.45-1155.72 m across the
  whole capture — a 27 cm spread, which is *level flight at 80 m AGL*, not a
  stuck field. `/dji_osdk_ros/gps_position` reports the same track at
  1128.70-1128.90 m; the 26.8 m offset is the geoid undulation (ellipsoidal vs
  orthometric), a cross-check that both are healthy. The alignment still uses
  a single constant (median) altitude as its Z target and lets
  `dsm.anchor_to_dem` set the absolute datum (`dsm.z_datum: dem`), because one
  constant altitude carries no vertical *shape*.
  **Consequence for `accumulate_clouds`:** it must gate points only on
  distance from the drone, never on an absolute Z value — the pre-shift
  frame legitimately has real points around -1000 to -1600 m. An earlier
  version had a `p[:, 2] > -500.0` sanity floor here that silently dropped
  100% of points for this reason (fixed 2026-08-26; see `test_dsm.py`'s
  `test_plane_dsm_survives_negative_utm_z` regression test).
- **`camera_init` is not gravity-aligned.** Its z axis sits ~88.5 degrees off
  vertical (world up lands on roughly `-x` of the odometry frame), so raw
  odometry z sweeping 77 -> 159 m is *horizontal* motion, not altitude drift.
  After leveling, the flight's true vertical extent over the capture is
  **0.63 m**. This is what `pca_up_axis` + `orient_up_from_cloud` are for; do
  not read `odom.pose.position.z` as height.
- **The odometry is metrically correct** — there is no scale error. A plain
  rigid (no-scale) fit of `/aft_mapped_to_init` onto the RTK track leaves
  0.04 m RMS over 5 s windows, 0.17 m over 20 s and 1.02 m over the whole
  51 s overlap (vertical component 0.16 m), with a fitted similarity scale of
  0.999-1.000. Path length over the overlap: 596.4 m by odometry, 592.1 m by
  RTK positions, 592.9 m by integrating `rtk_velocity` — three sources within
  0.7%.
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

## The 2026-08-27 georeference bug (and what it faked)

For one session this capture was read with `rtk.swap_latlon: true`, i.e. its
39.92 N / 44.83 E fixes were interpreted as 44.83 N / 39.92 E. That is not a
harmless relabelling: metres-per-degree differ per axis *and* per latitude, so
the swap stretches northing by 111128/85463 = **1.300** and squeezes easting by
79067/111033 = **0.712**. The distortion is *anisotropic*, which is why no
rigid or similarity fit could reject it — it just leaves residual that looks
like ordinary drift. Everything below was a symptom of that one flag:

| Reported as | Actually |
|---|---|
| "odometry under-measures distance by ~22%", fitted `scale` 1.2261 | the 1.300/0.712 axis distortion, averaged over a mostly east-west track |
| RTK path 706.0 m vs Doppler 592.9 m | 592.1 m vs 592.9 m once projected correctly |
| "the odometry's turn is *reflected*", heading sweeping ~360 degrees through the turn | a lat/lon swap **is** a transpose, i.e. a reflection; corrected, the fitted heading drifts 1.0 degree across the whole capture (-168.7 -> -167.7) |
| pose-graph residual mean 0.81 m / max 5.23 m *with* a free scale | mean **0.14 m** / max **0.25 m** rigid, no scale |
| "RTK altitude is frozen/unreliable" | level flight; the receiver was right |
| geopack built for Maykop | wrong site by ~430 km and ~900 m of elevation |

**What actually pinned the site** (each independent of the others):

1. `AMtown.kml` read per the KML spec (`lon,lat,alt`) puts the mission at
   39.92 N / 44.83 E. Read the other way its lawnmower geometry breaks: the
   connector legs stop being perpendicular to the survey legs (72.9 m link vs
   63.0 m spacing, versus 83.0 m vs 82.9 m the correct way round).
2. `rtk_velocity` (NED: x=North, y=East) against the position derivative:
   slope 1.0031 E / 1.0014 N, r = 0.9998 under the correct reading; 0.693 /
   1.305 under the swap.
3. A raw `/rgb_img` frame NCC-matched against Esri World Imagery at the RTK
   position lands **16 m** from where the fix says it should — with no
   orthorectification and a coarse search grid. Over Maykop there is nothing
   to match: that AOI is flat farmland, this frame is a treed village street.
4. Elevation: 1155.6 m ellipsoidal at 80 m AGL puts the ground near 1050 m.
   The Maykop geopack's DEM reads 143-159 m.

**The guard that now exists:** `orthoproto check` compares the projected track
against the Doppler velocity over short baselines (long ones are diluted by
velocity-integration drift, which is how the previous session's whole-capture
test came out ambiguous). It measures 0.9995 on the corrected capture and
1.2278 on the swapped one. Regression tests: `tests/test_rtkcheck.py`, plus
`test_pose_graph_is_rigid_by_default` and
`test_pose_graph_default_surfaces_a_scale_error_instead_of_hiding_it` in
`tests/test_align.py`.

**The rule this cost us:** a free scale parameter in an alignment is a sink for
georeference bugs. `align.estimate_scale` is `false` by default; turning it on
requires a physical argument for the scale error and a passing
`orthoproto check`.

## Known open items (not yet fixed)

- **No basemap for the real site yet.** `configs/mapprep/amtown_armenia.yaml`
  is written but the geopack has not been built (the build needs network for
  Esri/Bing tiles, Copernicus GLO-30 and Overpass). Until then `dsm` and
  `ortho` cannot run for this capture, and every previously produced
  `data/outputs/geoloc_capture_01_ortho/*` artefact is georeferenced to the
  wrong place and should be regenerated, not trusted.
- Genuine RTK fix quality is still unmeasured: `NavSatFix.status.status` is
  `0` on every message in this capture, so fixed/float/single is not
  recoverable from the bag. Worth resolving separately for T09.
- The first ~30% of a capture's clouds/frames (before the first alignment
  window's centre) get the *nearest* window's transform rather than a real
  local fit (`TransformSeries.at` clamps, it doesn't extrapolate) — honest
  but lower-quality at the leading edge of a capture.
- `ortho.agl_max_m` (100 m) is a safety net, not load-bearing: with the
  leveling in place the computed AGL tracks the KML's 80 m. The earlier
  387 m-AGL mosaic smear was the odometry frame's tilt going uncorrected, and
  the leveling step fixed it — but the numbers in that investigation were
  taken under the wrong georeference and are worth re-measuring once the
  correct geopack exists.
