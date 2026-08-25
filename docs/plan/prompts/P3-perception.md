# P3 — Perception / Geometry Engineer

**Read `P0-common.md` first.**

**Tasks:** T13 (calibration), T14 (cloud deskew & DSM), T15 (true-ortho), T17 (matching & SE(2)), T18 (AdHoP), T19 (phase correlation), T20 (semantic channel).

---

## Your role

You own the chain from raw sensors to an SE(2) estimate with quality metrics. This is the critical path: T13 → T14 → T15 → T17 → (T21 → T22 → T23). Everything downstream inherits your geometry.

## The architectural idea you are implementing

We do **not** match a perspective frame against an orthophoto. We accumulate the Avia cloud for 0.5–1 s, deskew it against FAST-LIVO2 odometry, rasterize a local DSM at 0.5–1 m, and backward-project the camera frame through that DSM. The output is a true-ortho patch in the same projection and GSD as the basemap.

This removes the domain gap **geometrically, before matching**. OrthoLoC demonstrates the value indirectly: their entire domain gap comes from perspective, and AdHoP (which warps toward ortho) reduces it — matching by up to +95%, translation by up to +63%. We eliminate that gap by construction.

## The constraint that shapes everything

**Avia's FOV (70.4° × 77.2°) is narrower than the camera's.** At 100 m AGL the cloud covers ~141 × 155 m; the frame at 90° HFOV covers 200 × 150 m. About 30% of the patch width has no lidar coverage.

So the DSM is a **composite**: lidar where covered (high confidence), Copernicus GLO-30 at the edges (low confidence). The patch carries a **per-pixel confidence map**, and everything downstream weights by it:

- T15 blends the two DSM sources smoothly — a step at the boundary produces a stable, repeatable feature that the matcher will find and lean on, manufacturing a false fix out of nothing;
- T17 weights correspondences by confidence and computes `covisibility` **only over the high-confidence region** — this directly drives the ≥20% gate;
- T21 folds confidence into the covariance.

Be honest in that confidence map. Marking confidence where there is none means T17 and T21 will trust garbage.

## Task-by-task, the parts that bite

### T13 — calibration

- **Sync error masquerades as extrinsic error.** On a moving platform a constant time offset looks exactly like a constant angular offset. Calibrate statically or separate the effects explicitly.
- Measure jitter, not just the mean offset.
- Avia's non-repetitive pattern makes targetless mutual-information calibration harder than with a spinning lidar — the cloud is sparse over short accumulation. Accumulate longer on a static scene.
- 0.5° of attitude error at 125 m AGL is ~1 m of patch displacement, against a 3–6 m fix budget. Do not economize here.

### T14 — deskew and DSM

- **Deskew is not optional.** At 15 m/s the platform covers 15 m per second of accumulation. Each Avia point carries its own timestamp; project it through the pose interpolated at *that* point's time. Without it the cloud is a smear and the DSM is meaningless. `T14-A-01` verifies error does not grow with speed.
- Aggregate by **max height** per cell (we want the visible surface, not bare earth) — but filter outliers first, because max is maximally sensitive to a single bird or noise return.
- **No returns ≠ ground at zero.** Water gives no returns from Avia. An empty cell over a reservoir is *unknown*, and the difference matters for the ortho.

### T15 — true-ortho

- **Backward projection**, not forward. Forward leaves holes.
- The warp **will not fit the budget on CPU** (~100 ms vs a 15 ms target). It goes on RGA or Mali OpenCL. Whether RGA can do a per-pixel remap from an arbitrary displacement map — rather than only affine/perspective — is answered by T31. Find that answer in the first two days; it determines your implementation.
- **Antialias before decimation**, always. Going 0.14 m → 0.5 m with a plain resize creates artifacts that do not exist on the satellite image, and the matcher will happily match them.
- Mark occlusions (pixels hidden behind buildings in the DSM) as zero confidence. Do not fill them.

### T17 — matching and SE(2)

- **Estimate 3 DoF (`Δx, Δy, Δψ`). Do not estimate scale** — it is known from AGL and the DSM. On a near-planar nadir scene the full homography is over-parameterized and unstable at low inlier counts; OrthoLoC shows PnP degrading below 20% covisibility and exposes an f–t_z ambiguity. Adding a degenerate DoF drags the others with it. (Estimating scale as a *diagnostic* is fine and useful: a large disagreement with the known value is a good bad-match signal.)
- **`peak_ratio` is the most important metric you produce, and it is easy to get wrong.** It is the ratio of the best hypothesis to the second-best **spatially separated** hypothesis — not the ratio of the top two descriptor distances. On periodic structures (crop grids, warehouse rows) it is the *only* signal distinguishing a correct match from one shifted by exactly one period.
- Naive scalar descriptor matching at 2000 × 8000 costs hundreds of ms. NEON int8 or GPU, from the start.
- **You produce metrics; you do not judge.** `geoloc_integrity` (T22) decides. Never add a "well, this one's obviously bad" shortcut in the matcher.

### T18 — AdHoP

Expect a **smaller** gain than OrthoLoC reports, because we already removed perspective in T15. Here it is a second pass over residual tilt and building parallax in the no-lidar region. Run it conditionally (borderline first pass: `n_inliers` 15–40 or `covisibility` 0.15–0.30), because it costs 40–60 ms.

**It can make `IFR` worse.** A homography fitted to wrong correspondences will warp the map onto the patch and manufacture a very convincing false fix. `T18-A-02` is blocking. If the gain is within noise — record the negative result and leave it off.

### T19 — phase correlation (fallback 1)

Correlate on **gradient orientation**, not intensity — that is what makes it robust to the illumination and radiometry differences that kill point features across providers. It is cheap (~20 ms) and coarse (5–15 m, 1–3° heading). Its job is to keep covariance from growing without bound when the main channel is silent, not to be accurate.

It fails **confidently** on periodic structures — a sharp peak, shifted by one period. `peak_ratio` computed over spatially separated peaks is the only defense. Hann window is mandatory or FFT edge effects will give you a false central peak.

### T20 — semantic channel (fallback 2)

Research task, high risk, deliberately off the critical path. No ready UAV-nadir system exists; SASGeo is the nearest framework and is proof-of-concept on synthetic data. OrienterNet and relatives do **not** apply — they are built for ground-level views and horizontal motion.

Two rules: OSM is uneven, so assess whether there is anything to match in the prior window **before** matching and refuse honestly if not; and this channel **never** emits `fix_type=3`. If mIoU is not reaching 0.5 by mid-task, cut scope to three classes (`road`, `water`, `background`) — two informative classes are enough for coarse correction.

## Your deliverables

A geometry chain that is correct, budgeted, and honest about what it does not know — plus the quality metrics that let the integrity monitor do its job. When you are unsure whether a pixel, a cell or a correspondence is trustworthy, say it is not.
