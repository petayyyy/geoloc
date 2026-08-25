# P5 — Data / Simulation Engineer

**Read `P0-common.md` first.**

**Tasks:** T08 (datasets), T09 (basemap & bias), T10–T11 (OrthoSim), T12 (metrics harness), T28–T30 (Isaac Sim & scenarios).

---

## Your role

You build the instruments the rest of the team measures with. If your instruments lie, everyone downstream optimizes toward the wrong target and nobody finds out for weeks. Two of your tasks (T10, T29) have automatic self-checks specifically to prevent that.

## The rule that governs everything you build

> **The query is rendered from one imagery source; the map comes from a different one.**

Esri ↔ Bing ↔ Mapbox ↔ NAIP of a different year ↔ different seasons. If you texture the query from the same basemap you match against, the domain gap is zero, the metrics come out fantastic, and the deception surfaces only at level B — weeks later.

This applies to **both** OrthoSim (T10) and the Isaac terrain (T29). Both have blocking self-checks: `T10-U-04` fails the generator if the providers match; `T29-U-03` refuses to build a scene textured with provider A. Keep them.

The upside is that cross-provider pairs give a **natural** domain gap — real differences in shadows, vegetation, construction and processing that no renderer reproduces.

## Level A — OrthoSim (T10, T11)

The workhorse: runs on every commit, on CPU, deterministically. Ground truth is exact because it is *specified*, not measured.

The most important set is **`adversarial`**. It exists not for the system to pass, but to measure **how often it is confidently wrong**: crop grids and greenhouses (periodic → correct-but-period-shifted match), identical warehouse rows, open water, full snow, new construction against a 2-year-old map, unbroken forest, symmetric interchanges. Target: accepted fixes with >50 m error **< 0.5%**. The refusal rate on these scenes can be arbitrarily high — that is correct behavior.

Two design rules:

- **GT is fixed before augmentation and never recomputed after.** An augmentation that perturbs geometry (attitude error, AGL error) changes the pipeline's *input*, not the truth.
- **OrthoSim feeds the same code that runs on the aircraft.** No "simplified version for tests."

For T11, the Avia cloud synthesis exists mainly to test **coverage deficit** — the ~30% of patch width with no lidar. Get the FOV (70.4° × 77.2°), range and point rate right; the exact non-repetitive pattern may be approximated. And keep motion distortion in: without it, deskew (T14) is never tested at all.

Note honestly that synthetic clouds are too clean — FAST-LIVO2 performs better on them than in reality. **Never use OrthoSim to evaluate odometry.**

## Level B — replay (T08, T09)

This is where the project's real accuracy numbers come from. Everything else is a proxy.

**Replay runs through the real ROS 2 graph**, not a script calling functions directly. The distinction matters: replay must catch integration bugs — QoS, timestamps, tf, message ordering — not just algorithm bugs.

Two specific traps:

- **Lidar message format.** FAST-LIVO2 uses each point's time field for deskew. Converting to plain `PointCloud2` drops it and silently changes odometry behavior. `T08-I-01` exists to catch this.
- **T09 is not a chore, it is a measurement of a fundamental parameter.** The basemap's georeferencing bias is the **floor on the whole system's accuracy**. Until it is measured, no localization error figure means anything — you cannot tell what is the matcher and what is the map. Measure it with an estimator **independent of our matcher** (phase correlation plus manual control points), or the error will circle back on itself.

If bias scatter across a region exceeds 8 m, that triggers a scope change in T23 (bias becomes an estimated state). Report it loudly.

Side benefit: MARS-LVIG (2023) versus available basemaps gives a **free, realistic map-staleness domain gap**. Do not hunt for a closer-dated basemap — document the difference and use it.

## Level E — Isaac Sim (T28, T29, T30)

Isaac replaces both Gazebo and a separate photoreal renderer. Its RTX Lidar takes a configuration profile for non-repetitive patterns, so the Avia model is configuration rather than a plugin.

**What Isaac is for:** the closed loop. EKF2 behavior on `GPS_INPUT`, failsafe, latency, mode switching, response to an injected false fix, controlled sun/season/weather sweeps, and — uniquely — real parallax and occlusion from OSM-extruded buildings, which OrthoSim cannot produce.

**What Isaac is not for:** predicting field accuracy (its domain gap is synthetic and controlled) or evaluating FAST-LIVO2 (no real intensity, atmosphere or multipath).

**S-05 is the most important test in the project.** Three mutually consistent false fixes carrying *plausible* quality metrics — exactly what a periodic structure produces in reality. Make the injected `fake_quality` values pass every gate except the ones that should fire; a false fix with `n_inliers=5` tests nothing. S-04 and S-05 are blocking: failing either means the aircraft does not fly.

When schedules slip and a blocking scenario will not pass, the temptation to soften its criterion peaks. `accepted_false_fixes: 0` is not negotiable. A failing S-05 is a T22 problem to solve, not a scenario to weaken.

## T12 — metrics harness

**One metric, one implementation, used by every level.** Otherwise numbers from different levels are incomparable and the argument about which to believe has no resolution.

Three things people get wrong:

- **`A@d` is computed over accepted fixes, not all attempts.** Otherwise it conflates accuracy with availability and becomes uninterpretable. Availability is `acceptance_rate`, separately.
- **Always break down by terrain class.** An average across classes mixes urban, where everything works, with forest, where nothing does — and hides both.
- **`NEES` is the only way to know whether the covariance is honest.** Implement it carefully and test it against a deliberately understated covariance, which must show inflated NEES.

Reports are self-contained HTML — they get forwarded and opened without an environment. And `failures/` (worst 50 cases with visualized correspondences) is not decoration: matcher debugging does not work without looking at specific failures.

## Your deliverables

Instruments that tell the truth, including when the truth is unflattering — plus the automatic self-checks that keep them honest when someone edits a config in a hurry.
