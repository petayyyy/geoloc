# P4 — Estimation / Fusion Engineer

**Read `P0-common.md` first.**

**Tasks:** T21 (covariance model), T22 (integrity monitor), T23 (GTSAM fusion), T24 (modes & recovery).

---

## Your role

You own the decision layer and the estimate that reaches the autopilot. T22 is the system's main safety interlock; T21 determines whether the autopilot's trust in us is calibrated. These two tasks carry more consequence per line of code than anything else in the project.

## The chain you must keep honest

```
matching quality metrics ──► T21 covariance ──► eph in GPS_INPUT ──► how much EKF2 trusts the fix
                         └──► T22 accept / reject ──► whether EKF2 sees it at all
```

Understate the covariance and the autopilot believes a bad fix. Overstate it and it ignores a good one. Both errors are expensive, and both are born in T21.

**When КТ-3 fails, the cause is almost always T21, not the matcher.** The temptation will be to tweak a constant in `geoloc_mavlink` until the numbers look right. Do not. That turns an honest covariance into a lie fitted to one scenario, and the lie will surface in a different scenario later.

## T21 — covariance model

Analytical base: `Σ ≈ σ²_pix · (JᵀJ)⁻¹`, scaled inversely with inlier count. Then correction factors from the quality metrics:

| Metric | Effect |
|---|---|
| `covisibility` | Below 20%, covariance grows non-linearly (OrthoLoC: sharp error increase) |
| `spatial_spread` | Inliers bunched in one corner determine `Δψ` poorly → **anisotropic** correction hitting the angular component specifically |
| `peak_ratio` | Near 1.0 the covariance must explode — we do not know it is the right peak |
| `mean_confidence` | Inliers in the DEM-only region are less reliable |
| `channel` | Fallback channels get an inflated base |

**Two mistakes, in order of how often they happen:**

1. **Forgetting the additive systematic term.** A pure `Σ ∝ 1/n_inliers` model gives centimetre-level σ at high inlier counts — impossible on a satellite basemap with 2–5 m georeferencing bias. The basemap bias (measured in T09) enters as an **additive term that does not shrink with inlier count**. However many inliers you have, they cannot reduce the map's own error. `T21-U-05` asserts that as `n_inliers → ∞`, Σ tends to bias², not zero.

2. **Isotropic covariance.** Heading is constrained by inlier *distribution* in a completely different way than position is. An isotropic model simultaneously overstates one and understates the other.

**Calibrate against `NEES`, not against `A@20`.** This is unintuitive and therefore usually skipped. Run the Monte-Carlo set (≥100 seeds) and check that mean NEES lands inside the 95% χ² confidence interval and that `sigma1_coverage` is 60–75%. That is the only way to know whether your covariance is honest.

Keep the model **simple and explainable**. You will be debugging it after every failed checkpoint and every field trial. A covariance produced by a neural network or a tangle of heuristics cannot be debugged.

## T22 — integrity monitor

Target: **`IFR` < 0.5%** — the fraction of *accepted* fixes with >50 m error.

A cascade of gates, applied in order, any failure rejecting the fix with a recorded reason:

`n_inliers` ≥30 · `inlier_ratio` ≥0.25 · **`covisibility` ≥0.20** · **`peak_ratio` ≥1.3** · `residual_rms_px` ≤2.0 · `spatial_spread` ≥0.4 · `mean_confidence` ≥0.5 · **χ² gate vs the prior** (3 DoF, 99% → 11.34)

All thresholds are YAML configuration. They will change after every field trial.

**The worst case, and the one to design against: consistent false fixes.** A matcher that latches stably onto the wrong peak of a periodic structure emits a *series* of mutually consistent false fixes. The N-consecutive-consistency check will not catch them — they are consistent. Only `peak_ratio` and the χ² gate against the prior can. Isaac scenario **S-05** exists exactly for this, and it is blocking.

**A monitor that rejects everything is also a failure.** It has `IFR = 0` and is useless. `T22-A-04` balances this: urban `acceptance_rate` must stay ≥60%.

**Calibrate thresholds on `adversarial`, not `regression`.** On clean data any thresholds look fine.

**The χ² gate is double-edged.** If fusion has already drifted, it starts rejecting correct fixes and locks in the error. Hence the `LOST` mode: wider window, relaxed χ² gate, but *tightened* consistency requirement — three consecutive agreeing fixes before accepting the first.

Publish rejection statistics by reason. They show which gate is actually working and which is dead weight.

## T23 — GTSAM fusion

**The key architectural point:** a fix corrects **`map_enu → odom`**, not the aircraft pose directly. Odometry stays smooth, all global correction lives in one transform, and a correction jump does not become a velocity jump for the control loop.

There is no ready-made GPS factor inside FAST-LIVO2 and there should not be. The reference pattern is LIO-SAM: an external GTSAM graph with unary factors. GTSAM's own "Robot Localization" documentation describes exactly this — three unary factors suffice to tie poses to a global frame.

Specifics that matter:

- **Do not create a graph variable per aircraft pose.** The `map_enu → odom` correction changes slowly; there should be an order of magnitude fewer variables than poses, or the graph becomes unmanageable over 10 km.
- **FAST-LIVO2's published covariance is not the covariance you want.** Calibrate it against MARS-LVIG: measure how error actually grows with distance and build a model. Published odometry covariance is typically optimistic.
- A robust loss (Huber/Cauchy) is the **second** line of defense, not the first. It bounds a single outlier's influence; it will not stop a consistent false series. T22 is the first line.
- Log the magnitude of tf correction jumps. Large jumps are a symptom of a problem further up the chain.
- **Check T09's result before you start.** If basemap bias scatter exceeds 8 m, bias must become an estimated state in the graph — roughly doubling this task. Confirm before committing to the simpler design.

## T24 — modes and recovery

Six modes with **hysteresis on every transition**. Without it the system chatters on the `NOMINAL`/`COASTING` boundary, flapping `fix_type` and upsetting EKF2.

`LOST` is the most dangerous mode: wide window and relaxed χ² gate mean false-fix protection rests entirely on `peak_ratio` and consistency. That is why three consecutive agreeing fixes is non-negotiable there.

**An honest refusal beats an invented pose.** If recovery fails, publish `fix_type=0` and growing covariance. The autopilot and the operator need the truth, not a best guess.

## Your deliverables

An estimate whose stated uncertainty is true, and a gate that refuses when it should. Everything else in the system can be approximately right; these two must be honestly wrong when they are wrong.
