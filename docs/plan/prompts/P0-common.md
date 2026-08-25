# P0 — Common rules for all agents

**Read this before every task. No exceptions.**

---

## Project in one paragraph

You are building `geoloc`: GNSS-denied global localization for a UAV. FAST-LIVO2 (Livox Avia + one nadir global-shutter camera + IMU) provides smooth but drifting local odometry. Our system periodically pins that odometry to a global frame by matching a true-ortho patch — rectified through the lidar-derived DSM — against a pre-loaded satellite orthophoto basemap, and feeds the globally corrected pose to PX4. Onboard computer: OrangePi 5 Pro (RK3588S). Flight profile: 100–150 m AGL, 5–10 km route.

## The one thing that matters most

**Position drift is not the problem. Heading is.**

FAST-LIVO2 gives 0.75–3.25 m ATE RMSE over routes up to 7.1 km. An uncorrected ±3° initial heading error over a straight 10 km gives **~523 m of lateral drift**. Everything in this architecture follows from that: fixes exist primarily to make heading observable, which is why "≥5 valid fixes in the first 2 km" is a hard requirement derived mathematically, not a nice-to-have.

**And therefore: `IFR` (Integrity Failure Rate — the fraction of *accepted* fixes with >50 m error) is the single most important metric in this project, target < 0.5%.** A missed fix costs growing covariance. An accepted false fix drags EKF2 off course and can cost the aircraft. The asymmetry is total. Resolve every trade-off toward refusing.

## Documents you must treat as authoritative

| Document | What it governs |
|---|---|
| `03-interfaces.md` | **Source of truth.** Message contracts, PX4 fields, map package format. Changing anything here requires updating dependent tasks — flag it, do not do it silently |
| `01-requirements.md` | Accuracy, compute, memory budgets. Your task has a budget; know it |
| `05-metrics.md` (testing/) | Metric definitions and thresholds. One metric, one definition, one implementation |
| Your task card in `tasks/` | Goal, DoD, tests, pitfalls. The pitfalls section is not filler — it lists mistakes that have specifically been anticipated |

Read your task card fully before writing code. Read `02-architecture.md` §3 to understand where your node's responsibility ends.

## Rules

### 1. Stay inside your boundary

Each node has an explicit "responsible for / NOT responsible for" entry in `02-architecture.md` §3. The most important boundary in the system:

> `geoloc_matcher` **always** produces a result with quality metrics and **never** decides whether it is valid. That decision lives only in `geoloc_integrity`.

Do not blur this. Isolating the integrity logic is what makes it independently testable and replaceable — and it is the system's main safety interlock.

### 2. Every task ships with its tests

Your task card lists test IDs with levels (0 / A / B / E) and pass criteria. Implement them. A task is not done when the code works; it is done when the tests in the card pass and the DoD checkboxes are all true.

- **Level 0** — unit and property tests, x86 CI, every commit
- **Level A** — OrthoSim, x86 CI, every commit
- **Level B** — replay of real datasets through the real ROS 2 graph, nightly, x86 and RK3588
- **Level E** — Isaac Sim + PX4 SITL, nightly on the GPU box

### 3. Determinism is not optional

Fixed seed → identical result. A non-deterministic test is a broken test, and it gets fixed like any other bug — not tolerated, not retried until green.

### 4. Optimization must not change results

After any optimization, level A and B metrics must match the pre-optimization baseline within tolerance. A discrepancy is a bug, not an acceptable cost of speed. The one exception is a deliberate precision change (FP16, INT8), which is made as a separate, explicitly measured change.

### 5. Never fabricate data

If the lidar returned nothing, that means *unknown*, not *ground at zero height*. If the DSM has no coverage, mark low confidence rather than filling in a plausible value. If localization is lost, publish `fix_type=0` and growing covariance rather than a best guess. Downstream consumers — and ultimately the operator — need the truth.

### 6. Thresholds are configuration, not constants

Every gate threshold, window radius and tuning coefficient goes in YAML. They will change after every field trial. A threshold buried in code is a threshold nobody will find at 6 a.m. on the flight line.

### 7. No allocations in the hot path

The per-frame processing loop must not allocate. Use buffer pools. This is a hard rule on RK3588, where the budget is tight and jitter is expensive.

### 8. Report negative results honestly

Several tasks (T18 AdHoP, T34 fine-tuning) may produce no measurable benefit. That is a valid outcome. Write it up in an ADR with the numbers and move on. Do not tune a benchmark until a feature looks worthwhile.

## Code conventions

| | |
|---|---|
| Language | C++17 for anything in the real-time loop; Python for tooling. The RK3588 budget does not tolerate Python in the loop |
| Math | Eigen 3.4 everywhere. Do not roll your own matrix types |
| Angles | Radians internally, degrees only in diagnostics and logs. Normalize to `[-π, π]` **always** |
| Frames | ENU. Yaw counter-clockwise from East. Follow REP-103/105 |
| Covariance | Row-major, ordered `(east, north, yaw)`. Symmetric and positive-definite — assert in debug builds |
| Time | ROS Time from the **sensor** timestamp, never the receive time |
| Errors | A node never fails silently. Any anomaly → `/geoloc/status.last_rejection_reason` + a WARN log |
| Style | `clang-format` (Google, 100 cols), `ruff` + `black` for Python |

## Definition of Done — the universal part

Beyond what your card lists:

- [ ] All tests from the card implemented and passing
- [ ] `pre-commit run --all-files` clean
- [ ] Cross-build for aarch64 succeeds
- [ ] No new allocations in the hot path (if you touched it)
- [ ] Parameters exposed in YAML, documented
- [ ] If you changed anything in `03-interfaces.md`, you flagged it and listed affected tasks

## When you are blocked or the plan is wrong

Say so. Specifically:

- If your task's budget (latency, memory) turns out to be unachievable — report the measured number and what it would take, do not silently miss it;
- if a document contradicts reality — the document is wrong, flag it;
- if your task has an **escalation rule** (T16 has one at day 10), follow it. Escalation rules exist because the task may not converge, and that must not stall the project.

## What good work looks like here

A correct implementation that reports honest uncertainty beats a clever one that occasionally lies with confidence. The whole system is a chain of estimates feeding an autopilot; every link that overstates its own confidence corrupts everything downstream. When in doubt, widen the covariance and refuse the fix.
