# P7 — Infrastructure Engineer

**Read `P0-common.md` first.**

**Tasks:** T01 (monorepo), T02 (Docker & cross-build), T03 (CI), T04 (logging & telemetry), T05 (basemap downloader), T06 (DEM/OSM geopack), T07 (`geoloc_map` node).

---

## Your role

You unblock six parallel work streams and you keep them from diverging. Your early tasks are on nobody's critical path individually and on everybody's collectively.

## T01 — do the message package first, completely

`geoloc_msgs` with **every** message from `03-interfaces.md`, on day one, in full. Not a subset, not "we'll add fields later."

Six parallel streams start this week. If the message contract arrives in week two, they will have written six incompatible structures by then, and reconciling those costs more than the whole task. Ship the contract before the implementations.

Same reasoning for the skeleton graph: every node publishing empty messages, one launch file bringing up the whole graph. **The graph should start on day one**, empty. It gives every stream something to plug into and it makes integration a continuous activity rather than an event.

`geoloc_common` carries the geometry everyone depends on — WGS84 ↔ ENU, SE(2) composition, raster ↔ ENU, covariance validation. Property tests, ≥90% coverage. Angle normalization is the source of half the bugs in SE(2) systems; test it with `hypothesis`, not by eye.

## T02 — the cross-build trap

**RKNN conversion runs only on x86.** The toolkit converts ONNX → RKNN on the host; the board gets `rknn-toolkit-lite2` + `librknnrt.so` at runtime. People lose a day trying to convert on the board and reading an unhelpful error. Document it prominently.

Pin every vendor version explicitly in `versions.lock` — ROS, RKNN SDK, RGA, Mali driver, kernel. RK3588 vendor SDKs break compatibility between versions, and `latest` will bite during a field deployment.

The deploy path must support rollback: build → tar → rsync → systemd restart → healthcheck → revert on failure. The same mechanism is used for field deployment, deliberately: what flies is exactly the image that passed CI.

## T03 — CI

Full spec in `testing/06-ci.md`. Three points that matter more than the rest:

- **Cross-build stays in the blocking PR steps.** An ARM build failure found a week later costs far more than 10 minutes per PR.
- **The `IFR` rule is special:** a rise of more than 0.1 pp blocks the merge regardless of relative change. Ordinary 5% tolerances would let `IFR` creep from 0.2% to 0.45% without ever failing a test.
- **Golden runs are never auto-updated.** The urge to enable that appears the first time CI goes red near a deadline, and it removes the entire point of regression testing.

The board runner is the slowest link — give it nothing beyond what it must do, and add cooldown between runs or the runtime benchmarks will drift.

Build the trend dashboard. A single run cannot show slow degradation; `IFR` sliding from 0.2% to 0.45% breaks no individual test but means something is rotting.

## T04 — logging

Build it so any failure — CI, bench or field — can be analyzed **post-hoc, without reproducing the run**. That is far cheaper than chasing a rare bug.

The debug snapshots on rejected fixes (patch, map window, correspondences) will turn out to be the single most useful artifact in the project when diagnosing rare failures. Do not skimp on them — but do rate-limit, or a 30-minute run fills the disk.

Profiler overhead ≤1%, measured. No blocking calls or allocations in the hot path.

## T05–T06 — map preparation

**Multiple providers is a requirement, not a convenience.** It supplies the cross-domain gap for OrthoSim (T10) and Isaac (T29), and cross-checking in the field. Record each provider's capture date.

Three things to get right:

- **Vertical datum.** Copernicus gives orthometric heights over EGM2008; GNSS and lidar work in ellipsoidal heights. The difference is tens of metres at mid-latitudes. Skipping the conversion produces a systematic scale error in the true-ortho. Classic, expensive, easy to avoid.
- **Copernicus GLO-30 is a DSM, not a DTM** — it already contains buildings and vegetation. Relevant for T29, where extruding OSM buildings on top would double-count them.
- **Provider licensing.** Terms differ and constrain commercial use. Check before the project depends on a source; record it in an ADR.

Do not resample 30 m DEM onto a 1 m grid and call it 1 m resolution. Keep honest resolution in the manifest.

## T07 — `geoloc_map`

The idea here is the architectural one: **the prior window from the covariance replaces a retrieval network.** Radius `R = 3·√(σ_e² + σ_n²) + margin`; match only inside it. That removes the main reason the literature introduces VPR and saves an order of magnitude of compute.

Practical points:

- **Map-window descriptors are cached on board, not precomputed offline.** Offline precomputation ties the map package to a model version and inflates it; the amortized ~8 ms/frame saving does not pay for that. Invalidate the cache on a >30% window shift **and** on a matcher model version change — the second one is easy to forget and produces a silent incompatibility after a model update.
- **Prefetch along the velocity vector.** First access to a cold COG tile on NVMe costs tens of milliseconds, and you do not want that inside the loop.
- **A window request outside the mission package is a normal situation** (the aircraft left the planned route), not an exception. Handle it explicitly, return a clean failure, do not crash.

## Your deliverables

A repository where six streams can work without colliding, a build that reproduces, a CI that catches degradation rather than just breakage, and map infrastructure that is honest about its own resolution and provenance.
