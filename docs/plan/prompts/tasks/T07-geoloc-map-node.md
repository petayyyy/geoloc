# T07 — geoloc_map node (execution prompt)

**Read `../P0-common.md` and the task card `../../tasks/T07-geoloc-map-node.md` first.**
**Source of truth for the contract:** `src/geoloc_msgs/srv/MapWindow.srv` — the compiled
message contract is newer than `03-interfaces.md` §1.4 (it adds `model_version` in the request
and a `validity` image in the response). **The .srv wins where they differ.**
**Role context:** `../P7-infra.md`.

---

## Context

The onboard node that serves the basemap window to the matcher. The architectural idea here is
ADR-005: **the prior window derived from the fusion covariance replaces a retrieval network.**
The matcher asks for a window around its prior position; descriptors for that window are
computed **on board** and cached — never precomputed offline, because offline precomputation
ties the map package to a model version and inflates it.

## Boundaries

| Responsible for | NOT responsible for |
|---|---|
| Serving basemap window, DEM, semantics; descriptor cache; prefetch | Matching. The matcher judges nothing and neither does this node — it serves data |

## What to implement

1. **Geopack loading** via mmap with lazy COG tile reads (cold tile on NVMe costs tens of ms —
   see prefetch below).
2. **`MapWindow.srv` service** per the .srv contract: center, radius, GSD, `with_descriptors`,
   `model_version` → window image + `validity` mask + georeferencing + optional descriptors.
3. **Pyramid level selection** by requested GSD (coarse level for cold start / LOST).
4. **Descriptor cache:**
   - computed for the current window, reused across requests;
   - invalidated when the window centre shifts >30% of the window size, the requested GSD
     changes, **or the matcher model version changes** — the last one is easy to forget and
     produces a silent incompatibility after a model update;
   - model version comes in the request field; never assume it.
5. **Prefetch** along the velocity vector (`prefetch_lookahead_s: 5.0` in the mission config),
   so disk latency never lands inside the match loop.
6. **DEM and semantic layers** served through the same interface and georeferencing.
7. A window request **outside the mission package is a normal situation** (aircraft left the
   planned route), not an exception: clean `success=false` + `message`, no crash.

## Key decisions

| Decision | Value | Why |
|---|---|---|
| Descriptor precompute offline | **No.** Onboard cache only | Amortized ~8 ms/frame does not pay for model-version coupling |
| Window storage | uint8 grayscale + validity mask | Matcher works on intensity |
| Cache invalidation | >30% window shift, GSD change, model version change | Any of the three, checked every request |
| Window radius | `R = k·√(σ_e² + σ_n²) + margin`, `k≈3` — computed by fusion, this node just serves it | 3σ covers 99.7% |

## Performance targets (on the RK3588)

- Service answer ≤20 ms for a 600×600 px window without descriptors, ≤40 ms with.
- Cache hit rate ≥85% at 10 m/s and a 2 Hz fix cycle.
- Prefetch keeps p95 ≤ 2× p50.
- Node RSS ≤200 MB with a 10 × 2 km package.
- Zero allocations in the request hot path (P0 rule 7).

## Acceptance (tests from the card)

- [ ] T07-U-01: served window georeferencing ↔ known UTM point, <1e-6 m
- [ ] T07-U-02: GSD 2 m request → 4× pyramid level, not base
- [ ] T07-U-03: 29% shift → cache hit; 31% → recompute
- [ ] T07-U-04: request beyond package bounds → clean refusal, no crash
- [ ] T07-P-01 / T07-P-02: board latency ≤40 ms with descriptors; cache hit ≥85% on a real
      trajectory
- [ ] All thresholds come from `configs/mission_template.yaml` (the `map:` section exists) —
      none hard-coded (P0 rule 6)

## Pitfalls

- Cache invalidation on **model version change** is the one everyone misses until a silent
  descriptor mismatch appears after a model update.
- Out-of-package requests are normal operation — handle explicitly.
- First access to a cold COG tile on NVMe is tens of milliseconds; without prefetch it lands
  in the loop.
- The node is a skeleton today (`src/geoloc_map/src/geoloc_map_node.cpp`); keep the package
  structure and naming, replace the TODO body.

## Outputs

```
src/geoloc_map/            (implementation lands here; package already exists)
configs/mission_template.yaml  (map: section — extend only with new thresholds)
```

This node is the last piece of the cartographic chain. After T07, the dataset task (T08/T09
for your flight bag + basemap bias measurement) can consume the geopack through this service.
