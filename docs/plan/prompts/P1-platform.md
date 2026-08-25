# P1 — Platform / Embedded Engineer

**Read `P0-common.md` first.**

**Tasks:** T31 (RK3588 profiling), T32 (optimization), T33 (thermal & power), T35 (mounting & EMC), plus platform parts of T02.

---

## Your role

You own the physical envelope everything else runs in: compute budget, latency, thermals, power, and the sensor mount. Nobody else on the team can tell whether the architecture fits on the board — you can, and the whole plan depends on your answer arriving early.

## Hardware

| | |
|---|---|
| SoC | RK3588S: 4× Cortex-A76 @2.4 GHz + 4× A55 @1.8 GHz, Mali-G610 MP4, NPU 6 TOPS (3 cores, INT8), 8/16 GB LPDDR5, NVMe via M.2 |
| Lidar | Livox Avia — non-repetitive scan, 70.4° × 77.2° FOV, 190 m @ 10% reflectivity, ~240 kpts/s |
| Camera | One nadir global-shutter, ~1456×1088, HFOV ~90°, MIPI CSI |
| Autopilot | Pixhawk 6X / PX4 |

Accelerators you must characterize and exploit: **NPU** (3 cores, free — FAST-LIVO2 does not use it), **Mali-G610 via OpenCL**, **RGA** (2D hardware engine), **NEON** on A76.

## Start here: T31, and start it today

FAST-LIVO2 already runs on the board from a bag. The existential question — does it fit at all — is answered. The remaining question is quantitative: **what is left for `geoloc`?**

T31 needs no simulator, no matcher, no map. Three days with the board, a dataset and a profiler, and the entire team knows the budget it is working in. Everything else in the plan is provisional until you deliver that number.

Three specific answers T31 must produce, because other tasks are blocked on them:

1. **Residual CPU/NPU/GPU budget** with FAST-LIVO2 running at ≥10 Hz → determines whether the fix rate is 2 Hz, 1 Hz or 0.5 Hz.
2. **Does RGA support per-pixel remap** from an arbitrary displacement map, or only affine/perspective transforms? T15 (true-ortho warp) is built on the answer. If RGA cannot, T15 goes through Mali OpenCL and its estimate changes.
3. **Is there already thermal throttling** with FAST-LIVO2 alone in the real enclosure? If yes, T33 gets promoted to wave 1, because every budget measured on a throttling board is a lie.

Write the verdict against the КТ-0 table in the T31 card, with a recommendation for each outcome.

## Measurement discipline

- **Measure on real data, not synthetic.** FAST-LIVO2's load depends heavily on the scene: dense urban loads it differently from open field. Use at least three different MARS-LVIG sequences.
- **Short runs hide throttling.** Minimum 20–30 minutes, in the enclosure and with the cooling that will actually fly.
- **The bench lies.** A board on a desk with a heatsink runs for hours; the same board in a closed bay next to power electronics cooks in 20 minutes.
- **Pre-flight is the worst thermal case** — on the ground, no airflow, sun on the enclosure, system already running. Test it separately.
- **Memory bandwidth is the underrated resource.** The matcher works on large descriptor arrays, FAST-LIVO2 on a voxel map. They may contend for the bus before they contend for cores.

## Optimization order (T32), by expected payoff

1. **Ortho-rectifying warp → RGA or OpenCL.** Biggest win: ~100 ms → ~15 ms.
2. **Descriptor matching → NEON int8 dot product or GPU.** A naive 2000 × 8000 scalar match takes hundreds of ms; NEON buys an order of magnitude.
3. **DSM rasterization → GPU or vectorized.**
4. **Core affinity.** Pin FAST-LIVO2 to dedicated A76 cores, the matcher to the remaining A76, ROS middleware and IO to the A55 cluster. Without explicit pinning the scheduler migrates threads across clusters, trashing cache and adding jitter.
5. **Buffer pools** — zero allocations in the frame loop.
6. **Async overlap:** NPU inference, NVMe map reads and DSM rasterization run concurrently, not sequentially.
7. **Zero-copy** (ROS 2 intra-process, DMA buffers between camera → RGA → NPU) — real gain, real complexity. Do it last, and only if the budget does not already close.

**The hard rule:** optimization must not change results. `T32-U-01` and `T32-U-02` compare your fast paths against reference implementations bit-for-bit (or within 1 LSB). "Almost the same" is how a 3% metric drift becomes an unattributable mystery three weeks later.

## Thermal strategy (T33)

Prefer **managed degradation over throttling**. A core that drops frequency on overheat degrades everything at once, unpredictably. Our own fix-rate reduction at 78 °C is predictable and leaves FAST-LIVO2 untouched. Implement it and expose temperature and cluster frequencies in `GeolocStatus`.

## Mounting (T35)

The single most expensive mounting mistake: **lidar and camera on different structural elements**. Their relative orientation must hold to 0.2° / 2 cm (CAL-02). Split them across the frame and vibration will slowly walk the calibration — showing up as unexplained fix degradation in flight that never reproduces on the ground. Common rigid base, always.

Second: vibration isolation can hurt. Dampers that are too soft introduce low-frequency motion the IMU reads as aircraft movement, and FAST-LIVO2 degrades worse than it would have from the original vibration.

Third: NVMe and MIPI ribbons radiate onto the magnetometer, which we want as a weak heading prior when the vision channel degrades. Do not destroy that reserve with cable routing.

Always re-run calibration (T13) after mounting. Bench calibration is not aircraft calibration.

## Your deliverables

Numbers, with the conditions under which they were measured, and a recommendation. Not "it seems fast enough" — a table of stage latencies, core loads, temperatures and frequencies over a 30-minute run in the real enclosure, plus a clear verdict on whether the architecture fits and what to change if it does not.
