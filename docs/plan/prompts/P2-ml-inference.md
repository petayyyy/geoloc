# P2 — ML / Inference Engineer

**Read `P0-common.md` first.**

**Tasks:** T16 (XFeat → RKNN INT8), T34 (training pairs & fine-tuning). Support role on T20 (semantic segmenter).

---

## Your role

You own the learned components: getting XFeat onto the RK3588 NPU with acceptable accuracy, and later improving it for our domain. **T16 is the highest-risk task in the project and it starts on day one.**

## T16: why it is hard

RKNN Toolkit2 does not support several operations XFeat is built on:

| Problem | What breaks | Workaround |
|---|---|---|
| `grid_sample` | Bilinear descriptor sampling. **Unsupported** | Replace with a mathematically equivalent composition of supported layers |
| Top-k keypoint selection | Unsupported or falls back to CPU | Move to CPU explicitly, or use a fixed grid plus threshold |
| Dynamic shapes | Unsupported | Fixed input size, batch=1 |
| Multiple outputs | Limited support | Split into subgraphs or concatenate |
| Fusion layer | Problematic under quantization | Replace or move out |

The Hailo community reports exactly the same set of blockers for XFeat: stock XFeat does not quantize directly, you need a fork with equivalent-layer substitutions and **your own calibration set** — no public calibration data exists. Treat that as a direct effort indicator for RKNN too.

## The escalation rule — this is binding

> **If by day 10 there is no working RKNN inference of XFeat at acceptable accuracy, switch immediately to plan B (CPU-ONNX XFeat on the A76 cores) and let T17 proceed on it.** The NPU port then becomes an optimization task (T32), not a critical-path blocker.

Not "three more days, it's almost working." Build plan B in parallel from day one, as a working alternative — not as a fallback you will get to later.

## Things that will cost you a day each if you don't know them

- **RKNN conversion only runs on x86.** Trying to convert on the board yields an unhelpful error. The runtime (`rknn-toolkit-lite2` + `librknnrt.so`) runs on the board; the toolkit does not.
- **The calibration set decides everything.** INT8 quantization calibrated on MegaDepth or COCO will produce poor results for our domain, and it will *look* like "the model doesn't work" rather than "the calibration is wrong." Build the calibration set from OrthoSim (T10) — true-ortho patches and map windows with the in-flight distribution — starting day one.
- **Do not evaluate degradation by activation MSE.** It tells you nothing about matching. Evaluate with matching metrics on `regression` and `adversarial`. This is why T16 depends on T10/T12 for its acceptance criteria.
- **Every layer substitution needs a numerical-equivalence test in FP32** before you quantize. Otherwise you will be debugging two problems at once — a wrong substitution and a bad quantization — and they look identical from the outside.

Start with **XFeat\*** (semi-dense; TE 0.66 m vs 0.96 m for plain XFeat on OrthoLoC). Fall back to plain XFeat if the budget does not allow it.

## T16 acceptance

| | |
|---|---|
| Latency | ≤35 ms per 640×480-equivalent patch, one NPU core |
| **INT8 degradation** | **`A@20` drops ≤5 pp vs FP32** on `regression` |
| **`IFR`** | Does not rise more than 0.1 pp on `adversarial` |
| Coexistence | With FAST-LIVO2 running, its rate stays ≥10 Hz |
| Runtime wrapper | No allocations in the hot path, thread-safe, no RSS growth over 10 000 calls |

## T34: fine-tuning, with honest expectations

There are no published numbers for fine-tuning XFeat/SuperPoint on region and season. The estimate is extrapolated from related cross-view domain-adaptation work (+11–16% R@1). **This is extrapolation, not fact.** So T34 is framed as an experiment with a pre-committed decision rule:

| Result | Decision |
|---|---|
| `A@20` +5 pp or more, `IFR` no worse | Adopt |
| +2 to +5 pp | Adopt if `IFR` and latency are unchanged |
| < +2 pp | **Reject.** Record the negative result in an ADR |
| `IFR` worse | **Reject unconditionally**, whatever the accuracy gain |

Two traps:

- **Geographic leakage.** If training and validation patches come from the same area, the network memorizes specific buildings and roads. The metric looks excellent and generalizes nowhere. `T34-U-02` automatically asserts that validation regions do not overlap training regions — keep it.
- **Domain overfitting raises `IFR`.** A network that confidently finds correspondences in the target domain will also confidently find them where none exist. `T34-A-02` is blocking for exactly this reason.

Training pairs need **no manual labeling** — the geometry is known (RTK + lidar DSM, or OrthoSim's exact poses), so correspondences follow from it. Same recipe XFeat itself is trained with.

After fine-tuning, **redo the INT8 calibration**: the new model has a different activation distribution, and stale calibration will dissolve your gains.

## Your deliverables

A working model on the NPU with measured degradation, a documented calibration set, a runtime wrapper that behaves in the real-time loop — and, for T34, a decision backed by numbers on all four evaluation sets, including the option "reject, here is why."
