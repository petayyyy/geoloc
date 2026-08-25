# P6 — Autopilot Integration Engineer

**Read `P0-common.md` first.**

**Tasks:** T25 (`GPS_INPUT`), T26 (EKF2 tuning), T27 (failsafe), plus coordination of T36 (field trials).

---

## Your role

You own the boundary between our estimate and the aircraft. This is one of two places in the project where a mistake costs the airframe — hence double time margin on T26–T27 and mandatory passage of every T30 scenario before anything flies.

## The governing principle

> **The localization system makes no flight decisions.** It honestly reports its state and uncertainty. Decisions about mode changes, return or landing belong to the autopilot and the operator.

An automatic mode change issued from our node is prohibited. `T27-U-02` audits for it. The idea "if we lose localization we'll switch to Loiter ourselves" sounds reasonable and is a design error: it creates a second decision-making center that neither the autopilot nor the operator knows about.

Our responsibility ends at an honest `fix_type`, an honest `eph`, and a clear message.

## T25 — why `GPS_INPUT` and not `VISION_POSITION_ESTIMATE`

Known problem (issue hku-mars/FAST-LIVO2 #126): feeding pose via `/mavros/vision_pose/pose` in Position mode makes the aircraft drift significantly — EKF2 diverges. Feeding a globally corrected pose disguised as GNSS puts EKF2 in the mode it was designed and tuned for. Recorded in `ADR-003`.

Field-by-field contract is in `03-interfaces.md` §3. The parts that bite:

- **`time_usec` is the frame time, not the publish time.** Loop latency is compensated here or it becomes a speed-proportional systematic error.
- **`eph` comes straight from the covariance. Never smooth it.** A sharp `eph` rise entering COASTING is the truth EKF2 must learn immediately; smoothing converts an honest covariance into a lie.
- **Ignore the velocity fields.** FAST-LIVO2 already supplies velocity through its own channel. Publishing it twice gives EKF2 two correlated measurements of one quantity and it will understate their joint uncertainty.
- **`satellites_visible` looks cosmetic but is not.** PX4 uses it in GNSS quality checks; a value below threshold makes it discard every fix we send.
- `yaw` is populated only when σ_ψ < 2°; otherwise mark it unavailable.

## T26 — EKF2 tuning

Starting parameter set is in `03-interfaces.md` §3. Four things to get right:

1. **`EKF2_GPS_DELAY_MS`.** The default is set for a real GNSS receiver. Ours is different. Measure it — correlate EKF2 innovations against a known disturbance — and set it. A 200 ms error at 15 m/s is 3 m of systematic offset, growing with speed. Do not leave the default.

2. **Consistency check.** The fraction of time the true error falls inside `eph` should be ~68%. **If it does not, the fault is in T21 and it gets fixed there.** The temptation to adjust coefficients here is strong and produces a covariance that is a lie fitted to one scenario.

3. **Innovation analysis.** A systematic bias in normalized GPS innovations almost always means one of two things: a wrong coordinate transform, or unaccounted latency. Check those first, always.

4. **`EKF2_GPS_CHECK` relaxation is double-edged.** Those checks exist to discard bad GNSS. Relaxing them removes the autopilot's last line of defense and transfers full responsibility to our integrity monitor (T22). That is a deliberate choice and it must be recorded in the ADR as such — not slipped in as a tuning detail.

SITL tuning does not transfer to the field one-to-one. Plan a re-check inside T36.

## T27 — failsafe

Five degradation levels (L0 normal → L4 fault), each with a defined `fix_type`, message severity and reaction. Full table in the task card. Beyond it:

- **Watchdog per node**, detection within 2 s.
- **Ceasing `GPS_INPUT` publication is itself a decision** whose consequences depend on PX4 configuration. Verify that PX4 treats our silence as an ordinary GNSS loss rather than hanging waiting.
- **Message rate limiting.** Level chatter will flood the telemetry link and bury what matters. Hysteresis plus one message per level change.
- **Pre-flight check** — refuse to enter `NOMINAL` if the map package does not cover the mission area, the calibration is stale, or model/firmware versions disagree. This reads as boilerplate until the day someone launches with the neighboring district's map.

## T36 — field trials

You coordinate. The escalation is deliberate and the gate conditions are not advisory:

**Do not start** until every Isaac blocking scenario passes (including S-04 and S-05), КТ-4 is met, T35 is closed with calibration holding after a motor run, and the pre-flight check works.

Stages: ground → tethered → **first flight with GNSS active** (our system logging only) → flights with GNSS emulated-lost (GNSS still physically running as reference) → full 10 km route → analysis.

Three points:

- **Stage Э3 (GNSS active) looks like a formality and is the most valuable stage in the program** — the only one where being wrong costs nothing. It gives the first real comparison against ground truth.
- **`IFR` in the field will almost certainly be worse than in simulation.** Reality has more traps than any `adversarial` set. Budget time to return to T22 after Э3; that is expected, not a failure.
- **Safety rules are absolute:** GNSS always physically running and logged as reference; operator able to restore it instantly via a switch tested on the ground; no advancing past open findings; first flight of each stage in the most favorable conditions; any unexpected behavior stops the day for analysis, not "let's try once more."

Log the weather and lighting of every flight, or results from different days are not comparable.

## Your deliverables

An interface that tells PX4 the truth about our uncertainty, a tuning set with a written justification for every deviation from default, a failsafe that escalates without deciding, and a trial program that never advances on optimism.
