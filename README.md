# geoloc

GNSS-denied global localization for a UAV: FAST-LIVO2 provides smooth but drifting local odometry; this stack periodically pins it to a global frame by matching a true-ortho patch — rectified through the lidar DSM — against a satellite orthophoto basemap, and feeds the corrected pose to PX4.

**Sensors:** Livox Avia + one nadir global-shutter camera + IMU. **Compute:** OrangePi 5 Pro (RK3588S). **Autopilot:** Pixhawk 6X / PX4. **Profile:** 100–150 m AGL, 5–10 km route.

Full plan, task cards, test strategy, agent prompts and ADRs live in [`docs/plan/`](docs/plan/README.md).

---

## Status: day-one scaffold (task T01)

What works right now:

| | |
|---|---|
| `geoloc_msgs` | **Complete** message contract — every message and service from `03-interfaces.md` |
| `geoloc_common` | **Complete and tested** — geodetic, SE(2), raster, covariance. 160 051 checks, 0 failures |
| `tools/mapprep` | **T05 done** — basemap downloader + COG mosaic builder; Maykop geopack from Esri + Bing, 34 level-0 tests green |
| Six pipeline nodes | Skeletons that launch. Implementation lands with their tasks |
| `configs/mission_template.yaml` | Every threshold from the plan, as configuration |

The message contract is complete on purpose. Six work streams start in parallel; if the contract landed in week two they would have written six incompatible structures by then.

## Quick start

```bash
make test-common     # property tests, no ROS needed — start here
make build-x86       # colcon build
make test            # full suite
ros2 launch geoloc_bringup all.launch.py
ros2 topic list      # every topic from docs/plan/03-interfaces.md
```

`make test-common` builds the geometry tests standalone with just g++ and Eigen, so it runs in a bare container before any ROS environment exists.

## Layout

```
src/
  geoloc_msgs/       message + service contracts        <- source of truth, complete
  geoloc_common/     geometry & estimation primitives   <- tested
  geoloc_map/        T07   basemap window, descriptor cache
  geoloc_ortho/      T14, T15   deskew -> DSM -> true-ortho patch
  geoloc_matcher/    T16-T20    features, SE(2), fallback channels
  geoloc_integrity/  T21, T22   accept/reject + covariance
  geoloc_fusion/     T23, T24   GTSAM graph, modes, tf map_enu->odom
  geoloc_mavlink/    T25-T27    GPS_INPUT, failsafe
  geoloc_bringup/    launch files
tools/               generators, deploy, bench
configs/             mission YAML
docs/plan/           the plan
```

## Four rules that shape this codebase

**1. The matcher never judges.** `geoloc_matcher` always publishes a result with quality metrics and never decides whether it is valid. That decision lives only in `geoloc_integrity`. Isolating it is what makes the integrity logic independently testable — and it is the system's main safety interlock.

**2. `IFR` outranks accuracy.** Integrity Failure Rate — accepted fixes with >50 m error — targets **< 0.5%**. A missed fix costs growing covariance; an accepted false fix drags EKF2 off course and can cost the aircraft. The asymmetry is total, so every trade-off resolves toward refusing. See [ADR-006](docs/plan/adr/ADR-006-integrity-over-accuracy.md).

**3. Never fabricate data.** No lidar return means *unknown*, not *ground at zero*. No DSM coverage means low confidence, not a plausible fill. Lost localization means `fix_type=0` and growing covariance, not a best guess.

**4. Thresholds are configuration.** They live in `configs/mission_template.yaml` and will change after every field trial. A threshold buried in code is one nobody finds on the flight line.

## Geometry conventions

| | |
|---|---|
| Angles | Radians internally, degrees only in diagnostics. Normalised to `[-π, π]` **always** |
| Yaw | ENU, counter-clockwise from East |
| Raster | Pixel (0,0) is the **top-left corner**; centres at `(col+0.5, row+0.5)`; row increases southward |
| Covariance | Row-major over `(east, north, yaw)`; symmetric and positive-definite, asserted in debug |
| Frames | `map_enu → odom → base_link`. A fix corrects **`map_enu → odom`**, never the aircraft pose directly |
| Time | Sensor timestamp, never receive time |

The ENU anchor is a mission parameter, never a code constant.

## Before starting work

Read [`docs/plan/prompts/P0-common.md`](docs/plan/prompts/P0-common.md), then your role prompt, then your task card. Two ADRs are still open and block wave 0: [ADR-007](docs/plan/adr/ADR-007-ros-distro-os.md) (ROS distro / OS — blocks T01 finalisation) and [ADR-008](docs/plan/adr/ADR-008-basemap-providers.md) (basemap providers and licensing). T05 shipped under the internal-development caveat from its execution prompt: provider terms are recorded in every manifest, and no commercial-use rights are claimed until ADR-008 is closed.
