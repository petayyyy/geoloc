# geoloc_metrics — T12 metrics harness

The single implementation of every metric in `docs/plan/testing/05-metrics.md`.
One metric, one implementation, used by every test level, so numbers from level
A (OrthoSim) and level B (replay) are comparable by construction.

## What's here

```
geoloc_metrics/
  schema.py     the common fix + trajectory record contract (CSV of numpy dtypes)
  metrics.py    A@d, IFR, RE, acceptance_rate, ATE_RMSE, lateral, NEES/NIS, latency
  terrain.py    terrain-class breakdown (same class table as mapprep)
  bias.py       dual bias accounting (with / without the basemap bias)
  summary.py    summary.json (05-metrics.md section 6)
  golden.py     golden-run comparison with degradation tolerances (section 7)
  report.py     self-contained report.html (inline SVG, base64 images)
  trends.py     append-to-parquet/csv + static dashboard
  cli.py        `geoloc-metrics report | compare | trends`
t19/
  match_main.cpp  the T19 phase-correlation matcher as a standalone C++ CLI
  run_t19.py      level-B runner: orthoproto patches -> t19_match -> records
tests/          T12-U-01..06, T12-I-01
```

## Rules the harness enforces

- **`A@d` and `IFR` are over accepted fixes, never all attempts.** Availability
  is `acceptance_rate`, separately.
- **Terrain breakdown is everywhere.** An average across classes mixes urban
  (works) with forest (doesn't) and hides both.
- **`NEES` is the only honesty check for the covariance** (T21 calibration
  signal); an understated covariance inflates it.
- **The matcher never judges.** The harness consumes an `accepted` column that
  the runner stamps; it only reports metrics conditioned on it.

## Usage

```bash
# test the harness (no ROS, no rasterio)
pytest tests

# level-B T19 evaluation (needs rosbags, rasterio, pyproj; run in the container)
python t19/run_t19.py --config ../../configs/metrics/t19_eval_capture01.yaml

# report / golden gate / trends
geoloc-metrics report <records> --out-dir runs/latest [--golden golden/summary.json]
geoloc-metrics compare runs/latest/summary.json golden/summary.json
geoloc-metrics trends runs/latest/summary.json --store-dir runs/trends
```

## Building `t19_match`

The C++ CLI reuses `geoloc_matcher`'s header-only matcher (the same code the ROS
node runs), so level-B numbers are not a Python re-implementation:

```bash
g++ -std=c++17 -O2 -I src/geoloc_matcher/include -I src/geoloc_common/include \
    -I /usr/include/eigen3 tools/metrics/t19/match_main.cpp -o tools/metrics/t19/t19_match
```
