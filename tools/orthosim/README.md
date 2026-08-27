# orthosim — T10/T11 synthetic ortho-pair generator

Offline generator of "true-ortho query patch ↔ map window" pairs with **exactly
known** ground truth, built on a geopack (or procedural synthetic scenes). It is
independent of any recorded bag, of the T14/T15 rectifier, and of the T19
matcher — the two halves of a pair are rendered directly from the ortho
providers, so the matcher and covariance model can be tested long before the
on-board perception pipeline exists.

```
python -m orthosim run --config configs/orthosim/smoke.yaml
```

## What it produces

For each pair, in `runs/<run_id>/`:

| file | content |
|---|---|
| `config.yaml` | resolved config + seed |
| `results.csv` / `results.parquet` | one row per pair: GT, prior, delta, providers, terrain class, augmentations (parquet when `pyarrow` is present) |
| `summary.json` | aggregates for CI (`by_terrain`, cross-provider guard, optional matcher metrics) |
| `failures/` | side-by-side query | map window PNG gallery (worst-first when a matcher is supplied) |

## The one rule that governs everything

> **The query is rendered from one source; the map comes from a different one.**

`render.ensure_cross_provider` (T10-U-04) raises `CrossProviderError` if the two
providers are equal. The guard is checked per pair, so a config edit that
accidentally aliases the providers fails the whole run instead of silently
producing a zero-gap dataset.

GT is fixed before augmentation and never recomputed after: an augmentation
perturbs the pipeline input, not the truth (P5-data-sim).

## Layout

```
orthosim/
  geopack.py     geopack manifest + georeferenced raster access (Field)
  camera.py      pinhole + radtan + rotation helpers
  dsm.py         height fields: flat / Copernicus DEM / OSM-building extrusion
  scene.py       ties the two ortho providers + terrain + semantic
  render.py      true-ortho / map-window / perspective / rectify + cross-provider guard
  augment.py     radiometric + sensor + season augmentations (identity at 0)
  cloud.py       Livox Avia point-cloud synthesis (v2): FOV/density/motion/water
  synthetic.py   procedural scenes for the `adversarial` set
  pairs.py       PairSpec / RenderedPair / generate_pair
  sets.py        named sets (smoke / regression / adversarial / sweep_*)
  run.py         orchestration
  io.py          run outputs
  cli.py         `orthosim run`
```

## Terrain modes (v1 vs v2)

| mode | what | task |
|---|---|---|
| `flat` | constant ground plane | v1 placeholder |
| `dem` | Copernicus GLO-30 from the geopack | v1 |
| `buildings` | DEM + OSM building extrusion (real parallax) | v2 |

## Avia cloud synthesis (v2, `cloud.py`)

`AviaSpec` fixes FOV 70.4° × 77.2° (~141 × 160 m footprint at 100 m AGL),
~240k points/s, range noise σ≈2 cm, motion distortion over the accumulation
interval, and water dropout via the semantic layer. The non-repetitive rosette
is approximated by a uniform draw over the FOV rectangle — the pattern is not
what the tests measure. Output is `(N,4)` `(x, y, z, t_rel)`; ROS `CustomMsg`
serialisation is a thin adapter left to the runtime (out of scope here, ROS is
not a dependency of this package).

## Tests

```bash
python -m pytest tools/orthosim/tests -q
```

- `test_render.py` — T10-U-01 (GT correctness), T10-U-02 (warp round-trip),
  T10-U-03 (determinism), T10-U-04 (cross-provider guard), T10-U-05
  (augmentations identity/effect).
- `test_cloud.py` — T11-U-01 (FOV footprint), T11-U-02 (density), T11-U-03
  (motion distortion), T11-U-04 (water dropout).
- `test_geopack.py` — multiband bilinear sampling + nodata handling.

The `adversarial` set (`configs/orthosim/adversarial.yaml`) generates the six
trap scenes (periodic grid, water, snow, forest, stale map, symmetric
interchange) — 1200 pairs. It exists to measure how often the matcher is
*confidently wrong*, not for the system to pass.
