# T06 — Copernicus DEM + OSM semantics → .geopack (execution prompt)

**Read `../P0-common.md` and the task card `../../tasks/T06-dem-osm-geopack.md` first.**
**Authoritative schema:** `../../03-interfaces.md` §4 (geopack layout / manifest).
**Terrain class rules:** `../../testing/05-metrics.md` §5.
**Role context:** `../P7-infra.md`.

---

## Context

The geopack from T05 holds the ortho layers. This task adds the two layers that make the
true-ortho patch work at the edges and the semantic fallback possible:

- **Copernicus GLO-30 DEM** — filler where the Livox Avia FOV does not cover the camera frame
  (~30% of patch width). Without it the true-ortho at patch edges is either wrong or absent.
- **OSM semantic raster** — the reference layer for fallback channel 2 (T20) and the terrain
  classifier that all metrics must be broken down by.

## This run: mission-specific scope

| | |
|---|---|
| Corridor | Maykop, Adygea — same bounds as the T05 geopack (≈500 × 1200 m) |
| DEM tile | Copernicus GLO-30, tile N44 E039. AWS Open Data: `s3://copernicus-dem-30m/` (fetch with `--no-sign-request`) |
| DEM processing | Reproject to EPSG:32637, resample to 10 m grid, **EGM2008 → ellipsoid** vertical datum conversion |
| OSM | Geofabrik extract `southern-fed-district` (includes Adygea) or Overpass query on the corridor bbox |
| Semantic grid | uint8, 1 m/px, classes: `0 background, 1 road, 2 building, 3 water, 4 farmland, 5 forest` |

## What to implement

1. **DEM layer:** download GLO-30 tile → reproject → resample 10 m (interpolation, honest
   resolution) → EGM2008 → ellipsoid → write into the geopack.
2. **OSM rasterization:** extract layers → rasterize to the 1 m/px class grid.
3. **Geopack assembly:** directory (not archive — mmap-readable) with `manifest.yaml` per the
   schema, all layers in one CRS, matching bounds.
4. **Package validator:** rejects CRS / bounds / resolution mismatch between layers with a
   clear message.
5. **Terrain classifier** per `05-metrics.md` §5 (urban ≥15% building, farmland ≥60%, etc.).

## Key decisions

| Decision | Value | Why |
|---|---|---|
| Vertical datum | Convert EGM2008 → ellipsoid **in the tool**, record it in the manifest | Copernicus ships orthometric heights; lidar/GNSS are ellipsoidal. The difference at 44.8° N is tens of metres — skipping this is a systematic scale error in true-ortho |
| DEM resolution | Honest 10 m in the manifest | Never resample to 1 m and call it 1 m |
| GLO-30 nature | It is a **DSM** (contains buildings/vegetation), not a DTM | Fine for our fill role; must be documented so T29 does not double-count OSM extrusion |
| OSM | `road` and `water` are the stable classes; `farmland` changes between seasons | Channel T20 must weight classes accordingly |

## Acceptance (tests from the card)

- [ ] T06-U-01: datum conversion, control points ≤0.5 m
- [ ] T06-U-02: synthetic OSM fragment → expected class raster
- [ ] T06-U-03: validator rejects an intentionally inconsistent package with a clear message
- [ ] T06-U-04: terrain classifier ≥85% agreement on a manually labelled test area
- [ ] T06-I-01: full geopack assembly for the corridor, validator green, all layers consistent

## Pitfalls

- **Vertical datum is the classic trap.** EGM2008 vs ellipsoid — tens of metres at
  mid-latitudes. Convert once, document once, test with control points.
- OSM is incomplete and uneven — the semantic channel must treat it as a hint, not truth.
- Keep honest resolution and provenance in the manifest; this manifest is what the pre-flight
  check (T27) later trusts.

## Outputs

```
data/missions/<mission>.geopack/
  ortho_a.tif, ortho_b.tif   (from T05)
  dem.tif                    (this task)
  semantic.tif               (this task)
  manifest.yaml              (updated)
tools/mapprep/               (validator + classifier live here)
```

Next in the chain: T07 (`geoloc_map` node) consumes this geopack on board.
