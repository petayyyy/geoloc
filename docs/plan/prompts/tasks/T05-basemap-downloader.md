# T05 — Basemap downloader and COG mosaic (execution prompt)

**Read `../P0-common.md` and the task card `../../tasks/T05-basemap-downloader.md` first.**
**Authoritative schema:** `../../03-interfaces.md` §4 (manifest / geopack layout).
**Role context:** `../P7-infra.md`.

---

## Context

The stack matches a true-ortho patch against a pre-loaded satellite basemap. This task builds
that basemap. **Multiple providers are a requirement, not a convenience**: the cross-provider
gap is the natural domain gap for OrthoSim (T10) and Isaac (T29), and it enables field
cross-checking. Each layer records provider, capture date, GSD and licensing — the last one is
an open decision ([ADR-008](../../adr/ADR-008-basemap-providers.md)); for internal development
and testing this is fine, but **never claim commercial-use rights in the manifest without
closing ADR-008**.

## This run: mission-specific scope

| | |
|---|---|
| Corridor | Maykop, Adygea — derived from the flight waypoint CSV |
| Approx. bounds | lon 39.918–39.926 E, lat 44.822–44.835 N (~500 × 1200 m) |
| UTM | Zone 37N, EPSG:32637 |
| Primary GSD | **0.3 m/px** — flight AGL is 80 m, below the 100 m design floor; at 0.5 m/px the patch would be ~320×240 px |
| Secondary GSD | 0.5 m/px |
| Pyramid | Overviews 2×, 4×, 8× (coarse level ~4 m/px for cold start / LOST search) |
| Providers | Esri World Imagery (A), Bing Aerial (B). Optional third (Yandex Satellite) for cross-check — record license terms |

The corridor is tiny (a few MB at 0.3 m/px). Do not over-engineer for huge areas, but keep the
tool general: the same binary must later serve 10 × 2 km corridors.

## What to implement

1. **Tile acquisition** with a provider abstraction. Two download paths:
   - direct tile fetch per provider (XYZ / quadkey protocols);
   - **import from a pre-downloaded tile cache** (SAS.Planet / QGIS XYZ output) — imagery is
     being downloaded in parallel right now; the tool must consume it, not re-fetch.
2. **Mosaic assembly:** tiles → COG (Cloud Optimized GeoTIFF) in UTM 37N, internal tiles +
   overview pyramid.
3. **GSD levels:** 0.3 and 0.5 m/px, plus pyramid up to ~4 m/px.
4. **Cache:** rebuilding the same area never re-downloads tiles.
5. **Manifest** exactly per `03-interfaces.md` §4 schema: provider, capture date, GSD, bounds,
   CRS, and a placeholder `georef_bias` entry (filled by T09 later — do not invent a value).
6. **Validity mask:** missing tiles, mosaic seams between different capture dates, cloud areas
   → zero in the mask. Honest masks only; a hole is a hole.

## Key decisions

| Decision | Value | Why |
|---|---|---|
| Format | COG, UTM zone of mission | mmap-readable on board, metric coordinates |
| Compression | JPEG q85 for RGB, DEFLATE for masks | ≤150 MB per 10 × 2 km @ 0.5 m/px |
| Channels | Store RGB, serve grayscale to the matcher | Matcher works on intensity; RGB needed for semantics/debug |
| Upsampling | **Forbidden.** A 0.5 m/px source resampled to 0.3 m/px looks nice and adds zero information | Fake resolution corrupts downstream GSD assumptions |

## Acceptance (tests from the card)

- [ ] T05-U-01: pixel ↔ UTM round-trip < 1e-9
- [ ] T05-U-02: synthetic tiles assemble without seam shifts
- [ ] T05-U-03: missing tile → zero validity mask in its bounds
- [ ] T05-U-04: manifest round-trip, schema valid
- [ ] T05-I-01: pyramid levels read and match the base downsampling
- [ ] Mosaic for the Maykop corridor from **two providers**, georeferencing verified ≤1 px on
      known points, validity mask honest, cache working

## Pitfalls

- **Provider licensing** — checked and recorded before the project depends on a source.
- **Seams between tiles of different capture dates** are stable false features for the
  matcher; mark them in the mask.
- Do not confuse source tile GSD with resampled mosaic GSD.
- Record **capture dates per provider**: date mismatch between the two providers is a free,
  realistic domain gap — document it, don't hide it.

## Outputs

```
data/missions/<mission>.geopack/ (stub: ortho_a.tif, ortho_b.tif, manifest.yaml)
tools/mapprep/                  (the tool itself)
```

Next in the chain: T06 fills DEM + OSM into the geopack; T07 serves it on board; T09 will
measure `georef_bias` against the flight RTK/GPS track — that is the dataset task launched
after this one.
