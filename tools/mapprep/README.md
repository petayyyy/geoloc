# mapprep — basemap, DEM and OSM geopack tooling (tasks T05/T06)

Builds the `.geopack` for a mission corridor: ortho tiles from multiple
providers are fetched (or imported from an existing cache) and assembled into
UTM Cloud-Optimized GeoTIFFs with an overview pyramid and an honest validity
mask (T05); the Copernicus GLO-30 DEM and an OSM semantic class raster are
added and the whole package is validated for cross-layer consistency (T06). A
manifest per `docs/plan/03-interfaces.md` section 4 records provider, GSD,
vertical datum and licensing.

## Install

```bash
pip install -e tools/mapprep          # deps: numpy, rasterio, pyproj, PyYAML, jsonschema
pip install -e "tools/mapprep[test]"  # + pytest (for the level-0 suite)
```

## Usage

```bash
# build a corridor geopack (Maykop example)
python -m mapprep build --config configs/mapprep/maykop_corridor.yaml \
    --out data/missions/maykop-corridor-2026-08.geopack

# rebuild without network: every tile comes from the on-disk cache
python -m mapprep build --config configs/mapprep/maykop_corridor.yaml \
    --out data/missions/maykop-corridor-2026-08.geopack --offline

# import a pre-downloaded cache instead of fetching (SAS.Planet / QGIS XYZ)
python -m mapprep import-cache --src /path/to/sas_cache \
    --provider esri_world_imagery --cache ~/.cache/geoloc/tiles --layout sasplanet
python -m mapprep import-cache --src /path/to/qgis_xyz \
    --provider bing_aerial --cache ~/.cache/geoloc/tiles --layout qgis_xyz

# verify a built geopack (geotransform, pyramid, cross-provider shift)
python -m mapprep verify --geopack data/missions/maykop-corridor-2026-08.geopack --cross
python -m mapprep inspect --geopack data/missions/maykop-corridor-2026-08.geopack

# cross-layer consistency check (CRS / bounds / resolution)
python -m mapprep validate --geopack data/missions/maykop-corridor-2026-08.geopack

# terrain class of the semantic layer (05-metrics.md section 5)
python -m mapprep classify --geopack data/missions/maykop-corridor-2026-08.geopack
```

## What the geopack contains

```
<mission>.geopack/
  manifest.yaml       # version, CRS, bounds, origin, per-layer metadata
  ortho_a.tif         # COG (LAYOUT=COG), RGB, JPEG q85, UTM zone of the mission
  ortho_b.tif         # second provider (OrthoSim domain gap, field cross-check)
  validity_a.tif      # COG, uint8 DEFLATE: 255 = valid, 0 = hole/seam
  validity_b.tif
  dem.tif             # GLO-30 DSM, float32 COG, ellipsoidal heights (T06)
  semantic.tif        # OSM class raster, uint8 COG, 1 m/px (T06)
  gcp.csv             # tile-corner GCPs with measured round-trip errors
```

## Key rules encoded here

- **Upsampling is forbidden for ortho.** A source whose native GSD is coarser
  than the target builds at its native GSD; the manifest records the honest
  value. The DEM is interpolated onto its requested grid by plan mandate (T06),
  and the manifest records both the grid spacing (`gsd`) and the source
  resolution (`native_gsd`).
- **Honest masks.** Missing tiles (404), seams between tiles of different
  capture dates, and imported cloud masks all zero the validity mask. Areas
  outside fetched tiles are invalid, never filled.
- **Deterministic.** Sorted tile order, single-threaded warping, fixed
  resampling: the same cache produces byte-identical geopacks.
- **Capture dates are never guessed.** Esri's public identify endpoint no
  longer exposes per-tile dates; the manifest records `null` plus the Wayback
  release at build time when known.
- **Vertical datum is converted, not ignored.** GLO-30 ships EGM2008
  orthometric heights; GNSS/lidar are ellipsoidal. The build adds the geoid
  undulation (EGM2008 grid from the PROJ CDN) and records the conversion in the
  manifest. A missing geoid grid fails the build rather than guessing.
- **Licensing (ADR-008 open).** Provider terms are recorded per layer; the
  manifest claims no commercial-use rights.

## Tests

```bash
python -m pytest tools/mapprep/tests
```

Level-0 suite from the task cards: T05-U-01 (pixel <-> UTM round-trip),
T05-U-02 (synthetic tiles assemble without seam shifts), T05-U-03 (missing
tile -> zero mask), T05-U-04 (manifest round-trip + schema), T05-I-01
(pyramid levels match the base downsampling), and T06-U-01 (vertical datum,
control points <= 0.5 m), T06-U-02 (synthetic OSM fragment -> class raster),
T06-U-03 (validator rejects an inconsistent package), T06-U-04 (terrain
classifier agreement), T06-I-01 (full assembly, validator green).

## Implementation notes

- COG writing goes through GDAL's C API via ctypes: rasterio's writer silently
  falls back to GTiff for the COG driver, and the COG driver supports
  `Create()` (via GDALTranslate) but not `CreateCopy`.
- `overviews: [2, 4, 8]` gives a coarse level of ~4 m/px at 0.5 m/px base
  (cold start / LOST search).
- Tile cache layout: `{cache}/{provider}/{z}/{x}/{y}.jpg` + `.missing`
  markers for 404s, `tiles_meta.json` for provenance. On this dev host the
  fetcher pins IPv4 (IPv6 connects hang).
