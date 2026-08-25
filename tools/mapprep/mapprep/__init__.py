"""Map preparation tooling for geoloc (tasks T05 + T06).

Ortho tiles from multiple providers are fetched (or imported from an existing
cache) and assembled into UTM COG mosaics with an honest validity mask (T05).
The Copernicus GLO-30 DEM and an OSM semantic class raster are added with
vertical-datum conversion, and the package is validated for cross-layer
consistency (T06). The manifest follows docs/plan/03-interfaces.md section 4.
"""

__version__ = "0.2.0"
