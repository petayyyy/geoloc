"""Basemap preparation tooling for geoloc (task T05).

Tiles from multiple providers are fetched (or imported from an existing cache)
and assembled into UTM COG mosaics with an honest validity mask and a manifest
that follows docs/plan/03-interfaces.md section 4.
"""

__version__ = "0.1.0"
