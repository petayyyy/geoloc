"""Best-effort capture-date enrichment for providers that expose metadata.

Esri World Imagery used to answer per-tile identify queries with acquisition
dates; the public identify endpoint now returns empty results. The Wayback
service lists the current release date, but not per-tile dates. This module
probes ONCE per layer (never one request per tile) and returns None when the
provider does not expose dates -- absence is recorded honestly in the
manifest, never guessed.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from .fetch import USER_AGENT
from .providers import Provider

ESRI_IDENTIFY_URL = (
    "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/identify"
)
ESRI_WAYBACK_URL = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer"
)


def _esri_identify(lon: float, lat: float, timeout_s: float = 15.0) -> str | None:
    params = {
        "f": "json",
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "tolerance": "5",
        "mapExtent": f"{lon - 0.001},{lat - 0.001},{lon + 0.001},{lat + 0.001}",
        "imageDisplay": "256,256,96",
        "returnGeometry": "false",
        "layers": "all:0",
    }
    url = f"{ESRI_IDENTIFY_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, json.JSONDecodeError):
        return None
    for result in payload.get("results") or []:
        attributes = result.get("attributes") or {}
        for key in ("AcquisitionDate", "Date", "SRC_DATE", "acquisition_date"):
            value = attributes.get(key)
            if value:
                return str(value)
    return None


def esri_latest_release(timeout_s: float = 15.0) -> str | None:
    """Latest World Imagery release date from the Wayback service list."""
    url = f"{ESRI_WAYBACK_URL}?f=json"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, json.JSONDecodeError):
        return None
    selection = payload.get("Selection") or []
    if not selection:
        return None
    return str(selection[0].get("Name", ""))


def enrich_capture_dates(
    provider: Provider,
    bounds_wgs84: dict,
    zoom: int,
    cache_root,
    *,
    request_interval_s: float = 0.2,
) -> str | None:
    """Return the layer capture date when the provider exposes it, else None."""
    del cache_root, request_interval_s
    if provider.id != "esri_world_imagery":
        return None
    center_lon = (bounds_wgs84["west"] + bounds_wgs84["east"]) / 2
    center_lat = (bounds_wgs84["south"] + bounds_wgs84["north"]) / 2
    return _esri_identify(center_lon, center_lat) or None
