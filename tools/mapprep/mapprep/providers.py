"""Provider registry: tile URL schemes, zoom ranges, licensing notes.

ADR-008 (basemap providers and licensing) is OPEN: terms below are recorded
honestly for internal development; commercial-use rights must NOT be claimed
in a manifest until ADR-008 is closed.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import webmercator


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    scheme: str  # "xyz" | "quadkey"
    url_template: str  # {z}, {x}, {y} or {q}; {s} subdomain
    subdomains: tuple[str, ...]
    tile_ext: str
    min_zoom: int
    max_zoom: int
    license: str
    attribution: str
    capture_date: str | None  # best known date for the service as a whole, or None

    def tile_url(self, x: int, y: int, z: int, subdomain_index: int = 0) -> str:
        s = self.subdomains[subdomain_index % len(self.subdomains)]
        if self.scheme == "quadkey":
            q = webmercator.tile_to_quadkey(x, y, z)
            return self.url_template.format(s=s, q=q, z=z)
        return self.url_template.format(s=s, x=x, y=y, z=z)

    def cache_rel_path(self, x: int, y: int, z: int) -> str:
        return f"{z}/{x}/{y}.{self.tile_ext}"


ESRI_LICENSE = (
    "Esri World Imagery, terms at https://www.esri.com/legal/terms/full-master-agreement "
    "and esri.com/data-attribution; no offline redistribution rights claimed. "
    "Internal development use only pending ADR-008."
)
BING_LICENSE = (
    "Microsoft Bing Maps aerial imagery; offline caching and derived use are restricted "
    "by the Bing Maps Platform Terms of Use. Internal development use only pending ADR-008."
)
YANDEX_LICENSE = (
    "Yandex Maps satellite imagery; Yandex Maps API terms restrict copying and offline "
    "storage. Internal development use only pending ADR-008."
)

PROVIDERS = {
    "esri_world_imagery": Provider(
        id="esri_world_imagery",
        name="Esri World Imagery",
        scheme="xyz",
        url_template=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
            "MapServer/tile/{z}/{y}/{x}"
        ),
        subdomains=("",),
        tile_ext="jpg",
        min_zoom=1,
        max_zoom=19,
        license=ESRI_LICENSE,
        attribution="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        capture_date=None,
    ),
    "bing_aerial": Provider(
        id="bing_aerial",
        name="Bing Maps Aerial",
        scheme="quadkey",
        url_template="https://ecn.{s}.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1",
        subdomains=("t0", "t1", "t2", "t3"),
        tile_ext="jpg",
        min_zoom=1,
        max_zoom=19,
        license=BING_LICENSE,
        attribution="Microsoft Bing Maps",
        capture_date=None,
    ),
    "yandex_satellite": Provider(
        id="yandex_satellite",
        name="Yandex Satellite",
        scheme="xyz",
        url_template=(
            "https://{s}.maps.yandex.net/tiles?l=sat&x={x}&y={y}&z={z}&scale=1&lang=en_US"
        ),
        subdomains=("core-sat", "sat01", "sat02", "sat03"),
        tile_ext="jpg",
        min_zoom=1,
        max_zoom=18,
        license=YANDEX_LICENSE,
        attribution="Yandex Maps",
        capture_date=None,
    ),
}


def get_provider(provider_id: str) -> Provider:
    try:
        return PROVIDERS[provider_id]
    except KeyError as exc:
        raise KeyError(f"unknown provider {provider_id!r}; known: {sorted(PROVIDERS)}") from exc
