"""Copernicus DEM tile naming, fetch URL construction and GLO-90 fallback (T06)."""

import urllib.error

import pytest

from mapprep.dem import (
    GLO30_BUCKET,
    GLO90_BUCKET,
    DemFetchError,
    fetch_dem_tiles,
    source_id_of_tile,
    tile_names_for,
)


def test_tile_names_for_maykop_corridor():
    bounds = {"west": 39.918, "south": 44.822, "east": 39.926, "north": 44.835}
    assert tile_names_for(bounds) == ["Copernicus_DSM_COG_10_N44_00_E039_00_DEM.tif"]


def test_tile_names_for_negative_lat_lon():
    bounds = {"west": -0.5, "south": -1.5, "east": 0.5, "north": -0.5}
    names = tile_names_for(bounds)
    assert "Copernicus_DSM_COG_10_S02_00_W001_00_DEM.tif" in names


def test_fetch_url_nests_tile_in_matching_directory(cache_root, monkeypatch):
    # The AWS Open Data bucket for Copernicus DEM GLO-30 stores each tile
    # inside a "directory" (S3 key prefix) that repeats the tile's own name,
    # e.g. .../Copernicus_DSM_COG_10_N44_00_E039_00_DEM/<same-name>.tif -- a
    # flat <bucket>/<name>.tif request 404s.
    requested_urls = []

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"fake-tif-bytes"

    def fake_urlopen(request, timeout=120):
        requested_urls.append(request.full_url)
        return _FakeResponse()

    monkeypatch.setattr("mapprep.dem.urllib.request.urlopen", fake_urlopen)

    bounds = {"west": 39.918, "south": 44.822, "east": 39.926, "north": 44.835}
    paths = fetch_dem_tiles(bounds, cache_dir=cache_root)

    name = "Copernicus_DSM_COG_10_N44_00_E039_00_DEM.tif"
    assert requested_urls == [f"{GLO30_BUCKET}/{name[: -len('.tif')]}/{name}"]
    assert paths == [cache_root / name]
    assert paths[0].read_bytes() == b"fake-tif-bytes"


# --- GLO-30 coverage holes (regression, 2026-08-27) -------------------------
#
# Copernicus GLO-30's *public* release omits whole countries; Armenia is one,
# so the AMtown site's tile (N39/E044) 404s while its neighbour E043 exists.
# The build used to die there. A corridor can now declare a GLO-90 fallback.

AMTOWN_BOUNDS = {"west": 44.8215, "south": 39.9180, "east": 44.8355, "north": 39.9265}
GLO30_TILE = "Copernicus_DSM_COG_10_N39_00_E044_00_DEM.tif"
GLO90_TILE = "Copernicus_DSM_COG_30_N39_00_E044_00_DEM.tif"


class _FakeResponse:
    def __init__(self, payload=b"fake-tif-bytes"):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.payload


def _fake_bucket(monkeypatch, available, log):
    """Serve only the URLs in `available`; everything else 404s."""

    def fake_urlopen(request, timeout=120):
        log.append(request.full_url)
        if request.full_url in available:
            return _FakeResponse()
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr("mapprep.dem.urllib.request.urlopen", fake_urlopen)


def _url(bucket, name):
    return f"{bucket}/{name[: -len('.tif')]}/{name}"


def test_tile_names_follow_the_source_arcsec_code():
    assert tile_names_for(AMTOWN_BOUNDS) == [GLO30_TILE]
    assert tile_names_for(AMTOWN_BOUNDS, "copernicus_glo90") == [GLO90_TILE]


def test_source_id_read_back_off_the_tile_name():
    assert source_id_of_tile(GLO30_TILE) == "copernicus_glo30"
    assert source_id_of_tile(GLO90_TILE) == "copernicus_glo90"
    assert source_id_of_tile("something_else.tif") == "unknown"


def test_glo90_fallback_used_when_glo30_has_no_tile(cache_root, monkeypatch):
    log = []
    _fake_bucket(monkeypatch, {_url(GLO90_BUCKET, GLO90_TILE)}, log)

    paths = fetch_dem_tiles(
        AMTOWN_BOUNDS,
        cache_dir=cache_root,
        fallback_source_ids=("copernicus_glo90",),
    )

    assert log == [_url(GLO30_BUCKET, GLO30_TILE), _url(GLO90_BUCKET, GLO90_TILE)]
    assert paths == [cache_root / GLO90_TILE]
    assert source_id_of_tile(paths[0].name) == "copernicus_glo90"


def test_no_fallback_configured_still_fails_loudly(cache_root, monkeypatch):
    log = []
    _fake_bucket(monkeypatch, {_url(GLO90_BUCKET, GLO90_TILE)}, log)
    with pytest.raises(DemFetchError, match="no DEM coverage"):
        fetch_dem_tiles(AMTOWN_BOUNDS, cache_dir=cache_root)


def test_glo30_is_preferred_when_it_does_have_the_tile(cache_root, monkeypatch):
    """The fallback must not quietly coarsen a corridor that GLO-30 covers."""
    log = []
    _fake_bucket(monkeypatch, {_url(GLO30_BUCKET, GLO30_TILE)}, log)

    paths = fetch_dem_tiles(
        AMTOWN_BOUNDS,
        cache_dir=cache_root,
        fallback_source_ids=("copernicus_glo90",),
    )

    assert log == [_url(GLO30_BUCKET, GLO30_TILE)]
    assert paths == [cache_root / GLO30_TILE]


def test_non_404_errors_are_not_treated_as_missing_coverage(cache_root, monkeypatch):
    """A 503 or a dropped connection must fail the build, not silently coarsen it."""

    def fake_urlopen(request, timeout=120):
        raise urllib.error.HTTPError(request.full_url, 503, "Service Unavailable", {}, None)

    monkeypatch.setattr("mapprep.dem.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(DemFetchError, match="failed to download"):
        fetch_dem_tiles(
            AMTOWN_BOUNDS,
            cache_dir=cache_root,
            fallback_source_ids=("copernicus_glo90",),
        )


def test_cached_fallback_tile_is_reused_without_network(cache_root, monkeypatch):
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / GLO90_TILE).write_bytes(b"cached")

    def explode(request, timeout=120):
        raise AssertionError("should not hit the network for a cached tile")

    monkeypatch.setattr("mapprep.dem.urllib.request.urlopen", explode)
    paths = fetch_dem_tiles(
        AMTOWN_BOUNDS,
        cache_dir=cache_root,
        offline=True,
        fallback_source_ids=("copernicus_glo90",),
    )
    assert paths == [cache_root / GLO90_TILE]
