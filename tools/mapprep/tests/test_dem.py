"""Copernicus DEM tile naming, fetch URL construction and GLO-90 fallback (T06)."""

import urllib.error

import pytest

from mapprep.dem import (
    GLO30_BUCKET,
    GLO90_BUCKET,
    OPENTOPO_URL,
    DemFetchError,
    dem_source,
    fetch_dem_tiles,
    opentopography_tile_name,
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


# --- OpenTopography COP30 source -------------------------------------------
#
# Same GLO-30 data, full coverage (the open S3 bucket omits whole countries),
# behind a free API key. The key lives in an environment variable named by the
# corridor config and must never reach a cache filename, a log line or a
# traceback.

OT_KEY_ENV = "TEST_OPENTOPO_KEY"
OT_TILE = opentopography_tile_name(dem_source("opentopography_cop30"), AMTOWN_BOUNDS)


def _fake_opentopo(monkeypatch, handler):
    log = []

    def fake_urlopen(request, timeout=300):
        log.append(request.full_url)
        return handler(request)

    monkeypatch.setattr("mapprep.dem.urllib.request.urlopen", fake_urlopen)
    return log


def test_opentopography_cache_name_carries_the_bbox_not_the_key():
    assert OT_TILE.startswith("OpenTopography_COP30_")
    assert OT_TILE.endswith("_DEM.tif")
    # bbox with the default 0.01 deg margin, rounded -- deterministic
    assert "S39.9080_N39.9365_W44.8115_E44.8455" in OT_TILE
    assert source_id_of_tile(OT_TILE) == "opentopography_cop30"


def test_opentopography_fetch_sends_the_key_and_caches_by_bbox(cache_root, monkeypatch):
    monkeypatch.setenv(OT_KEY_ENV, "secret-key-value")
    log = _fake_opentopo(monkeypatch, lambda request: _FakeResponse())

    paths = fetch_dem_tiles(
        AMTOWN_BOUNDS,
        cache_dir=cache_root,
        source_id="opentopography_cop30",
        api_key_env=OT_KEY_ENV,
    )

    assert len(log) == 1
    assert log[0].startswith(OPENTOPO_URL)
    assert "demtype=COP30" in log[0]
    assert "API_Key=secret-key-value" in log[0]
    assert paths == [cache_root / OT_TILE]
    assert "secret-key-value" not in paths[0].name


def test_missing_api_key_env_fails_loudly_instead_of_coarsening(cache_root, monkeypatch):
    """A missing key must not silently drop the build to 90 m data."""
    monkeypatch.delenv(OT_KEY_ENV, raising=False)
    with pytest.raises(DemFetchError, match=f"{OT_KEY_ENV} is unset"):
        fetch_dem_tiles(
            AMTOWN_BOUNDS,
            cache_dir=cache_root,
            source_id="opentopography_cop30",
            fallback_source_ids=("copernicus_glo90",),
            api_key_env=OT_KEY_ENV,
        )


def test_unconfigured_api_key_env_names_the_config_key(cache_root, monkeypatch):
    with pytest.raises(DemFetchError, match="dem.api_key_env"):
        fetch_dem_tiles(
            AMTOWN_BOUNDS, cache_dir=cache_root, source_id="opentopography_cop30"
        )


def test_rejected_api_key_is_a_hard_error_not_a_fallback(cache_root, monkeypatch):
    monkeypatch.setenv(OT_KEY_ENV, "bad-key")

    def handler(request):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    _fake_opentopo(monkeypatch, handler)
    with pytest.raises(DemFetchError, match="rejected the API key"):
        fetch_dem_tiles(
            AMTOWN_BOUNDS,
            cache_dir=cache_root,
            source_id="opentopography_cop30",
            fallback_source_ids=("copernicus_glo90",),
            api_key_env=OT_KEY_ENV,
        )


def test_api_key_is_redacted_from_error_messages(cache_root, monkeypatch):
    monkeypatch.setenv(OT_KEY_ENV, "super-secret")

    def handler(request):
        raise urllib.error.HTTPError(request.full_url, 500, "boom", {}, None)

    _fake_opentopo(monkeypatch, handler)
    with pytest.raises(DemFetchError) as excinfo:
        fetch_dem_tiles(
            AMTOWN_BOUNDS,
            cache_dir=cache_root,
            source_id="opentopography_cop30",
            api_key_env=OT_KEY_ENV,
        )
    message = str(excinfo.value)
    assert "super-secret" not in message
    assert "<redacted>" in message


def test_opentopography_no_coverage_falls_through_to_glo90(cache_root, monkeypatch):
    monkeypatch.setenv(OT_KEY_ENV, "key")
    calls = []

    def fake_urlopen(request, timeout=300):
        calls.append(request.full_url)
        if request.full_url.startswith(OPENTOPO_URL):
            raise urllib.error.HTTPError(request.full_url, 204, "No Content", {}, None)
        return _FakeResponse()

    monkeypatch.setattr("mapprep.dem.urllib.request.urlopen", fake_urlopen)

    paths = fetch_dem_tiles(
        AMTOWN_BOUNDS,
        cache_dir=cache_root,
        source_id="opentopography_cop30",
        fallback_source_ids=("copernicus_glo90",),
        api_key_env=OT_KEY_ENV,
    )

    assert paths == [cache_root / GLO90_TILE]
    assert len(calls) == 2
