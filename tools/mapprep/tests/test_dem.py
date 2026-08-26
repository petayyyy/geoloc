"""Copernicus GLO-30 tile naming and fetch URL construction (T06)."""

from mapprep.dem import GLO30_BUCKET, fetch_dem_tiles, tile_names_for


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
