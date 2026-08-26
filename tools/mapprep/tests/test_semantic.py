"""Overpass query construction for the OSM semantic layer (T06)."""

import urllib.parse

from mapprep.semantic import OVERPASS_URL, fetch_overpass, overpass_query

BOUNDS = {"west": 39.918, "south": 44.822, "east": 39.926, "north": 44.835}


def test_overpass_query_bbox_is_unquoted():
    # Overpass QL bbox filters take four bare numbers: way["k"](s,w,n,e).
    # Wrapping them in quotes turns it into a string literal and the server
    # returns HTTP 400 -- this regressed once already (T06 build failure).
    query = overpass_query(BOUNDS)
    bbox = f"{BOUNDS['south']},{BOUNDS['west']},{BOUNDS['north']},{BOUNDS['east']}"
    assert f"({bbox})" in query
    assert f'("{bbox}")' not in query


def test_fetch_overpass_posts_the_query(monkeypatch):
    posted = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"elements": []}'

    def fake_urlopen(request, timeout=90.0):
        posted["url"] = request.full_url
        posted["data"] = request.data
        return _FakeResponse()

    monkeypatch.setattr("mapprep.semantic.urllib.request.urlopen", fake_urlopen)

    text = fetch_overpass(BOUNDS)

    assert text == '{"elements": []}'
    assert posted["url"] == OVERPASS_URL
    decoded = urllib.parse.parse_qs(posted["data"].decode("utf-8"))
    assert decoded["data"][0] == overpass_query(BOUNDS)
