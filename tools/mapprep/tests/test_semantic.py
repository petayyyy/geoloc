"""Overpass query construction for the OSM semantic layer (T06)."""

import urllib.error
import urllib.parse

import pytest

from mapprep.semantic import (
    OVERPASS_URL,
    OVERPASS_URLS,
    OsmFetchError,
    fetch_overpass,
    overpass_query,
)

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


# --- Overpass load handling (regression, 2026-08-27) ------------------------
#
# The public Overpass instance runs a small fixed pool of query slots and
# answers 504 (not 429) when they are all busy. A single-shot fetch turned
# that into a dead geopack build -- after both ortho layers and the DEM had
# already been assembled -- for a query that succeeds in ~2 s when a slot is
# free. Retries rotate endpoints; a malformed query still fails immediately.


def _http_error(url, code):
    return urllib.error.HTTPError(url, code, "boom", {}, None)


class _Reply:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.payload


def _recording_urlopen(monkeypatch, responses):
    """Serve `responses` in order; each is either an int status or bytes."""
    seen = []

    def fake_urlopen(request, timeout=90.0):
        seen.append(request.full_url)
        item = responses[len(seen) - 1]
        if isinstance(item, int):
            raise _http_error(request.full_url, item)
        return _Reply(item)

    monkeypatch.setattr("mapprep.semantic.urllib.request.urlopen", fake_urlopen)
    return seen


def test_504_is_retried_on_the_next_endpoint(monkeypatch):
    seen = _recording_urlopen(monkeypatch, [504, b"{}"])
    slept = []

    text = fetch_overpass(BOUNDS, sleep=slept.append, backoff_s=5.0)

    assert text == "{}"
    assert seen == [OVERPASS_URLS[0], OVERPASS_URLS[1]]
    assert slept == [5.0]  # deterministic backoff, no jitter


def test_backoff_grows_and_endpoints_rotate(monkeypatch):
    seen = _recording_urlopen(monkeypatch, [504, 502, 429, b"{}"])
    slept = []

    assert fetch_overpass(BOUNDS, sleep=slept.append, backoff_s=2.0) == "{}"
    assert seen == [OVERPASS_URLS[0], OVERPASS_URLS[1], OVERPASS_URLS[2], OVERPASS_URLS[0]]
    assert slept == [2.0, 4.0, 6.0]


def test_all_attempts_exhausted_reports_every_endpoint(monkeypatch):
    _recording_urlopen(monkeypatch, [504, 504, 504, 504])
    with pytest.raises(OsmFetchError) as excinfo:
        fetch_overpass(BOUNDS, sleep=lambda _s: None)
    message = str(excinfo.value)
    assert "after 4 attempt(s)" in message
    for url in OVERPASS_URLS:
        assert url in message


def test_malformed_query_is_not_retried(monkeypatch):
    """400 means the query is wrong; retrying only burns other people's slots."""
    seen = _recording_urlopen(monkeypatch, [400, b"{}"])
    with pytest.raises(OsmFetchError, match="failed to query Overpass"):
        fetch_overpass(BOUNDS, sleep=lambda _s: None)
    assert seen == [OVERPASS_URL]


def test_no_sleep_after_the_final_attempt(monkeypatch):
    _recording_urlopen(monkeypatch, [504, 504])
    slept = []
    with pytest.raises(OsmFetchError):
        fetch_overpass(BOUNDS, attempts=2, sleep=slept.append, backoff_s=1.0)
    assert slept == [1.0]


def test_connection_errors_are_retried_too(monkeypatch):
    def fake_urlopen(request, timeout=90.0):
        if request.full_url == OVERPASS_URLS[0]:
            raise urllib.error.URLError("connection reset")
        return _Reply(b"{}")

    monkeypatch.setattr("mapprep.semantic.urllib.request.urlopen", fake_urlopen)
    assert fetch_overpass(BOUNDS, sleep=lambda _s: None) == "{}"
