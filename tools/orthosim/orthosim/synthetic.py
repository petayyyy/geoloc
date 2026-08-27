"""Procedural synthetic scenes for the ``adversarial`` (and self-contained smoke)
sets.

The adversarial set is not about accuracy -- it measures how often the matcher
is *confidently wrong* (periodic grids, featureless water/snow, stale maps,
symmetric interchanges). The real geopack does not contain those scenes, so they
are generated here as deterministic functions of (east, north). The query and
map providers are always two different functions (never the same object), which
keeps the cross-provider rule meaningful even in a synthetic scene.
"""

from __future__ import annotations

import numpy as np

from .dsm import FlatTerrain


def _hash2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Deterministic integer hash -> [0, 1)."""
    xi = x.astype(np.int64)
    yi = y.astype(np.int64)
    n = (xi * 374761393 + yi * 668265263) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFF) / 65535.0


def value_noise(east: np.ndarray, north: np.ndarray, freq: float, seed: int) -> np.ndarray:
    xi = np.floor(east * freq).astype(np.int64) + seed
    yi = np.floor(north * freq).astype(np.int64)
    return _hash2(xi, yi)


def _gray(east: np.ndarray, north: np.ndarray, base: float, amp: float, seed: int) -> np.ndarray:
    return base + amp * (2.0 * value_noise(east, north, 0.4, seed) - 1.0)


class SyntheticScene:
    """Duck-typed to the :class:`~orthosim.scene.Scene` interface (sample/z)."""

    def __init__(self, texture_a, texture_b, terrain=None, semantic=None, crs="EPSG:32637"):
        self.texture_a = texture_a
        self.texture_b = texture_b
        self.terrain = terrain if terrain is not None else FlatTerrain(0.0)
        self.semantic = semantic
        self.crs = crs

    def sample(self, provider: str, east: np.ndarray, north: np.ndarray) -> np.ndarray:
        fn = self.texture_a if provider == "a" else self.texture_b
        e = np.asarray(east, dtype=np.float64)
        n = np.asarray(north, dtype=np.float64)
        return np.asarray(fn(e, n))

    def z(self, east: np.ndarray, north: np.ndarray) -> np.ndarray:
        return np.asarray(self.terrain.z(np.asarray(east), np.asarray(north)), dtype=np.float64)


def make_synthetic_scene(kind: str, seed: int = 0, *, gsd: float = 0.5):
    """Build a ``SyntheticScene`` for the named adversarial kind.

    ``kind`` is one of: ``periodic`` (crop grid / greenhouses / warehouse rows),
    ``water``, ``snow``, ``forest``, ``stale_map`` (new construction against an
    older map) or ``symmetric`` (90/180-deg symmetric interchange).
    """
    if kind == "periodic":
        return _periodic(seed)
    if kind == "water":
        return _uniform(seed, base=40, amp=4)
    if kind == "snow":
        return _uniform(seed, base=235, amp=6)
    if kind == "forest":
        return _forest(seed)
    if kind == "stale_map":
        return _stale(seed)
    if kind == "symmetric":
        return _symmetric(seed)
    raise ValueError(f"unknown synthetic scene kind {kind!r}")


def _periodic(seed: int) -> SyntheticScene:
    period = 25.0

    def base(east, north, tint):
        col = np.floor(east / period).astype(np.int64)
        row = np.floor(north / period).astype(np.int64)
        cell = np.mod(col + 2 * row, 4)
        v = np.where(cell == 0, 60.0, np.where(cell == 1, 140.0, np.where(cell == 2, 90.0, 170.0)))
        v = v + 12.0 * (2.0 * value_noise(east, north, 1.0, seed) - 1.0)
        r = np.clip(v * (1.0 + tint), 0, 255)
        g = np.clip(v * (1.0 - 0.2 * tint), 0, 255)
        b = np.clip(v * (1.0 - 0.4 * tint), 0, 255)
        return np.stack([r, g, b], axis=-1)

    ta = lambda e, n: base(e, n, 0.0)  # noqa: E731
    tb = lambda e, n: base(e, n, 0.25)  # noqa: E731
    return SyntheticScene(ta, tb)


def _uniform(seed: int, base: float, amp: float) -> SyntheticScene:
    def mk(tint):
        def f(east, north):
            v = base + amp * (2.0 * value_noise(east, north, 0.15, seed) - 1.0)
            r = np.clip(v * (1.0 + tint), 0, 255)
            g = np.clip(v, 0, 255)
            b = np.clip(v * (1.0 - tint), 0, 255)
            return np.stack([r, g, b], axis=-1)

        return f

    return SyntheticScene(mk(0.0), mk(0.1))


def _forest(seed: int) -> SyntheticScene:
    def mk(tint):
        def f(east, north):
            n = value_noise(east, north, 1.5, seed)
            g = 40 + 160 * n
            r = np.clip(g * 0.5 * (1.0 + tint), 0, 255)
            gg = np.clip(g, 0, 255)
            b = np.clip(g * 0.35, 0, 255)
            return np.stack([r, gg, b], axis=-1)

        return f

    return SyntheticScene(mk(0.0), mk(0.2))


def _stale(seed: int) -> SyntheticScene:
    """Map is 2 years older: query shows a new building the map is missing."""

    def build(*, extra):
        def f(east, north):
            v = 90.0 + 40.0 * (2.0 * value_noise(east, north, 0.5, seed) - 1.0)
            out = np.stack([v, v, v * 0.9], axis=-1)
            for cx, cy, w, h, shade in [(0.0, 0.0, 30.0, 20.0, 200.0)] + extra:
                inside = (np.abs(east - cx) < w / 2) & (np.abs(north - cy) < h / 2)
                out[inside] = shade
            return out

        return f

    ta = build(extra=[])
    tb = build(extra=[(60.0, 10.0, 40.0, 25.0, 210.0)])  # new building only in query
    return SyntheticScene(ta, tb)


def _symmetric(seed: int) -> SyntheticScene:
    """Interchange with 90/180-deg symmetry: roads in a cloverleaf."""

    def mk(tint):
        def f(east, north):
            v = 80.0 + 20.0 * (2.0 * value_noise(east, north, 0.4, seed) - 1.0)
            out = np.stack([v, v, v], axis=-1)
            for cx, cy in [(0.0, 0.0), (60.0, 0.0), (-60.0, 0.0), (0.0, 60.0), (0.0, -60.0)]:
                near = np.minimum(np.abs(east - cx), np.abs(north - cy)) < 6.0
                out[near] = np.array([120.0, 120.0, 120.0]) * (1.0 + tint)
            return out

        return f

    return SyntheticScene(mk(0.0), mk(0.15))
