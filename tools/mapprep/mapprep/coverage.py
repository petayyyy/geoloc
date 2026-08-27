"""Detecting tiles a provider served but has no imagery for (T05/T06).

A tile server answering HTTP 200 does not mean it has a picture of the ground.
Esri's World Imagery returns a grey "Map data not yet available" card for any
zoom it lacks locally -- a perfectly valid 2.5 KB JPEG. Bing and Yandex have
their own equivalents.

This matters more here than "the mosaic looks bad". The whole localization
architecture rests on matching a camera patch against a *reference* basemap,
and a basemap of placeholder cards that reports 100% validity is precisely the
input that produces confident wrong fixes -- the thing IFR < 0.5% exists to
prevent. So placeholders are detected, excluded from the mosaic, marked
invalid in the validity mask, and counted; and a layer that is mostly
placeholder fails the build instead of shipping.

A tile is called a placeholder only when **both** signals agree:

1. it carries no texture (greyscale std below `placeholder_std`; the Esri card
   measures ~5, real imagery over the same village ~55), and
2. it is byte-identical to several other tiles in the same layer.

Either alone is too blunt. Real terrain can be textureless -- open water,
fresh snow -- and would be condemned by (1); and (2) alone would flag a
uniform field that happens to compress identically. Together they describe
what a placeholder actually is: the same textureless card repeated.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_PLACEHOLDER_STD = 8.0
DEFAULT_PLACEHOLDER_MIN_REPEATS = 4


def tile_texture_std(path: Path) -> float:
    """Greyscale standard deviation of a cached tile, 0.0 if unreadable."""
    try:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("L"), dtype=np.float64)
    except Exception:
        return 0.0
    return float(arr.std())


def tile_digest(path: Path) -> str:
    """Content hash of a cached tile, "" if unreadable."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def find_placeholder_tiles(
    paths: list[Path],
    std_threshold: float = DEFAULT_PLACEHOLDER_STD,
    min_repeats: int = DEFAULT_PLACEHOLDER_MIN_REPEATS,
) -> set[Path]:
    """Which of these cached tiles are the provider's "no imagery here" card.

    Args:
        paths: cached tile paths belonging to one layer.
        std_threshold: texture floor; 0 or less disables the check entirely.
        min_repeats: how many byte-identical copies make a repeat suspicious.

    Returns:
        The subset of `paths` that is both textureless and repeated.
    """
    if std_threshold <= 0.0 or not paths:
        return set()
    digests = {path: tile_digest(path) for path in paths}
    repeated = {
        digest
        for digest, count in Counter(digests.values()).items()
        if digest and count >= min_repeats
    }
    if not repeated:
        return set()
    # Texture is measured once per distinct digest, not once per tile.
    textureless = {
        digest
        for digest in repeated
        if tile_texture_std(next(p for p, d in digests.items() if d == digest)) < std_threshold
    }
    return {path for path, digest in digests.items() if digest in textureless}
