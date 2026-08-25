"""Vertical datum conversion between orthometric (EGM2008) and ellipsoidal.

The conversion is arithmetic once the geoid undulation N(lat, lon) is known:

    ellipsoidal = orthometric + N
    orthometric = ellipsoidal - N

N comes from a geoid model (see geoid.py). This module keeps the conversion
logic separate from the geoid source so it is testable with synthetic geoids
against control points (T06-U-01).
"""

from __future__ import annotations

import numpy as np


def orthometric_to_ellipsoidal(
    ortho: np.ndarray, lat: np.ndarray, lon: np.ndarray, geoid
) -> np.ndarray:
    """Convert orthometric heights to ellipsoidal heights (metres)."""
    return np.asarray(ortho, dtype=np.float32) + geoid.undulation_array(lat, lon)


def ellipsoidal_to_orthometric(
    ellip: np.ndarray, lat: np.ndarray, lon: np.ndarray, geoid
) -> np.ndarray:
    """Convert ellipsoidal heights to orthometric heights (metres)."""
    return np.asarray(ellip, dtype=np.float32) - geoid.undulation_array(lat, lon)
