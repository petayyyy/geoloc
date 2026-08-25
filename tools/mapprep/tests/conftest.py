import warnings

import numpy as np
import pytest
import rasterio

from mapprep import webmercator
from mapprep.providers import get_provider

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

Z = 18
LON0, LAT0 = 39.922, 44.8285
X0, Y0 = webmercator.lonlat_to_tile(LON0, LAT0, Z)
NATIVE_GSD = webmercator.gsd_m(LAT0, Z)
PROVIDER = get_provider("esri_world_imagery")


def tile_color(x: int, y: int) -> np.ndarray:
    return np.array(
        [(17 * x + 53 * y) % 256, (7 * x + 29 * y) % 256, (101 * x + 11 * y) % 256],
        dtype=np.uint8,
    )


def write_tile(cache_root, x: int, y: int, z: int, color: np.ndarray) -> None:
    path = cache_root / PROVIDER.id / f"{z}/{x}/{y}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="JPEG",
        width=256,
        height=256,
        count=3,
        dtype="uint8",
        quality=85,
    ) as ds:
        ds.write(np.broadcast_to(color.reshape(3, 1, 1), (3, 256, 256)))


def bounds_for_tiles(x0: int, y0: int, x1: int, y1: int) -> dict:
    nw = webmercator.tile_corners_lonlat(x0, y0, Z)[0]
    se = webmercator.tile_corners_lonlat(x1, y1, Z)[2]
    inset = 1e-9
    return {
        "west": nw[0] + inset,
        "north": nw[1] - inset,
        "east": se[0] - inset,
        "south": se[1] + inset,
    }


@pytest.fixture
def cache_root(tmp_path):
    return tmp_path / "cache"


@pytest.fixture
def bounds_2x2():
    return bounds_for_tiles(X0, Y0, X0 + 1, Y0 + 1)


@pytest.fixture
def filled_cache_2x2(cache_root, bounds_2x2):
    tiles = webmercator.tiles_for_bounds(
        bounds_2x2["west"], bounds_2x2["south"], bounds_2x2["east"], bounds_2x2["north"], Z
    )
    for x, y in tiles:
        write_tile(cache_root, x, y, Z, tile_color(x, y))
    return tiles
