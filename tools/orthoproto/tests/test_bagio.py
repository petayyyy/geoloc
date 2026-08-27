"""bagio.py: image decoding (T15 colour-channel regression)."""

import numpy as np
import pytest

from orthoproto.bagio import _downscale_rgb, decode_image_rgb


def test_bgr8_is_channel_reversed_to_rgb():
    # A single red pixel and a single blue pixel, as OpenCV/ROS bgr8 stores
    # them: byte order (B, G, R) per pixel.
    bgr = np.array([[[0, 0, 255], [255, 0, 0]]], dtype=np.uint8)  # red px, blue px in BGR bytes
    rgb = decode_image_rgb(bgr.tobytes(), height=1, width=2, encoding="bgr8")
    assert rgb.shape == (1, 2, 3)
    assert list(rgb[0, 0]) == [255, 0, 0]  # red
    assert list(rgb[0, 1]) == [0, 0, 255]  # blue


def test_rgb8_passes_through_unchanged():
    rgb_in = np.array([[[10, 20, 30]]], dtype=np.uint8)
    rgb_out = decode_image_rgb(rgb_in.tobytes(), height=1, width=1, encoding="rgb8")
    assert list(rgb_out[0, 0]) == [10, 20, 30]


def test_unknown_encoding_raises():
    with pytest.raises(ValueError, match="mono8"):
        decode_image_rgb(b"\x00", height=1, width=1, encoding="mono8")


# --- image topic and read-time downscale (2026-08-27) ----------------------
#
# /rgb_img is FAST-LIVO2's debug visualization: the VIO image with projected
# lidar points drawn on it in green and blue. Building the ortho patch from it
# burns synthetic marks into the one image the matcher compares against the
# satellite basemap. /left_camera/image is the raw sensor frame.



def test_downscale_area_averages_rather_than_subsamples():
    """A dot smaller than one output pixel must dim, not survive at full value."""
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[4:6, 4:6] = 255  # a 2x2 "lidar dot"
    out = _downscale_rgb(img, 0.25)
    assert out.shape == (4, 4, 3)
    assert out.max() < 255, "subsampling would have kept a 255 pixel"
    assert out.max() > 0, "the dot's energy must survive somewhere"
    assert int(out.sum()) == pytest.approx(int(img.sum()) // 16, rel=0.05)


def test_downscale_preserves_geometry_scale():
    img = np.zeros((2048, 2448, 3), dtype=np.uint8)
    out = _downscale_rgb(img, 0.25)
    assert out.shape == (512, 612, 3), "must match the intrinsics' working size"


def test_downscale_identity_and_bounds():
    img = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
    assert np.array_equal(_downscale_rgb(img, 1.0), img)
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="downscale must be"):
            _downscale_rgb(img, bad)
