"""bagio.py: image decoding (T15 colour-channel regression)."""

import numpy as np
import pytest

from orthoproto.bagio import decode_image_rgb


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
