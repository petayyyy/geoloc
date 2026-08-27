"""ROS 2 bag extraction for the orthoproto pipeline.

Reads a bag recorded from the FAST-LIVO2 pipeline and yields the topics the
offline prototype needs. All timestamps are converted to seconds of the bag's
simulation clock (header stamps -- the capture was recorded under
`use_sim_time`, so receive times are NOT the measurement times).

The bag is read once and cached to .npz per stream; clouds are streamed
(biggest topic, transformed and rasterized on the fly downstream).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

RTK_TOPIC = "/dji_osdk_ros/rtk_position"
RTK_VEL_TOPIC = "/dji_osdk_ros/rtk_velocity"
ODOM_TOPIC = "/aft_mapped_to_init"
CLOUD_TOPIC = "/cloud_registered"
RGB_TOPIC = "/rgb_img"
RAW_CAMERA_TOPIC = "/left_camera/image"


def decode_image_rgb(data: bytes, height: int, width: int, encoding: str) -> np.ndarray:
    """sensor_msgs/Image (3-channel, 8-bit) -> (H, W, 3) uint8 RGB.

    The capture's /rgb_img is published `bgr8` (OpenCV convention); this is
    the one place that reverses the channel order so every consumer
    downstream -- patch/mosaic writers, any future colour-based matching
    against the (RGB) satellite basemap -- works in RGB.
    """
    if encoding not in ("bgr8", "rgb8"):
        raise ValueError(f"unsupported /rgb_img encoding: {encoding!r}")
    img = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    if encoding == "bgr8":
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)


def _downscale_rgb(img: np.ndarray, scale: float) -> np.ndarray:
    """Area-average downscale to `scale` of the original size."""
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"downscale must be in (0, 1], got {scale}")
    height = int(round(img.shape[0] * scale))
    width = int(round(img.shape[1] * scale))
    box = getattr(Image, "Resampling", Image).BOX
    resized = Image.fromarray(img).resize((width, height), box)
    return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))


@dataclass
class Capture:
    bag_dir: Path

    def _read(self):
        return Reader(self.bag_dir)

    # -- extraction ---------------------------------------------------------

    def read_rtk(self, swap_latlon: bool) -> np.ndarray:
        """RTK fixes as (N, 4): (t_s, lat_deg, lon_deg, alt_m).

        `swap_latlon` exchanges the two fields, for a capture that genuinely
        recorded them the wrong way round. It is NOT the case for
        `geoloc_capture_01` -- its fields are correctly labelled, and swapping
        them silently anisotropically distorts the track (see
        `rtkcheck`). Never set it without running `orthoproto check` first.
        """
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        out = []
        with self._read() as reader:
            for conn, _ts, raw in reader.messages():
                if conn.topic != RTK_TOPIC:
                    continue
                m = typestore.deserialize_cdr(raw, conn.msgtype)
                t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                if swap_latlon:
                    lat, lon = m.longitude, m.latitude
                else:
                    lat, lon = m.latitude, m.longitude
                out.append((t, lat, lon, m.altitude))
        return np.asarray(out, dtype=np.float64)

    def read_rtk_velocity(self) -> np.ndarray:
        """RTK Doppler velocity as (N, 4): (t_s, vx, vy, vz).

        Axis meaning is receiver-dependent (NED on this DJI rig); the frame is
        config, not an assumption baked in here -- see `rtkcheck.velocity_en`.
        """
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        out = []
        with self._read() as reader:
            for conn, _ts, raw in reader.messages():
                if conn.topic != RTK_VEL_TOPIC:
                    continue
                m = typestore.deserialize_cdr(raw, conn.msgtype)
                t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                out.append((t, m.vector.x, m.vector.y, m.vector.z))
        return np.asarray(out, dtype=np.float64)

    def read_odom(self) -> np.ndarray:
        """Odometry as (N, 8): (t_s, x, y, z, qw, qx, qy, qz) in camera_init."""
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        out = []
        with self._read() as reader:
            for conn, _ts, raw in reader.messages():
                if conn.topic != ODOM_TOPIC:
                    continue
                m = typestore.deserialize_cdr(raw, conn.msgtype)
                t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                p = m.pose.pose.position
                q = m.pose.pose.orientation
                out.append((t, p.x, p.y, p.z, q.w, q.x, q.y, q.z))
        return np.asarray(out, dtype=np.float64)

    def iter_clouds(self):
        """Yield (t_s, Nx3 float32 xyz) for each /cloud_registered message."""
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        with self._read() as reader:
            for conn, _ts, raw in reader.messages():
                if conn.topic != CLOUD_TOPIC:
                    continue
                m = typestore.deserialize_cdr(raw, conn.msgtype)
                t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                if m.width == 0:
                    continue
                xyz = np.frombuffer(m.data, dtype=np.float32).reshape(m.width, m.point_step // 4)
                yield t, np.ascontiguousarray(xyz[:, :3])

    def read_images(
        self, cache: Path, topic: str = RGB_TOPIC, downscale: float = 1.0
    ) -> np.ndarray:
        """Camera frames as (M, H, W, 3) uint8 RGB, plus (M,) stamps; cached to .npz.

        `topic` matters more than it looks. `/rgb_img` is FAST-LIVO2's *debug
        visualization*: the VIO image with its projected lidar points drawn on
        top in green and blue. Those dots are burnt into the pixels, so every
        patch and mosaic built from it carries synthetic marks that the
        satellite basemap cannot possibly contain -- contamination in the one
        image the matcher is supposed to compare against the world.
        `/left_camera/image` is the raw sensor frame, clean and full
        resolution, and its header stamp is the capture time rather than the
        fused-frame time FAST-LIVO2 restamps its output with.

        `downscale` (typically `camera.scale`) shrinks each frame at read
        time, so the working resolution -- and the intrinsics that go with
        it -- stay what they were, while the pixels come from the clean topic.
        Downsampling a full-res frame also averages, which is slightly better
        than the VIO's own decimated image.
        """
        if cache.exists():
            with np.load(cache) as z:
                return z["stamps"], z["images"]
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        stamps, images = [], []
        with self._read() as reader:
            for conn, _ts, raw in reader.messages():
                if conn.topic != topic:
                    continue
                m = typestore.deserialize_cdr(raw, conn.msgtype)
                t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                img = decode_image_rgb(m.data, m.height, m.width, m.encoding)
                if downscale != 1.0:
                    img = _downscale_rgb(img, downscale)
                stamps.append(t)
                images.append(img)
        if not stamps:
            raise ValueError(f"no images on topic {topic!r} in {self.bag_dir}")
        stamps = np.asarray(stamps, dtype=np.float64)
        images = np.asarray(images, dtype=np.uint8)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, stamps=stamps, images=images)
        return stamps, images
