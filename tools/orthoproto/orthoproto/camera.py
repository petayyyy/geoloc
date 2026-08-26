"""Pinhole camera with radial-tangential distortion (OpenCV-compatible).

The capture camera is modelled by FAST-LIVO2 as a vikit PinholeCamera with
coefficients d0..d3 = (k1, k2, p1, p2); the VIO image (`/rgb_img`) is the raw
(distorted) frame resized to the VIO size, so both projection directions must
go through the distortion model. Pure numpy -- cv2 is not a dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Pinhole:
    """Pinhole + radtan camera. (cx, cy) and (fx, fy) are in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    @classmethod
    def from_config(cls, cfg: dict, scale: float) -> Pinhole:
        """Build from a FAST-LIVO2 camera config block, scaled to the VIO size.

        Distortion coefficients are invariant to the pixel scale; only the
        intrinsic matrix entries scale.
        """
        return cls(
            fx=float(cfg["fx"]) * scale,
            fy=float(cfg["fy"]) * scale,
            cx=float(cfg["cx"]) * scale,
            cy=float(cfg["cy"]) * scale,
            k1=float(cfg.get("d0", 0.0)),
            k2=float(cfg.get("d1", 0.0)),
            p1=float(cfg.get("d2", 0.0)),
            p2=float(cfg.get("d3", 0.0)),
        )

    def distort(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply distortion to normalized image coordinates (OpenCV model)."""
        r2 = x * x + y * y
        radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        xd = x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
        yd = y * radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y
        return xd, yd

    def undistort(
        self, xd: np.ndarray, yd: np.ndarray, iterations: int = 12
    ) -> tuple[np.ndarray, np.ndarray]:
        """Invert the distortion model by fixed-point iteration.

        Standard approach: start from the distorted point and correct by the
        difference between the distorted value of the current guess and the
        target. Converges quickly for terrestrial lenses; 12 iterations is
        far below float64 noise.
        """
        x = np.array(xd, dtype=np.float64, copy=True)
        y = np.array(yd, dtype=np.float64, copy=True)
        for _ in range(iterations):
            xd_est, yd_est = self.distort(x, y)
            x += xd - xd_est
            y += yd - yd_est
        return x, y

    def world2cam(self, xyz_cam: np.ndarray) -> np.ndarray:
        """Project 3D camera-frame points (N,3) to (N,2) pixel coordinates."""
        xyz = np.asarray(xyz_cam, dtype=np.float64)
        z = xyz[:, 2]
        xn = xyz[:, 0] / z
        yn = xyz[:, 1] / z
        xd, yd = self.distort(xn, yn)
        return np.column_stack([self.fx * xd + self.cx, self.fy * yd + self.cy])

    def cam2world(self, px: np.ndarray) -> np.ndarray:
        """Unproject (N,2) pixels to unit direction vectors (N,3) in camera frame."""
        px = np.asarray(px, dtype=np.float64)
        xd = (px[:, 0] - self.cx) / self.fx
        yd = (px[:, 1] - self.cy) / self.fy
        xn, yn = self.undistort(xd, yd)
        out = np.column_stack([xn, yn, np.ones(len(px))])
        return out / np.linalg.norm(out, axis=1, keepdims=True)
