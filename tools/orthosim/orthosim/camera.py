"""Pinhole camera (OpenCV radtan) + small rotation helpers.

Self-contained (no dependency on the orthoproto T14/T15 prototype): OrthoSim
needs to render the *camera* side of a pair, so the same lens model is
reproduced here rather than imported from a task that hasn't shipped yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Pinhole:
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    def distort(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r2 = x * x + y * y
        radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        xd = x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
        yd = y * radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y
        return xd, yd

    def undistort(self, xd: np.ndarray, yd: np.ndarray, iterations: int = 12):
        x = np.array(xd, dtype=np.float64, copy=True)
        y = np.array(yd, dtype=np.float64, copy=True)
        for _ in range(iterations):
            xd_est, yd_est = self.distort(x, y)
            x += xd - xd_est
            y += yd - yd_est
        return x, y

    def world2cam(self, xyz_cam: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz_cam, dtype=np.float64)
        z = xyz[:, 2]
        xn = xyz[:, 0] / z
        yn = xyz[:, 1] / z
        xd, yd = self.distort(xn, yn)
        return np.column_stack([self.fx * xd + self.cx, self.fy * yd + self.cy])

    def cam2world(self, px: np.ndarray) -> np.ndarray:
        px = np.asarray(px, dtype=np.float64)
        xd = (px[:, 0] - self.cx) / self.fx
        yd = (px[:, 1] - self.cy) / self.fy
        xn, yn = self.undistort(xd, yd)
        out = np.column_stack([xn, yn, np.ones(len(px))])
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def quat_to_rotm(q: np.ndarray) -> np.ndarray:
    """(w, x, y, z) quaternion -> 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotz(theta: float) -> np.ndarray:
    """Rotation about the +Z (up) axis by ``theta`` -- heading in ENU."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def normalize_angle(a: float) -> float:
    """Wrap ``a`` to [-pi, pi]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)
