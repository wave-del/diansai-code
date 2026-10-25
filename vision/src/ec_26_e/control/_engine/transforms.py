"""二维刚体变换；在 y 向下坐标系中正角度为视觉顺时针。"""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray


def normalize_angle(degrees: float) -> float:
    """将角度归一化到 [-180, 180)。"""
    return float((degrees + 180.0) % 360.0 - 180.0)


def rotation_matrix(degrees_clockwise: float) -> NDArray[np.float64]:
    radians = math.radians(degrees_clockwise)
    c, s = math.cos(radians), math.sin(radians)
    return np.array([[c, -s], [s, c]], dtype=float)


def transform_points(
    points: ArrayLike,
    rotation_deg_clockwise: float,
    translation: ArrayLike = (0.0, 0.0),
) -> NDArray[np.float64]:
    pts = np.asarray(points, dtype=float)
    return pts @ rotation_matrix(rotation_deg_clockwise).T + np.asarray(translation, dtype=float)


def estimate_rigid(source: ArrayLike, target: ArrayLike) -> tuple[float, NDArray[np.float64], float]:
    """Kabsch 估计仅含旋转和平移的变换，明确禁止反射。"""
    src = np.asarray(source, dtype=float)
    dst = np.asarray(target, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("source 与 target 必须是相同形状的 Nx2 坐标")
    src_c, dst_c = src.mean(axis=0), dst.mean(axis=0)
    h = (src - src_c).T @ (dst - dst_c)
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    theta = normalize_angle(math.degrees(math.atan2(r[1, 0], r[0, 0])))
    translation = dst_c - src_c @ r.T
    fitted = src @ r.T + translation
    rms = float(np.sqrt(np.mean(np.sum((fitted - dst) ** 2, axis=1))))
    return theta, translation, rms


def compose_transform(
    first_angle: float,
    first_translation: ArrayLike,
    second_angle: float,
    second_translation: ArrayLike,
) -> tuple[float, NDArray[np.float64]]:
    """返回先执行 first、再执行 second 的合成刚体变换。"""
    angle = normalize_angle(first_angle + second_angle)
    trans = (
        np.asarray(first_translation, dtype=float) @ rotation_matrix(second_angle).T
        + np.asarray(second_translation, dtype=float)
    )
    return angle, trans

