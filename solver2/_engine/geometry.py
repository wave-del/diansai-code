"""通用多边形几何工具。"""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray
from shapely.geometry import Point, Polygon


def as_array(vertices: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(vertices, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3 or not np.isfinite(arr).all():
        raise ValueError("多边形必须包含至少 3 个有限 Nx2 顶点")
    return arr


def signed_area(vertices: ArrayLike) -> float:
    pts = as_array(vertices)
    return float(0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))


def ensure_visual_clockwise(vertices: ArrayLike) -> NDArray[np.float64]:
    """y 向下时鞋带面积为正表示视觉顺时针。"""
    pts = as_array(vertices)
    return pts if signed_area(pts) > 0 else pts[::-1].copy()


def edge_lengths(vertices: ArrayLike) -> NDArray[np.float64]:
    pts = as_array(vertices)
    return np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)


def edge_angles(vertices: ArrayLike) -> NDArray[np.float64]:
    pts = as_array(vertices)
    d = np.roll(pts, -1, axis=0) - pts
    return np.degrees(np.arctan2(d[:, 1], d[:, 0]))


def internal_angles(vertices: ArrayLike) -> NDArray[np.float64]:
    pts = ensure_visual_clockwise(vertices)
    result: list[float] = []
    for i in range(len(pts)):
        a = pts[i - 1] - pts[i]
        b = pts[(i + 1) % len(pts)] - pts[i]
        cosine = float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))
        result.append(math.degrees(math.acos(cosine)))
    return np.asarray(result)


def clean_vertices(vertices: ArrayLike, tolerance: float = 1e-4) -> NDArray[np.float64]:
    pts = as_array(vertices)
    kept = [pts[0]]
    for point in pts[1:]:
        if np.linalg.norm(point - kept[-1]) > tolerance:
            kept.append(point)
    if len(kept) > 1 and np.linalg.norm(kept[0] - kept[-1]) <= tolerance:
        kept.pop()
    arr = as_array(kept)
    poly = Polygon(arr)
    if not poly.is_valid or poly.area <= tolerance:
        raise ValueError("多边形自相交或退化")
    return ensure_visual_clockwise(arr)


def polygon(vertices: ArrayLike) -> Polygon:
    return Polygon(as_array(vertices))


def centroid(vertices: ArrayLike) -> tuple[float, float]:
    c = polygon(vertices).centroid
    return float(c.x), float(c.y)


def max_inscribed_point(vertices: ArrayLike, tolerance: float = 0.25) -> tuple[tuple[float, float], float]:
    """使用 Shapely polylabel 求距边界较远的安全拾取点。"""
    poly = polygon(vertices)
    if not poly.is_valid or poly.area <= 0:
        raise ValueError("无法为无效多边形计算拾取点")
    try:
        from shapely.ops import polylabel

        p = polylabel(poly, tolerance=tolerance)
    except (ImportError, ValueError):
        p = poly.representative_point()
    clearance = float(p.distance(poly.boundary))
    return (float(p.x), float(p.y)), clearance


def pairwise_overlap(polygons: list[Polygon]) -> float:
    return float(
        sum(polygons[i].intersection(polygons[j]).area for i in range(len(polygons)) for j in range(i + 1, len(polygons)))
    )

