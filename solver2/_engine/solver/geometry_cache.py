"""不可变位姿几何缓存和安全的 NumPy 前置筛选。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from shapely.geometry import Polygon


@dataclass(frozen=True)
class CachedPoseGeometry:
    vertices: np.ndarray
    polygon: Polygon
    bounds: tuple[float, float, float, float]
    area: float
    centroid: tuple[float, float]
    convex_hull: Polygon
    hull_points: np.ndarray
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    bounding_radius: float

    @classmethod
    def from_vertices(cls, vertices: np.ndarray) -> "CachedPoseGeometry":
        points = np.array(vertices, dtype=np.float64, copy=True)
        points.setflags(write=False)
        polygon = Polygon(points)
        hull = polygon.convex_hull
        hull_points = np.asarray(hull.exterior.coords[:-1], dtype=np.float64).copy()
        hull_points.setflags(write=False)
        min_x, min_y, max_x, max_y = map(float, polygon.bounds)
        centroid = (float(polygon.centroid.x), float(polygon.centroid.y))
        radius = float(
            np.linalg.norm(points - np.asarray(centroid), axis=1).max(initial=0.0)
        )
        return cls(
            vertices=points, polygon=polygon,
            bounds=(min_x, min_y, max_x, max_y),
            area=float(polygon.area), centroid=centroid,
            convex_hull=hull, hull_points=hull_points,
            min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y,
            bounding_radius=radius,
        )


def bbox_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    tolerance: float = 0.0,
) -> bool:
    return not (
        first[2] <= second[0] + tolerance
        or second[2] <= first[0] + tolerance
        or first[3] <= second[1] + tolerance
        or second[3] <= first[1] + tolerance
    )


def bbox_distance(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    dx = max(first[0] - second[2], second[0] - first[2], 0.0)
    dy = max(first[1] - second[3], second[1] - first[3], 0.0)
    return float(math.hypot(dx, dy))


def projected_extent(points: np.ndarray, direction: np.ndarray) -> tuple[float, float]:
    unit = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(unit))
    if norm <= 1e-15:
        raise ValueError("projection direction 不能为零")
    values = np.asarray(points, dtype=np.float64) @ (unit / norm)
    return float(values.min()), float(values.max())


def diameter_bound(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(
        max(
            np.linalg.norm(pts[index + 1:] - pts[index], axis=1).max(initial=0.0)
            for index in range(len(pts) - 1)
        )
    )


def area_lower_bound(total_area: float, accumulated_overlap: float) -> float:
    return max(0.0, float(total_area) - float(accumulated_overlap))


def interval_conflict(
    occupied: Iterable[tuple[float, float]],
    candidate: tuple[float, float],
    tolerance: float = 1e-9,
) -> bool:
    start, end = sorted(candidate)
    return any(
        min(end, max(a, b)) - max(start, min(a, b)) > tolerance
        for a, b in occupied
    )


def orientation_side_test(
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    first_point: np.ndarray,
    second_point: np.ndarray,
    tolerance: float = 1e-9,
) -> bool:
    direction = np.asarray(edge_end) - np.asarray(edge_start)
    a = float(np.cross(direction, np.asarray(first_point) - edge_start))
    b = float(np.cross(direction, np.asarray(second_point) - edge_start))
    return a * b < -(tolerance * tolerance)


def pairwise_overlap_cached(
    candidate: CachedPoseGeometry,
    existing: Iterable[CachedPoseGeometry],
) -> float:
    overlap = 0.0
    for other in existing:
        if bbox_overlap(candidate.bounds, other.bounds):
            overlap += float(candidate.polygon.intersection(other.polygon).area)
    return overlap
