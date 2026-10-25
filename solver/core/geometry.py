from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin
from typing import Iterable

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon


EPS = 1e-8


@dataclass(frozen=True)
class Edge:
    piece_id: int
    index: int
    p0: np.ndarray
    p1: np.ndarray
    length: float


def identity() -> np.ndarray:
    return np.eye(3, dtype=float)


def transform_point(matrix: np.ndarray, point: Iterable[float]) -> np.ndarray:
    p = np.array([float(point[0]), float(point[1]), 1.0])
    return (matrix @ p)[:2]


def transform_polygon(poly: Polygon, matrix: np.ndarray) -> Polygon:
    return affinity.affine_transform(
        poly,
        [matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]],
    )


def rigid_from_segments(source0: np.ndarray, source1: np.ndarray, target0: np.ndarray, target1: np.ndarray) -> np.ndarray | None:
    """Return the rigid transform mapping the directed source segment to target."""
    sv = source1 - source0
    tv = target1 - target0
    sl = float(np.linalg.norm(sv))
    tl = float(np.linalg.norm(tv))
    if sl < EPS or tl < EPS:
        return None
    angle = atan2(tv[1], tv[0]) - atan2(sv[1], sv[0])
    c, s = cos(angle), sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    translation = target0 - rotation @ source0
    return np.array([[c, -s, translation[0]], [s, c, translation[1]], [0.0, 0.0, 1.0]])


def rotation_translation(angle_rad: float, tx: float, ty: float, center: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    c, s = cos(angle_rad), sin(angle_rad)
    cx, cy = center
    translation = np.array([tx, ty]) + np.array([cx, cy]) - np.array([[c, -s], [s, c]]) @ np.array([cx, cy])
    return np.array([[c, -s, translation[0]], [s, c, translation[1]], [0.0, 0.0, 1.0]])


def polygon_edges(poly: Polygon, piece_id: int) -> list[Edge]:
    coords = list(poly.exterior.coords)[:-1]
    result: list[Edge] = []
    for index, (a, b) in enumerate(zip(coords, coords[1:] + coords[:1])):
        p0 = np.asarray(a, dtype=float)
        p1 = np.asarray(b, dtype=float)
        result.append(Edge(piece_id, index, p0, p1, float(np.linalg.norm(p1 - p0))))
    return result


def matrix_angle(matrix: np.ndarray) -> float:
    return atan2(matrix[1, 0], matrix[0, 0])


def matrix_pose_error(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    delta = actual @ np.linalg.inv(expected)
    return float(np.linalg.norm(delta[:2, 2])), abs(matrix_angle(delta))


def polygon_from_coords(coords: list[list[float]] | list[tuple[float, float]]) -> Polygon:
    poly = Polygon(coords)
    if not poly.is_valid or poly.area <= EPS:
        raise ValueError("无效多边形")
    return poly
