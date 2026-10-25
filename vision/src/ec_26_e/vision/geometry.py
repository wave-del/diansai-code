"""提供统一视觉检测器复用的多边形和坐标几何基础函数。"""

from __future__ import annotations

import cv2
import numpy as np


def polygon_edge_lengths(points: np.ndarray) -> np.ndarray:
    """返回按轮廓顺序排列的各条边长。"""

    vertices = np.asarray(points, dtype=np.float64)
    return np.linalg.norm(vertices - np.roll(vertices, -1, axis=0), axis=1)


def polygon_turn_angles_deg(points: np.ndarray) -> np.ndarray:
    """返回各顶点带符号转角；规范顺时针轮廓中的凹角为负。"""

    vertices = np.asarray(points, dtype=np.float64)
    incoming = vertices - np.roll(vertices, 1, axis=0)
    outgoing = np.roll(vertices, -1, axis=0) - vertices
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    dot = np.sum(incoming * outgoing, axis=1)
    return np.degrees(np.arctan2(cross, dot))


def polygon_corner_depths(points: np.ndarray) -> np.ndarray:
    """计算每个顶点到相邻两顶点连线的垂直距离。"""

    vertices = np.asarray(points, dtype=np.float64)
    previous = np.roll(vertices, 1, axis=0)
    following = np.roll(vertices, -1, axis=0)
    chords = following - previous
    relative = vertices - previous
    cross = np.abs(
        chords[:, 0] * relative[:, 1]
        - chords[:, 1] * relative[:, 0]
    )
    return cross / np.maximum(np.linalg.norm(chords, axis=1), 1e-9)


def _orientation(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
    """返回三个二维点的有向叉积。"""

    left = second - first
    right = third - first
    return float(left[0] * right[1] - left[1] * right[0])


def _point_on_segment(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    epsilon: float,
) -> bool:
    """判断共线点是否位于闭线段包围盒内。"""

    return bool(
        np.all(point >= np.minimum(start, end) - epsilon)
        and np.all(point <= np.maximum(start, end) + epsilon)
    )


def _segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    epsilon: float,
) -> bool:
    """判断两条闭线段是否相交或共线重叠。"""

    first_a = _orientation(first_start, first_end, second_start)
    first_b = _orientation(first_start, first_end, second_end)
    second_a = _orientation(second_start, second_end, first_start)
    second_b = _orientation(second_start, second_end, first_end)
    if (
        (
            (first_a > epsilon and first_b < -epsilon)
            or (first_a < -epsilon and first_b > epsilon)
        )
        and (
            (second_a > epsilon and second_b < -epsilon)
            or (second_a < -epsilon and second_b > epsilon)
        )
    ):
        return True
    collinear_cases = (
        (first_a, second_start, first_start, first_end),
        (first_b, second_end, first_start, first_end),
        (second_a, first_start, second_start, second_end),
        (second_b, first_end, second_start, second_end),
    )
    return any(
        abs(value) <= epsilon
        and _point_on_segment(point, start, end, epsilon)
        for value, point, start, end in collinear_cases
    )


def is_simple_polygon(points: np.ndarray, epsilon: float = 1e-6) -> bool:
    """校验有限、非退化且没有非相邻边相交的简单多边形。"""

    vertices = np.asarray(points, dtype=np.float64)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (2,)
        or len(vertices) < 3
        or not np.all(np.isfinite(vertices))
        or abs(signed_polygon_area(vertices)) <= epsilon
        or np.any(polygon_edge_lengths(vertices) <= epsilon)
    ):
        return False
    count = len(vertices)
    for first in range(count):
        first_next = (first + 1) % count
        for second in range(first + 1, count):
            second_next = (second + 1) % count
            if (
                first == second
                or first_next == second
                or second_next == first
            ):
                continue
            if _segments_intersect(
                vertices[first],
                vertices[first_next],
                vertices[second],
                vertices[second_next],
                epsilon,
            ):
                return False
    return True


def signed_polygon_area(points: np.ndarray) -> float:
    """计算图像坐标多边形的有向面积。"""

    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(
        np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
    )


def canonicalize_corners(points: np.ndarray) -> np.ndarray:
    """固定角点为顺时针，并从最靠上、同高最靠左的点开始。"""

    ordered = points.astype(np.float32)
    if signed_polygon_area(ordered) < 0:
        ordered = ordered[::-1]
    start = min(
        range(len(ordered)),
        key=lambda index: (
            float(ordered[index, 1]),
            float(ordered[index, 0]),
        ),
    )
    return np.roll(ordered, -start, axis=0)


def polygon_iou(contour: np.ndarray, polygon: np.ndarray) -> float:
    """计算原始轮廓与拟合多边形的像素级交并比。"""

    # Include both operands. Cropping to the original contour alone silently
    # discarded fitted vertices outside it and could report IoU=1 for a much
    # larger polygon, defeating the existing quality gate.
    contour_points = np.rint(contour.reshape(-1, 2)).astype(np.int32)
    polygon_points = np.rint(polygon.reshape(-1, 2)).astype(np.int32)
    x, y, width, height = cv2.boundingRect(np.vstack((contour_points, polygon_points)))
    if width <= 0 or height <= 0:
        return 0.0
    offset = np.array([x, y], dtype=np.int32)
    contour_mask = np.zeros((height, width), dtype=np.uint8)
    polygon_mask = np.zeros_like(contour_mask)
    cv2.drawContours(
        contour_mask,
        [(contour_points - offset).reshape(-1, 1, 2)],
        -1,
        255,
        -1,
    )
    cv2.fillPoly(
        polygon_mask,
        [
            np.rint(polygon - offset)
            .astype(np.int32)
            .reshape(-1, 1, 2)
        ],
        255,
    )
    intersection = cv2.countNonZero(
        cv2.bitwise_and(contour_mask, polygon_mask)
    )
    union = cv2.countNonZero(cv2.bitwise_or(contour_mask, polygon_mask))
    return float(intersection / union) if union else 0.0


def relative_point(
    point: tuple[float, float],
    width: int,
    height: int,
) -> tuple[float, float]:
    """把校正图像像素坐标转换为闭区间相对坐标。"""

    x, y = point
    return (
        round(float(np.clip(x / (width - 1), 0.0, 1.0)), 6),
        round(float(np.clip(y / (height - 1), 0.0, 1.0)), 6),
    )
