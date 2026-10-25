"""生成案例的统一强约束验证器。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from ..config import (
    BOUNDARY_LENGTH_TOLERANCE_MM,
    GEOMETRY_TOLERANCE_MM,
)
from ..geometry import edge_lengths, internal_angles, pairwise_overlap, polygon
from ..models import Piece


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    area_sum_mm2: float
    union_area_mm2: float
    overlap_area_mm2: float


def rectangle_sides(
    width_mm: float,
    height_mm: float,
) -> dict[str, LineString]:
    """返回四条独立矩形外边，不能把不同边上的点接触相加。"""
    return {
        "top": LineString([(0.0, 0.0), (width_mm, 0.0)]),
        "bottom": LineString(
            [(0.0, height_mm), (width_mm, height_mm)]
        ),
        "left": LineString([(0.0, 0.0), (0.0, height_mm)]),
        "right": LineString(
            [(width_mm, 0.0), (width_mm, height_mm)]
        ),
    }


def _same_side_edge_projection_lengths(
    vertices: list[tuple[float, float]],
    width_mm: float,
    height_mm: float,
    coordinate_tolerance_mm: float,
) -> dict[str, float]:
    """只接受两个端点都位于同一外边附近的多边形边，不使用 buffer。"""
    result = {name: 0.0 for name in ("top", "bottom", "left", "right")}
    points = np.asarray(vertices, dtype=float)
    for index, first in enumerate(points):
        second = points[(index + 1) % len(points)]
        side_tests = {
            "top": (
                abs(first[1]) <= coordinate_tolerance_mm
                and abs(second[1]) <= coordinate_tolerance_mm
            ),
            "bottom": (
                abs(first[1] - height_mm) <= coordinate_tolerance_mm
                and abs(second[1] - height_mm) <= coordinate_tolerance_mm
            ),
            "left": (
                abs(first[0]) <= coordinate_tolerance_mm
                and abs(second[0]) <= coordinate_tolerance_mm
            ),
            "right": (
                abs(first[0] - width_mm) <= coordinate_tolerance_mm
                and abs(second[0] - width_mm) <= coordinate_tolerance_mm
            ),
        }
        horizontal_projection = max(
            0.0,
            min(max(first[0], second[0]), width_mm)
            - max(min(first[0], second[0]), 0.0),
        )
        vertical_projection = max(
            0.0,
            min(max(first[1], second[1]), height_mm)
            - max(min(first[1], second[1]), 0.0),
        )
        for name, matches in side_tests.items():
            if matches:
                projection = (
                    horizontal_projection
                    if name in {"top", "bottom"}
                    else vertical_projection
                )
                result[name] = max(result[name], projection)
    return result


def outer_boundary_overlap_lengths(
    poly: Polygon,
    width_mm: float,
    height_mm: float,
    *,
    coordinate_tolerance_mm: float = GEOMETRY_TOLERANCE_MM,
) -> dict[str, float]:
    """计算碎片边界与上、下、左、右四边各自的共线重合长度。"""
    direct = {
        name: float(poly.boundary.intersection(side).length)
        for name, side in rectangle_sides(width_mm, height_mm).items()
    }
    vertices = [
        (float(x), float(y))
        for x, y in list(poly.exterior.coords)[:-1]
    ]
    endpoint_fallback = _same_side_edge_projection_lengths(
        vertices,
        width_mm,
        height_mm,
        coordinate_tolerance_mm,
    )
    return {
        name: max(direct[name], endpoint_fallback[name])
        for name in direct
    }


def has_outer_boundary_edge(
    poly: Polygon,
    width_mm: float,
    height_mm: float,
    *,
    boundary_length_tolerance_mm: float = BOUNDARY_LENGTH_TOLERANCE_MM,
    coordinate_tolerance_mm: float = GEOMETRY_TOLERANCE_MM,
) -> bool:
    vertices = [
        (float(x), float(y))
        for x, y in list(poly.exterior.coords)[:-1]
    ]
    endpoint_lengths = _same_side_edge_projection_lengths(
        vertices,
        width_mm,
        height_mm,
        coordinate_tolerance_mm,
    )
    # 对直边多边形，这是正长度共线重合的充要判断：一条多边形边
    # 的两个端点必须同时落在同一矩形边所在直线上，且投影长度为正。
    return (
        max(endpoint_lengths.values(), default=0.0)
        > boundary_length_tolerance_mm
    )


def check_partition(
    pieces: list[Piece],
    width_mm: float,
    height_mm: float,
    min_edge_mm: float = 20.0,
    min_area_mm2: float = 400.0,
    min_interior_angle_deg: float = 0.0,
    max_aspect_ratio: float = float("inf"),
    collinear_merge_tolerance_deg: float = 0.25,
) -> ValidationResult:
    from .general_partition_generator import merge_collinear_vertices

    errors: list[str] = []
    if not 2 <= len(pieces) <= 4:
        errors.append("碎片数量不在 2～4")
    if not 90.0 - 1e-6 <= width_mm <= 120.0 + 1e-6:
        errors.append("目标矩形宽度不在 90～120 mm")
    if not 50.0 - 1e-6 <= height_mm <= 90.0 + 1e-6:
        errors.append("目标矩形高度不在 50～90 mm")

    polygons = []
    for piece in pieces:
        try:
            cleaned = merge_collinear_vertices(
                piece.vertices, collinear_merge_tolerance_deg
            )
            poly = polygon(cleaned)
        except (TypeError, ValueError) as exc:
            errors.append(f"{piece.piece_id} 顶点无效: {exc}")
            continue
        polygons.append(poly)
        if not poly.is_valid or poly.area <= 1e-8:
            errors.append(f"{piece.piece_id} 多边形无效或退化")
        if poly.interiors:
            errors.append(f"{piece.piece_id} 存在内部空洞")
        if len(cleaned) > 5:
            errors.append(f"{piece.piece_id} 边数超过 5")
        if poly.area < min_area_mm2 - 1e-6:
            errors.append(f"{piece.piece_id} 面积小于 {min_area_mm2}")
        lengths = edge_lengths(cleaned)
        if np.min(lengths) < min_edge_mm - 1e-6:
            errors.append(f"{piece.piece_id} 存在短于 {min_edge_mm} mm 的边")
        if not has_outer_boundary_edge(poly, width_mm, height_mm):
            errors.append(f"{piece.piece_id} 没有与矩形外边重合的正长度边")
        if (
            min_interior_angle_deg > 0
            and np.min(internal_angles(cleaned)) < min_interior_angle_deg - 1e-6
        ):
            errors.append(
                f"{piece.piece_id} 最小内角小于 {min_interior_angle_deg} 度"
            )
        rectangle = poly.minimum_rotated_rectangle
        coords = np.asarray(rectangle.exterior.coords[:4], dtype=float)
        sides = np.linalg.norm(np.roll(coords, -1, axis=0) - coords, axis=1)
        positive = sides[sides > 1e-8]
        aspect = (
            float(positive.max() / positive.min())
            if len(positive)
            else float("inf")
        )
        if aspect > max_aspect_ratio + 1e-6:
            errors.append(f"{piece.piece_id} 长宽比超过 {max_aspect_ratio}")

    target = box(0.0, 0.0, width_mm, height_mm)
    union = unary_union(polygons) if polygons else box(0, 0, 0, 0)
    overlap = pairwise_overlap(polygons)
    area_sum = float(sum(poly.area for poly in polygons))
    if overlap > 1e-5:
        errors.append("碎片存在面积重叠")
    if union.symmetric_difference(target).area > 1e-5:
        errors.append("碎片并集不等于目标矩形")
    return ValidationResult(
        not errors, errors, area_sum, float(union.area), overlap
    )


def validate_partition(
    pieces: list[Piece],
    width_mm: float,
    height_mm: float,
    min_edge_mm: float = 20.0,
    min_area_mm2: float = 400.0,
    min_interior_angle_deg: float = 0.0,
    max_aspect_ratio: float = float("inf"),
    collinear_merge_tolerance_deg: float = 0.25,
) -> ValidationResult:
    result = check_partition(
        pieces,
        width_mm,
        height_mm,
        min_edge_mm,
        min_area_mm2,
        min_interior_angle_deg,
        max_aspect_ratio,
        collinear_merge_tolerance_deg,
    )
    if not result.valid:
        raise ValueError("拼图约束失败：" + "；".join(result.errors))
    return result
