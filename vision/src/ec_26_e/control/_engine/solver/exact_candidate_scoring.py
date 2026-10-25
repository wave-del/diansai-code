"""候选完成后一次性执行的精确 Shapely 几何评分。"""

from __future__ import annotations

from typing import Iterable
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from .geometry_cache import bbox_overlap
from .output_quality import MINIMUM_OUTPUT_FILL_RATIO


def exact_geometry_metrics(
    transformed: Iterable[np.ndarray],
    width: float,
    height: float,
) -> dict[str, float | bool]:
    polygons = [Polygon(np.asarray(points, dtype=np.float64)) for points in transformed]
    target = box(0.0, 0.0, max(float(width), 1e-9), max(float(height), 1e-9))
    union = unary_union(polygons)
    overlap = 0.0
    for index, first in enumerate(polygons):
        for second in polygons[index + 1:]:
            if bbox_overlap(first.bounds, second.bounds):
                overlap += float(first.intersection(second).area)
    gap = float(target.difference(union).area)
    outside = float(union.difference(target).area)
    intersection = float(union.intersection(target).area)
    denominator = float(union.union(target).area)
    return {
        "overlap_area_mm2": overlap,
        "gap_area_mm2": gap,
        "outside_area_mm2": outside,
        "iou": intersection / denominator if denominator else 0.0,
        "fill_ratio": intersection / float(target.area) if target.area else 0.0,
        "target_area_mm2": float(target.area),
        "connected_exact": union.geom_type == "Polygon",
        "connected_with_tolerance_10mm": unary_union(
            [polygon.buffer(5.0) for polygon in polygons]
        ).geom_type == "Polygon",
        "rectangle_legal": (
            90.0 - 1e-6 <= width <= 120.0 + 1e-6
            and 50.0 - 1e-6 <= height <= 90.0 + 1e-6
        ),
    }


def exact_geometry_objective(metrics: dict[str, float | bool]) -> tuple[float, ...]:
    """稳定、门槛优先的精确几何排序键；越小越好。"""
    area = max(float(metrics["target_area_mm2"]), 1e-9)
    overlap_ratio = float(metrics["overlap_area_mm2"]) / area
    iou = float(metrics["iou"])
    return (
        0.0 if bool(metrics["rectangle_legal"]) else 1.0,
        0.0 if bool(metrics["connected_with_tolerance_10mm"]) else 1.0,
        max(0.0, 0.90 - iou),
        # Rank proposals inside the existing 2.5% geometry-repair envelope
        # before preferring tiny overlaps. Otherwise a 98% rectangle with 1.1%
        # temporary overlap loses to a 94% layout that can never pass output.
        # Clearance and final admission still require zero physical overlap.
        max(0.0, overlap_ratio - 0.025),
        max(0.0, MINIMUM_OUTPUT_FILL_RATIO - iou),
        # 1% 内的小重叠属于执行间隙规划可消除的修复层；在该层内先
        # 比较整体矩形 IoU，避免拒绝“缺口显著收敛、仅局部轻微穿插”
        # 的优化结果。最终执行门仍要求物理重叠为零。
        max(0.0, overlap_ratio - 0.01),
        -iou,
        overlap_ratio,
        (
            float(metrics["outside_area_mm2"])
            + float(metrics["gap_area_mm2"])
        ) / area,
    )
