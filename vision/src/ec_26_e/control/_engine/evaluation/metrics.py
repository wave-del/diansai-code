"""评估几何指标。"""

from __future__ import annotations

from itertools import permutations
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


def polygon_iou(first: Polygon, second: Polygon) -> float:
    union = first.union(second).area
    return float(first.intersection(second).area / union) if union else 0.0


def best_correspondence(solved: dict[str, Polygon], expected: dict[str, Polygon]) -> tuple[dict[str, str], float]:
    solved_ids, expected_ids = list(solved), list(expected)
    best_map: dict[str, str] = {}
    best_score = -1.0
    for perm in permutations(expected_ids):
        score = sum(polygon_iou(solved[sid], expected[eid]) for sid, eid in zip(solved_ids, perm))
        if score > best_score:
            best_score = score
            best_map = dict(zip(solved_ids, perm))
    return best_map, float(best_score / max(len(solved_ids), 1))


def _connected_with_tolerance(polygons: list[Polygon], tolerance_mm: float) -> bool:
    buffered = [poly.buffer(tolerance_mm / 2.0) for poly in polygons]
    return unary_union(buffered).geom_type == "Polygon"


def _numerically_exact_connected(polygons: list[Polygon], epsilon_mm: float = 1e-7) -> bool:
    """忽略刚体浮点运算产生的亚微米级缝隙，做连通图遍历。"""
    if not polygons:
        return False
    reached = {0}
    pending = [0]
    while pending:
        current = pending.pop()
        for index, other in enumerate(polygons):
            if index not in reached and polygons[current].distance(other) <= epsilon_mm:
                reached.add(index)
                pending.append(index)
    return len(reached) == len(polygons)


def union_metrics(polygons: list[Polygon], target: Polygon) -> dict:
    """在 solution 声明的同一 A4 目标矩形中计算完整拼图指标。"""
    union = unary_union(polygons)
    minx, miny, maxx, maxy = union.bounds
    target_minx, target_miny, target_maxx, target_maxy = target.bounds
    width, height = target_maxx - target_minx, target_maxy - target_miny
    overlap = sum(
        polygons[i].intersection(polygons[j]).area
        for i in range(len(polygons))
        for j in range(i + 1, len(polygons))
    )
    return {
        "union_rectangle_iou": float(union.intersection(target).area / union.union(target).area),
        "overlap_area_mm2": float(overlap),
        "gap_area_mm2": float(target.difference(union).area),
        "rectangle_fill_ratio": float(union.intersection(target).area / target.area),
        "final_width_error_mm": float(abs((maxx - minx) - width)),
        "final_height_error_mm": float(abs((maxy - miny) - height)),
        "exact_connected": _numerically_exact_connected(polygons),
        "connected_with_tolerance_2mm": _connected_with_tolerance(polygons, 2.0),
        "connected_with_tolerance_5mm": _connected_with_tolerance(polygons, 5.0),
        "connected_with_tolerance_10mm": _connected_with_tolerance(polygons, 10.0),
    }
