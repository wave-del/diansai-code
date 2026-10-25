"""基于完整重建拓扑的随机拼图联合留缝规划。"""

from __future__ import annotations

from itertools import combinations
import json
import logging
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from shapely.geometry import Polygon

from ..config import A4_HEIGHT_MM, A4_WIDTH_MM, DIVIDER_Y_MM
from ..planning.clearance_planner import (
    ClearancePlan,
    InterfaceClearanceMetrics,
    pairwise_positive_overlap_area,
)
from .random_safety import ReconstructedInterface, ReconstructedTopology
from ec_26_e.control.candidate_acceleration import scan_candidate_distances, cannot_improve


def _workspace_shift(arrays: list[np.ndarray], deltas: np.ndarray) -> np.ndarray | None:
    """用不改变相对拼缝的整体平移把结果放进A4下半区。"""

    points = np.vstack([
        vertices + delta for vertices, delta in zip(arrays, deltas)
    ])
    width = float(points[:, 0].max() - points[:, 0].min())
    height = float(points[:, 1].max() - points[:, 1].min())
    if width > A4_WIDTH_MM + 1e-7 or height > A4_HEIGHT_MM - DIVIDER_Y_MM + 1e-7:
        return None
    shift = np.zeros(2, dtype=float)
    if points[:, 0].min() < 0.0:
        shift[0] = -float(points[:, 0].min())
    if points[:, 0].max() + shift[0] > A4_WIDTH_MM:
        shift[0] += A4_WIDTH_MM - float(points[:, 0].max() + shift[0])
    if points[:, 1].min() < DIVIDER_Y_MM:
        shift[1] = DIVIDER_Y_MM - float(points[:, 1].min())
    if points[:, 1].max() + shift[1] > A4_HEIGHT_MM:
        shift[1] += A4_HEIGHT_MM - float(points[:, 1].max() + shift[1])
    shifted = deltas + shift
    shifted_points = np.vstack([
        vertices + delta for vertices, delta in zip(arrays, shifted)
    ])
    if (
        shifted_points[:, 0].min() < -1e-7
        or shifted_points[:, 0].max() > A4_WIDTH_MM + 1e-7
        or shifted_points[:, 1].min() < DIVIDER_Y_MM - 1e-7
        or shifted_points[:, 1].max() > A4_HEIGHT_MM + 1e-7
    ):
        return None
    return shifted


def _minimum_norm_interface_solution(
    topology: ReconstructedTopology,
    index_by_id: dict[str, int],
    piece_count: int,
    target_gap_mm: float,
) -> np.ndarray:
    """同时满足全部拼缝法向约束并选择总位移最小的平移解。"""

    matrix: list[np.ndarray] = []
    values: list[float] = []
    weights: list[float] = []
    for interface in topology.interfaces:
        first = index_by_id[interface.piece_a]
        second = index_by_id[interface.piece_b]
        normal = np.asarray(interface.normal_a_to_b, dtype=float)
        tangent = np.asarray([-normal[1], normal[0]], dtype=float)
        normal_row = np.zeros(2 * piece_count, dtype=float)
        normal_row[2 * first:2 * first + 2] = -normal
        normal_row[2 * second:2 * second + 2] = normal
        matrix.append(normal_row)
        values.append(float(target_gap_mm + interface.support_line_distance_mm))
        weights.append(max(1.0, interface.length_mm))
        tangent_row = np.zeros(2 * piece_count, dtype=float)
        tangent_row[2 * first:2 * first + 2] = -tangent
        tangent_row[2 * second:2 * second + 2] = tangent
        matrix.append(tangent_row)
        values.append(0.0)
        weights.append(max(0.2, 0.1 * interface.length_mm))
    # 固定整体平移零空间，不偏袒任何一片作为锚点。
    for axis in range(2):
        row = np.zeros(2 * piece_count, dtype=float)
        row[axis::2] = 1.0
        matrix.append(row)
        values.append(0.0)
        weights.append(1000.0)
    if not matrix:
        return np.zeros((piece_count, 2), dtype=float)
    weight = np.sqrt(np.asarray(weights, dtype=float))
    lhs = np.asarray(matrix, dtype=float) * weight[:, None]
    rhs = np.asarray(values, dtype=float) * weight
    return np.linalg.lstsq(lhs, rhs, rcond=None)[0].reshape(piece_count, 2)


def _radial_solution(arrays: list[np.ndarray], scale: float) -> np.ndarray:
    """按碎片质心相对整体中心做统一膨胀，稳定处理带环和T形拓扑。"""

    centroids = np.asarray([
        [Polygon(vertices).centroid.x, Polygon(vertices).centroid.y]
        for vertices in arrays
    ])
    center = np.mean(centroids, axis=0)
    return float(scale) * (centroids - center)


def _interface_metrics(
    polygons: list[Polygon],
    deltas: np.ndarray,
    topology: ReconstructedTopology,
    index_by_id: dict[str, int],
    target_gap_mm: float,
) -> tuple[tuple[InterfaceClearanceMetrics, ...], float]:
    """按重建原子拼缝计算真实间隙和对应端点距离。"""

    metrics: list[InterfaceClearanceMetrics] = []
    maximum_endpoint_distance = 0.0
    for interface in topology.interfaces:
        first = index_by_id[interface.piece_a]
        second = index_by_id[interface.piece_b]
        normal = np.asarray(interface.normal_a_to_b, dtype=float)
        tangent = np.asarray([-normal[1], normal[0]], dtype=float)
        relative = deltas[second] - deltas[first]
        gap = float(polygons[first].distance(polygons[second]))
        endpoint_distance = float(np.hypot(
            interface.support_line_distance_mm + np.dot(relative, normal),
            np.dot(relative, tangent),
        ))
        maximum_endpoint_distance = max(
            maximum_endpoint_distance, endpoint_distance
        )
        metrics.append(InterfaceClearanceMetrics(
            interface_id=interface.interface_id,
            piece_a=interface.piece_a,
            piece_b=interface.piece_b,
            nominal_normal_gap_mm=gap,
            tangential_misalignment_mm=float(abs(np.dot(relative, tangent))),
            target_gap_mm=float(target_gap_mm),
            estimated_intrusion_risk_mm=max(0.0, target_gap_mm - gap),
            endpoint_distance_max_mm=endpoint_distance,
            compactable=gap > target_gap_mm + 1e-3,
        ))
    return tuple(metrics), float(maximum_endpoint_distance)


def _buffer_overlap(
    polygons: list[Polygon],
    pairs: set[tuple[int, int]],
    radius: float,
) -> float:
    """计算指定碎片对的不确定性缓冲交叠面积。"""

    buffered = [polygon.buffer(radius) for polygon in polygons]
    return float(sum(
        buffered[first].intersection(buffered[second]).area
        for first, second in pairs
    ))


def _refine_interface_solution(
    arrays: list[np.ndarray], topology: ReconstructedTopology,
    index_by_id: dict[str, int], initial: np.ndarray,
    target_gap_mm: float, minimum_gap_mm: float, maximum_gap_mm: float,
    maximum_vertex_distance_mm: float,
) -> np.ndarray:
    """Refine every piece translation using actual polygon distances.

    Scaling one linear solution or expanding all centroids cannot independently
    open several T-junction seams. Prefer compact translations within the gap
    interval, rather than forcing every seam to exactly 5 mm at a T junction.
    This is another proposal; the same exact
    overlap, gap, workspace and endpoint checks still decide feasibility.
    """
    def residual(flat):
        deltas = flat.reshape(-1, 2)
        deltas = deltas - deltas.mean(axis=0)
        polygons = [Polygon(vertices + delta) for vertices, delta in zip(arrays, deltas)]
        metrics, endpoint_distance = _interface_metrics(
            polygons, deltas, topology, index_by_id, target_gap_mm,
        )
        overlap = pairwise_positive_overlap_area(polygons)
        gaps = np.asarray([metric.nominal_normal_gap_mm for metric in metrics])
        interior = min(0.05, (maximum_gap_mm - minimum_gap_mm) * 0.01)
        return np.r_[
            10.0 * np.maximum(0.0, minimum_gap_mm + interior - gaps),
            10.0 * np.maximum(0.0, gaps - maximum_gap_mm + interior),
            0.1 * (gaps - target_gap_mm),
            4.0 * np.sqrt(max(0.0, overlap)),
            4.0 * max(0.0, endpoint_distance - maximum_vertex_distance_mm),
            0.1 * deltas.ravel(),
        ]

    centered = initial - initial.mean(axis=0)
    fitted = least_squares(residual, centered.ravel(), max_nfev=80,
                           ftol=1e-7, xtol=1e-7, gtol=1e-7)
    deltas = fitted.x.reshape(-1, 2)
    return deltas - deltas.mean(axis=0)


def plan_random_clearance(
    vertices_by_id: dict[str, np.ndarray],
    topology: ReconstructedTopology,
    *,
    target_gap_mm: float,
    minimum_gap_mm: float,
    maximum_gap_mm: float,
    maximum_vertex_distance_mm: float,
    measurement_error_radius_mm: float,
    machine_position_margin_mm: float,
    overlap_tolerance_mm2: float,
) -> ClearancePlan:
    """使用完整邻接图联合规划全部碎片的纯平移执行位置。"""

    piece_ids = list(vertices_by_id)
    arrays = [np.asarray(vertices_by_id[piece_id], dtype=float) for piece_id in piece_ids]
    index_by_id = {piece_id: index for index, piece_id in enumerate(piece_ids)}
    adjacency = {
        tuple(sorted((index_by_id[first], index_by_id[second])))
        for first, second in topology.adjacency_pairs
    }
    all_pairs = set(combinations(range(len(arrays)), 2))
    nonadjacent = all_pairs - adjacency
    least_squares = _minimum_norm_interface_solution(
        topology, index_by_id, len(arrays), target_gap_mm
    )
    relative_centroid_span = max(
        (
            float(np.linalg.norm(
                np.mean(arrays[first], axis=0)
                - np.mean(arrays[second], axis=0)
            ))
            for first, second in adjacency
        ),
        default=1.0,
    )
    maximum_radial_scale = min(
        1.0, maximum_vertex_distance_mm / max(relative_centroid_span, 1e-6)
    )
    candidates: list[np.ndarray] = []
    # 线性拼缝解处理普通链式拓扑；统一质心膨胀处理带环和多向结点。
    for factor in np.linspace(0.65, 1.35, 29):
        candidates.append(least_squares * float(factor))
    radial = _radial_solution(arrays, 1.0)
    for scale in np.linspace(0.0, maximum_radial_scale, 161):
        candidates.append(float(scale) * radial)

    scan_started = perf_counter()
    distances = scan_candidate_distances(arrays, candidates)
    scan_seconds = perf_counter() - scan_started
    adjacency_indices = [i for i, pair in enumerate(combinations(range(len(arrays)), 2)) if pair in adjacency]
    pruned = 0

    def evaluate_translation(raw_deltas):
        deltas = _workspace_shift(arrays, raw_deltas)
        if deltas is None:
            return None
        polygons = [
            Polygon(vertices + delta) for vertices, delta in zip(arrays, deltas)
        ]
        overlap = pairwise_positive_overlap_area(polygons)
        metrics, endpoint_distance = _interface_metrics(
            polygons, deltas, topology, index_by_id, target_gap_mm,
        )
        gaps = [metric.nominal_normal_gap_mm for metric in metrics]
        gap_violation = sum(
            max(0.0, minimum_gap_mm - gap) ** 2
            + max(0.0, gap - maximum_gap_mm) ** 2
            for gap in gaps
        )
        score = (
            0.0 if overlap <= overlap_tolerance_mm2 else 1.0,
            0.0 if endpoint_distance <= maximum_vertex_distance_mm + 1e-7 else 1.0,
            float(gap_violation),
            float(sum((gap - target_gap_mm) ** 2 for gap in gaps)),
            float(overlap),
            float(np.sum(np.linalg.norm(raw_deltas, axis=1))),
        )
        return score, deltas, polygons, metrics, endpoint_distance

    best = None
    for candidate_index, raw_deltas in enumerate(candidates):
        if distances is not None and cannot_improve(distances[candidate_index], adjacency_indices,
                best[0] if best else None, minimum_gap_mm, maximum_gap_mm, target_gap_mm):
            pruned += 1
            continue
        evaluated = evaluate_translation(raw_deltas)
        if evaluated is not None and (best is None or evaluated[0] < best[0]):
            best = evaluated

    if (best is not None and topology.graph_connected and topology.interfaces
            and (best[0][0] or best[0][1] or best[0][2] > 1e-12)):
        refined = _refine_interface_solution(
            arrays, topology, index_by_id, best[1],
            target_gap_mm, minimum_gap_mm, maximum_gap_mm, maximum_vertex_distance_mm,
        )
        evaluated = evaluate_translation(refined)
        if evaluated is not None and evaluated[0] < best[0]:
            best = evaluated

    if best is None:
        zero = np.zeros((len(arrays), 2), dtype=float)
        polygons = [Polygon(vertices) for vertices in arrays]
        metrics, endpoint_distance = _interface_metrics(
            polygons, zero, topology, index_by_id, target_gap_mm
        )
        best = ((1.0, 1.0, float("inf"), float("inf"), float("inf"), 0.0), zero, polygons, metrics, endpoint_distance)

    score, deltas, polygons, metrics, endpoint_distance = best
    logging.getLogger(__name__).info('CANDIDATE_CLEARANCE_PERFORMANCE %s', json.dumps(
        dict(candidates=len(candidates), pruned=pruned, accelerated=distances is not None,
             scan_seconds=scan_seconds)))
    overlap = pairwise_positive_overlap_area(polygons)
    gaps = [metric.nominal_normal_gap_mm for metric in metrics]
    buffer_radius = float(
        measurement_error_radius_mm + machine_position_margin_mm / 2.0
    )
    adjacent_robust = _buffer_overlap(polygons, adjacency, buffer_radius)
    nonadjacent_robust = _buffer_overlap(polygons, nonadjacent, buffer_radius)
    robust_overlap = adjacent_robust + nonadjacent_robust
    feasible = bool(
        topology.graph_connected
        and metrics
        and overlap <= overlap_tolerance_mm2
        and endpoint_distance <= maximum_vertex_distance_mm + 1e-7
        and all(
            minimum_gap_mm - 1e-7
            <= gap
            <= maximum_gap_mm + 1e-7
            for gap in gaps
        )
    )
    pairwise_distances = [
        float(polygons[first].distance(polygons[second]))
        for first, second in all_pairs
    ]
    return ClearancePlan(
        feasible=feasible,
        translations_mm=tuple(
            (float(delta[0]), float(delta[1])) for delta in deltas
        ),
        planned_vertices=tuple(
            tuple(
                (float(x), float(y))
                for x, y in np.asarray(polygon.exterior.coords[:-1])
            )
            for polygon in polygons
        ),
        requested_clearance_mm=float(target_gap_mm),
        achieved_minimum_clearance_mm=min(pairwise_distances, default=0.0),
        maximum_pairwise_clearance_mm=max(pairwise_distances, default=0.0),
        max_corresponding_vertex_distance_mm=float(endpoint_distance),
        estimated_physical_overlap_area_mm2=float(overlap),
        robust_buffer_overlap_area_mm2=float(robust_overlap),
        uncertainty_buffer_radius_mm=buffer_radius,
        failure_reason=(None if feasible else "random_joint_clearance_infeasible"),
        strategy="random_topology_joint_translation",
        pre_compaction_max_gap_mm=0.0,
        post_compaction_max_gap_mm=max(gaps, default=0.0),
        adjacent_robust_buffer_overlap_area_mm2=float(adjacent_robust),
        nonadjacent_robust_buffer_overlap_area_mm2=float(nonadjacent_robust),
        total_translation_mm=float(sum(
            np.linalg.norm(delta) for delta in deltas
        )),
        interface_metrics=metrics,
    )
