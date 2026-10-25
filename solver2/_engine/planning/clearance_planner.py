"""Deterministic translation-only clearance planning for 2--4 rigid pieces."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
from shapely.affinity import translate
from shapely.geometry import Polygon
from solver2.acceleration import ClearanceScanBackend

from ..config import (
    A4_HEIGHT_MM,
    A4_WIDTH_MM,
    DIVIDER_Y_MM,
    OVERLAP_NUMERICAL_EPSILON_MM2,
    VERTEX_DISTANCE_LIMIT_MM,
)
from ..transforms import transform_points

# Compatibility alias retained for external callers.
PHYSICAL_OVERLAP_TOLERANCE_MM2 = OVERLAP_NUMERICAL_EPSILON_MM2

# ============================================================
# 拼图最终公共接口间隙调参（唯一权威默认值）
# ============================================================
DEFAULT_TARGET_SEAM_GAP_MM: float = 2.0
MIN_TARGET_SEAM_GAP_MM: float = 0.8
MAX_TARGET_SEAM_GAP_MM: float = 4.0
LARGE_SEAM_GAP_WARNING_MM: float = 6.0
COMPETITION_VERTEX_DISTANCE_LIMIT_MM: float = VERTEX_DISTANCE_LIMIT_MM


def pairwise_positive_overlap_area(polygons: Iterable[Polygon]) -> float:
    """Return positive-area intersections; point/line contacts contribute zero."""
    items = list(polygons)
    return float(sum(
        items[first].intersection(items[second]).area
        for first, second in combinations(range(len(items)), 2)
    ))


def pairwise_clearances(polygons: Iterable[Polygon]) -> list[float]:
    items = list(polygons)
    return [
        float(items[first].distance(items[second]))
        for first, second in combinations(range(len(items)), 2)
    ]


def interface_constraints_from_records(
    records: list[dict],
    interfaces: Iterable[object],
) -> list[dict]:
    """Build truth-free local seam normals from selected search interfaces."""
    by_id = {record["piece_id"]: record for record in records}
    index_by_id = {
        record["piece_id"]: index for index, record in enumerate(records)
    }
    constraints: list[dict] = []
    for interface in interfaces:
        first_id = str(interface.first_piece)
        second_id = str(interface.second_piece)
        if first_id not in by_id or second_id not in by_id:
            continue
        first = by_id[first_id]
        second = by_id[second_id]
        first_points = transform_points(
            [
                interface.first_segment.start_point,
                interface.first_segment.end_point,
            ],
            first["angle"],
            first["translation"],
        )
        second_points = transform_points(
            [
                interface.second_segment.end_point,
                interface.second_segment.start_point,
            ],
            second["angle"],
            second["translation"],
        )
        tangent = _unit_vector(
            np.mean(
                np.vstack((
                    first_points[1] - first_points[0],
                    second_points[1] - second_points[0],
                )),
                axis=0,
            )
        )
        normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
        first_center = np.mean(np.asarray(first["target"], dtype=float), axis=0)
        second_center = np.mean(
            np.asarray(second["target"], dtype=float), axis=0
        )
        if float(np.dot(normal, second_center - first_center)) < 0.0:
            normal = -normal
        constraints.append({
            "interface_id": interface.interface_id,
            "first_index": index_by_id[first_id],
            "second_index": index_by_id[second_id],
            "piece_a": first_id,
            "piece_b": second_id,
            "normal": normal.tolist(),
            "first_endpoints": first_points.tolist(),
            "second_endpoints": second_points.tolist(),
        })
    return constraints


@dataclass(frozen=True)
class ClearancePlan:
    feasible: bool
    translations_mm: tuple[tuple[float, float], ...]
    planned_vertices: tuple[tuple[tuple[float, float], ...], ...]
    requested_clearance_mm: float
    achieved_minimum_clearance_mm: float
    maximum_pairwise_clearance_mm: float
    max_corresponding_vertex_distance_mm: float
    estimated_physical_overlap_area_mm2: float
    robust_buffer_overlap_area_mm2: float
    uncertainty_buffer_radius_mm: float
    failure_reason: str | None = None
    strategy: str = "legacy_centroid_expansion"
    pre_compaction_max_gap_mm: float = 0.0
    post_compaction_max_gap_mm: float = 0.0
    adjacent_robust_buffer_overlap_area_mm2: float = 0.0
    nonadjacent_robust_buffer_overlap_area_mm2: float = 0.0
    total_translation_mm: float = 0.0
    interface_metrics: tuple["InterfaceClearanceMetrics", ...] = ()


@dataclass(frozen=True)
class InterfaceClearanceMetrics:
    interface_id: str
    piece_a: str
    piece_b: str
    nominal_normal_gap_mm: float
    tangential_misalignment_mm: float
    target_gap_mm: float
    estimated_intrusion_risk_mm: float
    endpoint_distance_max_mm: float
    compactable: bool

    def as_json(self) -> dict[str, object]:
        return {
            "interface_id": self.interface_id,
            "piece_a": self.piece_a,
            "piece_b": self.piece_b,
            "nominal_normal_gap_mm": self.nominal_normal_gap_mm,
            "tangential_misalignment_mm": self.tangential_misalignment_mm,
            "target_gap_mm": self.target_gap_mm,
            "estimated_intrusion_risk_mm": (
                self.estimated_intrusion_risk_mm
            ),
            "endpoint_distance_max_mm": self.endpoint_distance_max_mm,
            "compactable": self.compactable,
        }


def _translated(
    polygons: list[Polygon],
    deltas: np.ndarray,
) -> list[Polygon]:
    return [
        translate(poly, xoff=float(delta[0]), yoff=float(delta[1]))
        for poly, delta in zip(polygons, deltas)
    ]


def _inside_lower_a4(polygons: list[Polygon], buffer_radius: float) -> bool:
    for poly in polygons:
        bounds = poly.buffer(buffer_radius).bounds
        if (
            bounds[0] < -1e-7
            or bounds[2] > A4_WIDTH_MM + 1e-7
            or bounds[1] < DIVIDER_Y_MM - 1e-7
            or bounds[3] > A4_HEIGHT_MM + 1e-7
        ):
            return False
    return True


def _adjacent_pairs(
    polygons: list[Polygon],
    explicit_pairs: Iterable[tuple[int, int]] | None,
    tolerance_mm: float,
) -> list[tuple[int, int]]:
    if explicit_pairs is not None:
        return sorted({
            tuple(sorted((int(first), int(second))))
            for first, second in explicit_pairs
            if first != second
        })
    result: list[tuple[int, int]] = []
    for first, second in combinations(range(len(polygons)), 2):
        boundary_length = polygons[first].boundary.intersection(
            polygons[second].boundary
        ).length
        if (
            boundary_length > 1e-6
            or polygons[first].distance(polygons[second]) <= tolerance_mm
            or polygons[first].intersection(polygons[second]).area
            > PHYSICAL_OVERLAP_TOLERANCE_MM2
        ):
            result.append((first, second))
    return result


def _plan_clearance_legacy(
    estimated_vertices: list[list[list[float]] | np.ndarray],
    measurement_error_radius_mm: float,
    *,
    machine_position_margin_mm: float = 1.0,
    minimum_clearance_mm: float | None = None,
    maximum_vertex_distance_mm: float = VERTEX_DISTANCE_LIMIT_MM,
    base_correspondence_error_mm: float = 0.0,
    adjacency_pairs: Iterable[tuple[int, int]] | None = None,
    overlap_tolerance_mm2: float = PHYSICAL_OVERLAP_TOLERANCE_MM2,
    clearance_backend: ClearanceScanBackend | None = None,
) -> ClearancePlan:
    """Expand piece centroids about one fixed anchor until buffers are disjoint.

    Every candidate is a pure per-piece translation.  Rotations and vertex
    coordinates are never changed.  All anchors are tried deterministically;
    the feasible candidate with the smallest total translation is returned.
    """
    radius = float(measurement_error_radius_mm)
    margin = float(machine_position_margin_mm)
    requested = float(
        2.0 * radius + margin
        if minimum_clearance_mm is None
        else minimum_clearance_mm
    )
    buffer_radius = radius + margin / 2.0
    arrays = [np.asarray(vertices, dtype=float) for vertices in estimated_vertices]
    polygons = [Polygon(vertices) for vertices in arrays]
    if (
        not 2 <= len(polygons) <= 4
        or any(not poly.is_valid or poly.area <= 0.0 for poly in polygons)
    ):
        raise ValueError("clearance planner requires 2--4 valid polygons")

    adjacency = _adjacent_pairs(
        polygons,
        adjacency_pairs,
        tolerance_mm=max(1e-5, 3.0 * radius + margin),
    )
    centroids = np.asarray([
        [poly.centroid.x, poly.centroid.y] for poly in polygons
    ])
    buffered_bounds = np.asarray([poly.buffer(buffer_radius).bounds for poly in polygons])
    coarse_distances = None
    if clearance_backend is not None:
        coarse_distances = clearance_backend.minimum_distances(arrays, centroids)
        if coarse_distances is not None:
            coarse_distances = np.asarray(coarse_distances, dtype=float)
            if (coarse_distances.shape != (401,) or not np.isfinite(coarse_distances).all()
                    or np.any(coarse_distances < 0)):
                raise RuntimeError("间隙粗筛后端返回无效结果")
    best: tuple[float, np.ndarray, list[Polygon], float, float, float] | None = None

    def inspect(deltas: np.ndarray):
        # Reject impossible translations before constructing GEOS polygons.
        # All successful candidates still pass the original exact checks below.
        max_corresponding = max(
            (
                base_correspondence_error_mm
                + float(np.linalg.norm(deltas[first] - deltas[second]))
                for first, second in adjacency
            ),
            default=float(base_correspondence_error_mm),
        )
        rejected = (False, None, 0.0, 0.0, 0.0, 0.0, max_corresponding)
        if max_corresponding > maximum_vertex_distance_mm + 1e-7:
            return rejected
        bounds = buffered_bounds + deltas[:, [0, 1, 0, 1]]
        # A wider guard than the exact 1e-7 check avoids rejecting borderline
        # cases due to buffer/translation floating-point evaluation order.
        if (np.any(bounds[:, 0] < -1e-6)
                or np.any(bounds[:, 2] > A4_WIDTH_MM + 1e-6)
                or np.any(bounds[:, 1] < DIVIDER_Y_MM - 1e-6)
                or np.any(bounds[:, 3] > A4_HEIGHT_MM + 1e-6)):
            return rejected
        planned = _translated(polygons, deltas)
        clearances = pairwise_clearances(planned)
        minimum = min(clearances, default=float("inf"))
        if minimum + 1e-7 < requested:
            return rejected
        maximum = max(clearances, default=0.0)
        estimated_overlap = pairwise_positive_overlap_area(planned)
        if estimated_overlap > overlap_tolerance_mm2:
            return rejected
        buffered = [poly.buffer(buffer_radius) for poly in planned]
        robust_overlap = pairwise_positive_overlap_area(buffered)
        feasible = (
            minimum + 1e-7 >= requested
            and estimated_overlap <= overlap_tolerance_mm2
            and robust_overlap <= overlap_tolerance_mm2
            and max_corresponding <= maximum_vertex_distance_mm + 1e-7
            and _inside_lower_a4(planned, buffer_radius)
        )
        return (
            feasible,
            planned,
            minimum,
            maximum,
            estimated_overlap,
            robust_overlap,
            max_corresponding,
        )

    # A centroid dilation around an unchanged piece is deterministic, preserves
    # sector ordering at T/multi-way junctions, and separates collinear strips.
    for anchor in range(len(polygons)):
        vectors = centroids - centroids[anchor]
        if max(np.linalg.norm(vector) for vector in vectors) <= 1e-9:
            continue
        previous_alpha = 0.0
        for sample, alpha in enumerate(np.linspace(0.0, 1.0, 401)):
            if coarse_distances is not None and coarse_distances[sample] + 0.02 < requested:
                previous_alpha = float(alpha)
                continue
            deltas = vectors * float(alpha)
            checked = inspect(deltas)
            if checked[0]:
                low, high = previous_alpha, float(alpha)
                for _ in range(30):
                    middle = (low + high) / 2.0
                    if inspect(vectors * middle)[0]:
                        high = middle
                    else:
                        low = middle
                deltas = vectors * high
                (
                    _feasible,
                    planned,
                    minimum,
                    maximum,
                    estimated_overlap,
                    robust_overlap,
                    max_corresponding,
                ) = inspect(deltas)
                objective = float(np.sum(deltas**2))
                candidate = (
                    objective,
                    deltas,
                    planned,
                    minimum,
                    maximum,
                    max_corresponding,
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate
                break
            previous_alpha = float(alpha)

    if best is None:
        estimated_overlap = pairwise_positive_overlap_area(polygons)
        robust_overlap = pairwise_positive_overlap_area([
            poly.buffer(buffer_radius) for poly in polygons
        ])
        clearances = pairwise_clearances(polygons)
        return ClearancePlan(
            False,
            tuple((0.0, 0.0) for _ in polygons),
            tuple(tuple(map(tuple, vertices)) for vertices in arrays),
            requested,
            min(clearances, default=0.0),
            max(clearances, default=0.0),
            float(base_correspondence_error_mm),
            estimated_overlap,
            robust_overlap,
            buffer_radius,
            "clearance_constraints_infeasible",
        )

    _objective, deltas, planned, minimum, maximum, max_corresponding = best
    estimated_overlap = pairwise_positive_overlap_area(planned)
    robust_overlap = pairwise_positive_overlap_area([
        poly.buffer(buffer_radius) for poly in planned
    ])
    return ClearancePlan(
        True,
        tuple((float(delta[0]), float(delta[1])) for delta in deltas),
        tuple(
            tuple(
                (float(x), float(y))
                for x, y in np.asarray(poly.exterior.coords[:-1])
            )
            for poly in planned
        ),
        requested,
        float(minimum),
        float(maximum),
        float(max_corresponding),
        float(estimated_overlap),
        float(robust_overlap),
        buffer_radius,
    )


def _pair_overlap_area(
    polygons: list[Polygon],
    pairs: Iterable[tuple[int, int]],
) -> float:
    return float(sum(
        polygons[first].intersection(polygons[second]).area
        for first, second in pairs
    ))


def _unit_vector(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        return np.asarray([1.0, 0.0], dtype=float)
    return np.asarray(value, dtype=float) / norm


def _constraint_rows(
    polygons: list[Polygon],
    adjacency: list[tuple[int, int]],
    interface_constraints: Iterable[dict] | None,
    target_gap: float,
) -> list[dict]:
    supplied: dict[tuple[int, int], dict] = {}
    for item in interface_constraints or ():
        first = int(item["first_index"])
        second = int(item["second_index"])
        key = tuple(sorted((first, second)))
        supplied[key] = dict(item)
    rows: list[dict] = []
    centroids = np.asarray([
        [polygon.centroid.x, polygon.centroid.y]
        for polygon in polygons
    ])
    for ordinal, (first, second) in enumerate(adjacency, 1):
        item = supplied.get((first, second), {})
        normal = _unit_vector(np.asarray(
            item.get("normal", centroids[second] - centroids[first]),
            dtype=float,
        ))
        if float(np.dot(normal, centroids[second] - centroids[first])) < 0.0:
            normal = -normal
        tangent = np.asarray([-normal[1], normal[0]], dtype=float)
        rows.append({
            "interface_id": str(
                item.get("interface_id", f"adjacency_{ordinal:03d}")
            ),
            "first_index": first,
            "second_index": second,
            "piece_a": str(item.get("piece_a", first)),
            "piece_b": str(item.get("piece_b", second)),
            "normal": normal,
            "tangent": tangent,
            "target_gap_mm": float(
                item.get("target_gap_mm", target_gap)
            ),
            "first_endpoints": (
                np.asarray(item["first_endpoints"], dtype=float)
                if "first_endpoints" in item else None
            ),
            "second_endpoints": (
                np.asarray(item["second_endpoints"], dtype=float)
                if "second_endpoints" in item else None
            ),
        })
    return rows


def _apply_workspace_shift(
    arrays: list[np.ndarray],
    deltas: np.ndarray,
) -> np.ndarray:
    planned = [
        vertices + delta for vertices, delta in zip(arrays, deltas)
    ]
    points = np.vstack(planned)
    shift = np.zeros(2, dtype=float)
    if points[:, 0].min() < 0.0:
        shift[0] = -float(points[:, 0].min())
    if points[:, 0].max() + shift[0] > A4_WIDTH_MM:
        shift[0] += A4_WIDTH_MM - float(points[:, 0].max() + shift[0])
    if points[:, 1].min() < DIVIDER_Y_MM:
        shift[1] = DIVIDER_Y_MM - float(points[:, 1].min())
    if points[:, 1].max() + shift[1] > A4_HEIGHT_MM:
        shift[1] += A4_HEIGHT_MM - float(points[:, 1].max() + shift[1])
    return deltas + shift


def _separate_positive_overlaps(
    arrays: list[np.ndarray],
    deltas: np.ndarray,
    rows: list[dict],
    *,
    anchor_index: int,
    tolerance_mm2: float,
    maximum_iterations: int = 24,
) -> np.ndarray:
    """Project positive-area intersections apart with minimal translations."""
    result = np.asarray(deltas, dtype=float).copy()
    normal_by_pair = {
        tuple(sorted((row["first_index"], row["second_index"]))): row[
            "normal"
        ]
        for row in rows
    }
    for _ in range(maximum_iterations):
        polygons = [
            Polygon(vertices + delta)
            for vertices, delta in zip(arrays, result)
        ]
        changed = False
        for first, second in combinations(range(len(polygons)), 2):
            if (
                polygons[first].intersection(polygons[second]).area
                <= tolerance_mm2
            ):
                continue
            normal = normal_by_pair.get((first, second))
            if normal is None:
                normal = _unit_vector(np.asarray([
                    polygons[second].centroid.x
                    - polygons[first].centroid.x,
                    polygons[second].centroid.y
                    - polygons[first].centroid.y,
                ]))
            low, high = 0.0, 0.05
            while high <= COMPETITION_VERTEX_DISTANCE_LIMIT_MM:
                trial = result.copy()
                if first == anchor_index:
                    trial[second] += normal * high
                elif second == anchor_index:
                    trial[first] -= normal * high
                else:
                    trial[first] -= normal * high / 2.0
                    trial[second] += normal * high / 2.0
                trial_polygons = [
                    Polygon(vertices + delta)
                    for vertices, delta in zip(arrays, trial)
                ]
                if (
                    trial_polygons[first]
                    .intersection(trial_polygons[second]).area
                    <= tolerance_mm2
                ):
                    break
                low, high = high, high * 2.0
            if high > COMPETITION_VERTEX_DISTANCE_LIMIT_MM:
                continue
            for _binary in range(32):
                middle = (low + high) / 2.0
                trial = result.copy()
                if first == anchor_index:
                    trial[second] += normal * middle
                elif second == anchor_index:
                    trial[first] -= normal * middle
                else:
                    trial[first] -= normal * middle / 2.0
                    trial[second] += normal * middle / 2.0
                trial_polygons = [
                    Polygon(vertices + delta)
                    for vertices, delta in zip(arrays, trial)
                ]
                if (
                    trial_polygons[first]
                    .intersection(trial_polygons[second]).area
                    <= tolerance_mm2
                ):
                    high = middle
                else:
                    low = middle
            if first == anchor_index:
                result[second] += normal * high
            elif second == anchor_index:
                result[first] -= normal * high
            else:
                result[first] -= normal * high / 2.0
                result[second] += normal * high / 2.0
            changed = True
        if not changed:
            break
    return result


def _interface_gap(
    polygons: list[Polygon],
    row: dict,
) -> float:
    return float(
        polygons[row["first_index"]].distance(
            polygons[row["second_index"]]
        )
    )


def compact_clearance_layout(
    estimated_vertices: list[list[list[float]] | np.ndarray],
    translations_mm: np.ndarray,
    *,
    adjacency_pairs: Iterable[tuple[int, int]],
    interface_constraints: Iterable[dict] | None = None,
    target_gap_mm: float = DEFAULT_TARGET_SEAM_GAP_MM,
    anchor_index: int = 0,
    overlap_tolerance_mm2: float = PHYSICAL_OVERLAP_TOLERANCE_MM2,
    maximum_iterations: int = 8,
) -> tuple[np.ndarray, tuple[InterfaceClearanceMetrics, ...], float, float]:
    """Tighten seams by a deterministic global translation least-squares pass."""
    arrays = [np.asarray(vertices, dtype=float) for vertices in estimated_vertices]
    adjacency = sorted({
        tuple(sorted((int(first), int(second))))
        for first, second in adjacency_pairs
        if first != second
    })
    initial_polygons = [
        Polygon(vertices + delta)
        for vertices, delta in zip(arrays, translations_mm)
    ]
    rows = _constraint_rows(
        initial_polygons,
        adjacency,
        interface_constraints,
        target_gap_mm,
    )
    result = np.asarray(translations_mm, dtype=float).copy()
    pre_gap = max(
        (_interface_gap(initial_polygons, row) for row in rows),
        default=0.0,
    )
    count = len(arrays)
    for _ in range(maximum_iterations):
        polygons = [
            Polygon(vertices + delta)
            for vertices, delta in zip(arrays, result)
        ]
        matrix: list[np.ndarray] = []
        values: list[float] = []
        weights: list[float] = []
        for row in rows:
            first = row["first_index"]
            second = row["second_index"]
            normal = row["normal"]
            current_gap = _interface_gap(polygons, row)
            error = row["target_gap_mm"] - current_gap
            equation = np.zeros(2 * count, dtype=float)
            equation[2 * first:2 * first + 2] = -normal
            equation[2 * second:2 * second + 2] = normal
            matrix.append(equation)
            values.append(error)
            weights.append(5.0)
            # Preserve tangential alignment while the normal gap is changed.
            tangent_equation = np.zeros(2 * count, dtype=float)
            tangent_equation[2 * first:2 * first + 2] = -row["tangent"]
            tangent_equation[2 * second:2 * second + 2] = row["tangent"]
            matrix.append(tangent_equation)
            values.append(0.0)
            weights.append(1.0)
        for piece_index in range(count):
            for axis in range(2):
                equation = np.zeros(2 * count, dtype=float)
                equation[2 * piece_index + axis] = 1.0
                matrix.append(equation)
                values.append(0.0)
                weights.append(
                    1000.0 if piece_index == anchor_index else 0.08
                )
        if not matrix:
            break
        weighted = np.asarray(matrix) * np.sqrt(
            np.asarray(weights)[:, None]
        )
        rhs = np.asarray(values) * np.sqrt(np.asarray(weights))
        step = np.linalg.lstsq(weighted, rhs, rcond=None)[0].reshape(
            count, 2
        )
        if float(np.max(np.linalg.norm(step, axis=1))) <= 1e-5:
            break
        accepted = False
        factor = 1.0
        while factor >= 1.0 / 1024.0:
            trial = _apply_workspace_shift(arrays, result + factor * step)
            trial_polygons = [
                Polygon(vertices + delta)
                for vertices, delta in zip(arrays, trial)
            ]
            if (
                pairwise_positive_overlap_area(trial_polygons)
                <= overlap_tolerance_mm2
                and _inside_lower_a4(trial_polygons, 0.0)
            ):
                result = trial
                accepted = True
                break
            factor /= 2.0
        if not accepted:
            break
    polygons = [
        Polygon(vertices + delta)
        for vertices, delta in zip(arrays, result)
    ]
    metrics: list[InterfaceClearanceMetrics] = []
    for row in rows:
        first = row["first_index"]
        second = row["second_index"]
        relative = result[second] - result[first]
        gap = _interface_gap(polygons, row)
        if (
            row["first_endpoints"] is not None
            and row["second_endpoints"] is not None
        ):
            endpoint_errors = (
                row["first_endpoints"] + result[first]
                - row["second_endpoints"] - result[second]
            )
            endpoint_distance = float(np.max(
                np.linalg.norm(endpoint_errors, axis=1)
            ))
            tangential_misalignment = float(np.max(np.abs(
                endpoint_errors @ row["tangent"]
            )))
        else:
            endpoint_distance = float(gap)
            tangential_misalignment = float(abs(np.dot(
                relative, row["tangent"]
            )))
        metrics.append(InterfaceClearanceMetrics(
            interface_id=row["interface_id"],
            piece_a=row["piece_a"],
            piece_b=row["piece_b"],
            nominal_normal_gap_mm=gap,
            tangential_misalignment_mm=tangential_misalignment,
            target_gap_mm=float(row["target_gap_mm"]),
            estimated_intrusion_risk_mm=max(
                0.0, float(row["target_gap_mm"]) - gap
            ),
            endpoint_distance_max_mm=endpoint_distance,
            compactable=gap > row["target_gap_mm"] + 1e-3,
        ))
    post_gap = max(
        (metric.nominal_normal_gap_mm for metric in metrics),
        default=0.0,
    )
    return result, tuple(metrics), float(pre_gap), float(post_gap)


def _plan_interface_local_clearance(
    estimated_vertices: list[list[list[float]] | np.ndarray],
    measurement_error_radius_mm: float,
    *,
    machine_position_margin_mm: float,
    target_seam_gap_mm: float,
    minimum_target_seam_gap_mm: float,
    maximum_target_seam_gap_mm: float,
    typical_uncertainty_mm: float | None,
    uncertainty_multiplier: float,
    maximum_vertex_distance_mm: float,
    base_correspondence_error_mm: float,
    adjacency_pairs: Iterable[tuple[int, int]] | None,
    interface_constraints: Iterable[dict] | None,
    overlap_tolerance_mm2: float,
) -> ClearancePlan:
    arrays = [np.asarray(vertices, dtype=float) for vertices in estimated_vertices]
    polygons = [Polygon(vertices) for vertices in arrays]
    if (
        not 2 <= len(polygons) <= 4
        or any(not polygon.is_valid or polygon.area <= 0.0 for polygon in polygons)
    ):
        raise ValueError("clearance planner requires 2--4 valid polygons")
    adjacency = _adjacent_pairs(
        polygons,
        adjacency_pairs,
        tolerance_mm=max(target_seam_gap_mm, 1e-5),
    )
    sigma = typical_uncertainty_mm
    requested = (
        target_seam_gap_mm
        if sigma is None
        else (
            machine_position_margin_mm
            + uncertainty_multiplier * float(sigma)
        )
    )
    requested = float(np.clip(
        requested,
        minimum_target_seam_gap_mm,
        maximum_target_seam_gap_mm,
    ))
    rows = _constraint_rows(
        polygons, adjacency, interface_constraints, requested
    )
    deltas = _separate_positive_overlaps(
        arrays,
        np.zeros((len(arrays), 2), dtype=float),
        rows,
        anchor_index=0,
        tolerance_mm2=overlap_tolerance_mm2,
    )
    deltas, metrics, pre_gap, post_gap = compact_clearance_layout(
        arrays,
        deltas,
        adjacency_pairs=adjacency,
        interface_constraints=interface_constraints,
        target_gap_mm=requested,
        anchor_index=0,
        overlap_tolerance_mm2=overlap_tolerance_mm2,
    )
    planned = [
        Polygon(vertices + delta)
        for vertices, delta in zip(arrays, deltas)
    ]
    all_pairs = set(combinations(range(len(planned)), 2))
    adjacent_set = set(adjacency)
    nonadjacent = sorted(all_pairs - adjacent_set)
    buffer_radius = (
        float(measurement_error_radius_mm)
        + machine_position_margin_mm / 2.0
    )
    buffered = [polygon.buffer(buffer_radius) for polygon in planned]
    estimated_overlap = pairwise_positive_overlap_area(planned)
    robust_overlap = pairwise_positive_overlap_area(buffered)
    adjacent_robust = _pair_overlap_area(buffered, adjacency)
    nonadjacent_robust = _pair_overlap_area(buffered, nonadjacent)
    max_corresponding = max(
        (
            metric.endpoint_distance_max_mm
            for metric in metrics
        ),
        default=float(base_correspondence_error_mm),
    )
    feasible = bool(
        estimated_overlap <= overlap_tolerance_mm2
        and max_corresponding <= maximum_vertex_distance_mm + 1e-7
        and _inside_lower_a4(planned, 0.0)
    )
    clearances = pairwise_clearances(planned)
    return ClearancePlan(
        feasible=feasible,
        translations_mm=tuple(
            (float(delta[0]), float(delta[1])) for delta in deltas
        ),
        planned_vertices=tuple(
            tuple(
                (float(x), float(y))
                for x, y in polygon.exterior.coords[:-1]
            )
            for polygon in planned
        ),
        requested_clearance_mm=requested,
        achieved_minimum_clearance_mm=min(clearances, default=0.0),
        maximum_pairwise_clearance_mm=max(clearances, default=0.0),
        max_corresponding_vertex_distance_mm=float(max_corresponding),
        estimated_physical_overlap_area_mm2=float(estimated_overlap),
        robust_buffer_overlap_area_mm2=float(robust_overlap),
        uncertainty_buffer_radius_mm=float(buffer_radius),
        failure_reason=(
            None if feasible else "nominal_clearance_constraints_infeasible"
        ),
        strategy="interface_local_compaction",
        pre_compaction_max_gap_mm=pre_gap,
        post_compaction_max_gap_mm=post_gap,
        adjacent_robust_buffer_overlap_area_mm2=float(adjacent_robust),
        nonadjacent_robust_buffer_overlap_area_mm2=float(nonadjacent_robust),
        total_translation_mm=float(sum(
            np.linalg.norm(delta) for delta in deltas
        )),
        interface_metrics=metrics,
    )


def plan_clearance(
    estimated_vertices: list[list[list[float]] | np.ndarray],
    measurement_error_radius_mm: float,
    *,
    machine_position_margin_mm: float = 1.0,
    minimum_clearance_mm: float | None = None,
    maximum_vertex_distance_mm: float = VERTEX_DISTANCE_LIMIT_MM,
    base_correspondence_error_mm: float = 0.0,
    adjacency_pairs: Iterable[tuple[int, int]] | None = None,
    overlap_tolerance_mm2: float = PHYSICAL_OVERLAP_TOLERANCE_MM2,
    strategy: str = "legacy_centroid_expansion",
    target_seam_gap_mm: float = DEFAULT_TARGET_SEAM_GAP_MM,
    minimum_target_seam_gap_mm: float = MIN_TARGET_SEAM_GAP_MM,
    maximum_target_seam_gap_mm: float = MAX_TARGET_SEAM_GAP_MM,
    typical_uncertainty_mm: float | None = None,
    uncertainty_multiplier: float = 2.0,
    interface_constraints: Iterable[dict] | None = None,
    clearance_backend: ClearanceScanBackend | None = None,
) -> ClearancePlan:
    """Plan translations using either compatibility or production strategy."""
    if strategy == "legacy_centroid_expansion":
        return _plan_clearance_legacy(
            estimated_vertices,
            measurement_error_radius_mm,
            machine_position_margin_mm=machine_position_margin_mm,
            minimum_clearance_mm=minimum_clearance_mm,
            maximum_vertex_distance_mm=maximum_vertex_distance_mm,
            base_correspondence_error_mm=base_correspondence_error_mm,
            adjacency_pairs=adjacency_pairs,
            overlap_tolerance_mm2=overlap_tolerance_mm2,
            clearance_backend=clearance_backend,
        )
    if strategy != "interface_local_compaction":
        raise ValueError(f"unknown clearance strategy: {strategy}")
    return _plan_interface_local_clearance(
        estimated_vertices,
        measurement_error_radius_mm,
        machine_position_margin_mm=machine_position_margin_mm,
        target_seam_gap_mm=target_seam_gap_mm,
        minimum_target_seam_gap_mm=minimum_target_seam_gap_mm,
        maximum_target_seam_gap_mm=maximum_target_seam_gap_mm,
        typical_uncertainty_mm=typical_uncertainty_mm,
        uncertainty_multiplier=uncertainty_multiplier,
        maximum_vertex_distance_mm=maximum_vertex_distance_mm,
        base_correspondence_error_mm=base_correspondence_error_mm,
        adjacency_pairs=adjacency_pairs,
        interface_constraints=interface_constraints,
        overlap_tolerance_mm2=overlap_tolerance_mm2,
    )
