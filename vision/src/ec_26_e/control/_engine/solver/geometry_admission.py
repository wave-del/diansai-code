"""Truth-free admission gate between fully placed states and solver candidates.

The search tree may place every observed piece without producing a plausible
rectangle.  This module deliberately separates that *fully placed* condition
from a geometry candidate that is worth optimizing, clearing and executing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from ..geometry import pairwise_overlap
from ..runtime_config import RuntimeConfig
from .interface_matching import PairwiseInterfacePlacement
from .observation_uncertainty import ObservationUncertaintyProfile
from .placed_interface_discovery import discover_interfaces_in_placed_state
from .directed_roles import (
    JunctionClosureMetric,
    PieceOuterRole,
    RectangleFrameFit,
    RectangleRoleConsistency,
    evaluate_junction_closure,
    fit_rectangle_frame,
)


@dataclass(frozen=True)
class SearchState:
    """A partial or fully expanded node in the discrete search tree."""

    poses: dict[int, Any]
    unplaced_piece_indices: tuple[int, ...]
    declared_interfaces: tuple[PairwiseInterfacePlacement, ...]
    score_lower_bound: float = 0.0
    depth: int = 0
    anchor_index: int = 0
    stage_name: str = "unknown"


@dataclass(frozen=True)
class FullyPlacedState:
    """A search path with one rigid pose per piece, not yet a candidate."""

    poses: tuple[Any, ...]
    declared_interfaces: tuple[PairwiseInterfacePlacement, ...]
    expected_piece_count: int
    anchor_index: int = 0
    stage_name: str = "unknown"
    source_pieces: tuple[Any, ...] = ()


@dataclass
class InterfaceExplanationResult:
    complete: bool
    declared_interface_count: int
    discovered_interface_count: int
    missing_declared_interfaces: list[dict[str, Any]] = field(default_factory=list)
    unexplained_interfaces: list[dict[str, Any]] = field(default_factory=list)
    unexpected_contacts: list[dict[str, Any]] = field(default_factory=list)
    interval_coverage_errors: list[dict[str, Any]] = field(default_factory=list)
    interval_order_errors: list[dict[str, Any]] = field(default_factory=list)
    declared_coverage_ratio: float = 0.0
    discovered_coverage_ratio: float = 0.0
    t_junction_consistent: bool = True

    def as_json(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "declared_interface_count": self.declared_interface_count,
            "discovered_interface_count": self.discovered_interface_count,
            "missing_declared_interfaces": self.missing_declared_interfaces,
            "unexplained_interfaces": self.unexplained_interfaces,
            "unexpected_contacts": self.unexpected_contacts,
            "interval_coverage_errors": self.interval_coverage_errors,
            "interval_order_errors": self.interval_order_errors,
            "declared_coverage_ratio": self.declared_coverage_ratio,
            "discovered_coverage_ratio": self.discovered_coverage_ratio,
            "t_junction_consistent": self.t_junction_consistent,
        }


@dataclass
class GeometryAdmissibilityResult:
    accepted: bool
    rejection_reasons: list[str]
    target_rectangle_width_mm: float
    target_rectangle_height_mm: float
    rectangle_fill_ratio: float
    union_rectangle_iou_estimate: float
    overlap_area_mm2: float
    overlap_ratio: float
    nonadjacent_overlap_area_mm2: float
    uncovered_area_mm2: float
    uncovered_ratio: float
    outside_area_mm2: float
    outside_ratio: float
    boundary_rectangularity: float
    boundary_complexity: float
    declared_interface_count: int
    discovered_interface_count: int
    interface_explanation_complete: bool
    max_declared_interface_residual_mm: float
    max_discovered_interface_residual_mm: float
    repairability: str
    interface_explanation: InterfaceExplanationResult
    directed_endpoint_context_valid: bool = True
    opposite_side_valid: bool = True
    rectangle_frame_fit_valid: bool = True
    rectangle_role_consistency_valid: bool = True
    junction_closure_valid: bool = True
    per_piece_outer_role_valid: bool = True
    rectangle_frame_fit: RectangleFrameFit | None = None
    outer_roles: dict[str, PieceOuterRole] = field(default_factory=dict)
    rectangle_role_consistency: RectangleRoleConsistency | None = None
    junction_closure: list[JunctionClosureMetric] = field(default_factory=list)

    def rank_key(self) -> tuple[float, ...]:
        return (
            0.0 if self.accepted else 1.0,
            0.0 if self.directed_endpoint_context_valid else 1.0,
            0.0 if self.opposite_side_valid else 1.0,
            0.0 if self.interface_explanation_complete else 1.0,
            0.0 if self.rectangle_role_consistency_valid else 1.0,
            0.0 if self.junction_closure_valid else 1.0,
            (
                self.rectangle_role_consistency.normalized_cost
                if self.rectangle_role_consistency is not None
                else math.inf
            ),
            -self.rectangle_fill_ratio,
            -self.union_rectangle_iou_estimate,
            self.nonadjacent_overlap_area_mm2,
            self.overlap_area_mm2,
            self.outside_area_mm2,
            self.uncovered_area_mm2,
            self.max_discovered_interface_residual_mm,
            self.boundary_complexity,
        )

    def as_json(self) -> dict[str, Any]:
        result = dict(vars(self))
        result["interface_explanation"] = self.interface_explanation.as_json()
        result["rectangle_frame_fit"] = (
            self.rectangle_frame_fit.as_json()
            if self.rectangle_frame_fit is not None else None
        )
        result["outer_roles"] = {
            piece_id: role.as_json()
            for piece_id, role in self.outer_roles.items()
        }
        result["rectangle_role_consistency"] = (
            self.rectangle_role_consistency.as_json()
            if self.rectangle_role_consistency is not None else None
        )
        result["junction_closure"] = [
            row.as_json() for row in self.junction_closure
        ]
        return result


def _interval_length(interval: Sequence[float]) -> float:
    return max(0.0, float(max(interval) - min(interval)))


def _interval_intersection(
    first: Sequence[float], second: Sequence[float]
) -> float:
    return max(
        0.0,
        min(float(max(first)), float(max(second)))
        - max(float(min(first)), float(min(second))),
    )


def _declared_rows(
    declared: Iterable[PairwiseInterfacePlacement],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in declared:
        rows.append({
            "interface_id": item.interface_id,
            "composite_id": item.composite_id,
            "piece_a": item.first_piece,
            "edge_a": int(item.first_segment.edge_index),
            "interval_a": [
                float(item.first_segment.t_start),
                float(item.first_segment.t_end),
            ],
            "piece_b": item.second_piece,
            "edge_b": int(item.second_segment.edge_index),
            "interval_b": [
                float(item.second_segment.t_start),
                float(item.second_segment.t_end),
            ],
        })
    return rows


def _pair_edge_key(row: dict[str, Any]) -> tuple[tuple[str, int], tuple[str, int]]:
    first = (str(row["piece_a"]), int(row["edge_a"]))
    second = (str(row["piece_b"]), int(row["edge_b"]))
    return tuple(sorted((first, second)))  # type: ignore[return-value]


def _interval_for(
    row: dict[str, Any], piece_id: str, edge_index: int
) -> Sequence[float]:
    if row["piece_a"] == piece_id and int(row["edge_a"]) == edge_index:
        return row["interval_a"]
    return row["interval_b"]


def _merged_interval_length(intervals: Iterable[Sequence[float]]) -> float:
    ordered = sorted(
        (float(min(row)), float(max(row))) for row in intervals
    )
    if not ordered:
        return 0.0
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def validate_interface_explanation(
    declared_interfaces: Iterable[PairwiseInterfacePlacement],
    discovered: dict[str, Any],
    *,
    interval_tolerance: float = 0.08,
) -> InterfaceExplanationResult:
    """Compare declared edge intervals with all contacts in the placed layout."""

    declared = _declared_rows(declared_interfaces)
    actual = list(discovered.get("interfaces", []))
    declared_by_key: dict[tuple, list[dict[str, Any]]] = {}
    actual_by_key: dict[tuple, list[dict[str, Any]]] = {}
    for row in declared:
        declared_by_key.setdefault(_pair_edge_key(row), []).append(row)
    for row in actual:
        actual_by_key.setdefault(_pair_edge_key(row), []).append(row)

    missing: list[dict[str, Any]] = []
    unexplained: list[dict[str, Any]] = []
    declared_covered = 0.0
    declared_total = 0.0
    actual_covered = 0.0
    actual_total = 0.0

    for key, rows in declared_by_key.items():
        actual_rows = actual_by_key.get(key, [])
        for row in rows:
            piece_id, edge_index = key[0]
            interval = _interval_for(row, piece_id, edge_index)
            length = _interval_length(interval)
            declared_total += length
            covered = sum(
                _interval_intersection(
                    interval, _interval_for(other, piece_id, edge_index)
                )
                for other in actual_rows
            )
            covered = min(length, covered)
            declared_covered += covered
            if length > interval_tolerance and (
                covered / max(length, 1e-12) < 1.0 - interval_tolerance
            ):
                missing.append({
                    "interface_id": row["interface_id"],
                    "piece_edge_key": list(key),
                    "coverage_ratio": covered / max(length, 1e-12),
                })

    for key, rows in actual_by_key.items():
        declared_rows = declared_by_key.get(key, [])
        for row in rows:
            piece_id, edge_index = key[0]
            interval = _interval_for(row, piece_id, edge_index)
            length = _interval_length(interval)
            actual_total += length
            covered = sum(
                _interval_intersection(
                    interval, _interval_for(other, piece_id, edge_index)
                )
                for other in declared_rows
            )
            covered = min(length, covered)
            actual_covered += covered
            if length > interval_tolerance and (
                covered / max(length, 1e-12) < 1.0 - interval_tolerance
            ):
                unexplained.append({
                    "piece_edge_key": list(key),
                    "length_mm": float(row.get("length_mm", 0.0)),
                    "coverage_ratio": covered / max(length, 1e-12),
                    "declared_same_piece_edge_pair": bool(declared_rows),
                })

    coverage_errors: list[dict[str, Any]] = []
    order_errors: list[dict[str, Any]] = []
    composites: dict[str, list[dict[str, Any]]] = {}
    for row in declared:
        if row["composite_id"]:
            composites.setdefault(str(row["composite_id"]), []).append(row)
    for composite_id, rows in composites.items():
        by_edge: dict[tuple[str, int], list[Sequence[float]]] = {}
        for row in rows:
            for side in ("a", "b"):
                by_edge.setdefault(
                    (str(row[f"piece_{side}"]), int(row[f"edge_{side}"])),
                    [],
                ).append(row[f"interval_{side}"])
        for edge_key, intervals in by_edge.items():
            if len(intervals) < 2:
                continue
            sum_length = sum(_interval_length(row) for row in intervals)
            merged_length = _merged_interval_length(intervals)
            overlap = max(0.0, sum_length - merged_length)
            ordered = sorted((min(row), max(row)) for row in intervals)
            gap = sum(
                max(0.0, ordered[index + 1][0] - ordered[index][1])
                for index in range(len(ordered) - 1)
            )
            if overlap > interval_tolerance or gap > interval_tolerance:
                coverage_errors.append({
                    "composite_id": composite_id,
                    "piece_edge": list(edge_key),
                    "overlap_fraction": overlap,
                    "gap_fraction": gap,
                })
            declared_order: list[tuple[float, str]] = []
            discovered_order: list[tuple[float, str]] = []
            for row in rows:
                if not (
                    (row["piece_a"], int(row["edge_a"])) == edge_key
                    or (row["piece_b"], int(row["edge_b"])) == edge_key
                ):
                    continue
                declared_interval = _interval_for(
                    row, edge_key[0], edge_key[1]
                )
                declared_order.append((
                    0.5 * (
                        float(min(declared_interval))
                        + float(max(declared_interval))
                    ),
                    str(row["interface_id"]),
                ))
                actual_rows = actual_by_key.get(_pair_edge_key(row), [])
                if not actual_rows:
                    continue
                best_actual = max(
                    actual_rows,
                    key=lambda actual_row: _interval_intersection(
                        declared_interval,
                        _interval_for(
                            actual_row, edge_key[0], edge_key[1]
                        ),
                    ),
                )
                actual_interval = _interval_for(
                    best_actual, edge_key[0], edge_key[1]
                )
                discovered_order.append((
                    0.5 * (
                        float(min(actual_interval))
                        + float(max(actual_interval))
                    ),
                    str(row["interface_id"]),
                ))
            if (
                len(declared_order) >= 2
                and len(discovered_order) == len(declared_order)
                and [item[1] for item in sorted(declared_order)]
                != [item[1] for item in sorted(discovered_order)]
            ):
                order_errors.append({
                    "composite_id": composite_id,
                    "piece_edge": list(edge_key),
                    "declared_order": [
                        item[1] for item in sorted(declared_order)
                    ],
                    "discovered_order": [
                        item[1] for item in sorted(discovered_order)
                    ],
                })

    unexpected_contacts = [
        row for row in discovered.get("point_contacts", [])
        if row.get("geometry_type") not in {"Point", "MultiPoint"}
    ]
    declared_ratio = declared_covered / max(declared_total, 1e-12)
    discovered_ratio = actual_covered / max(actual_total, 1e-12)
    complete = not (
        missing or unexplained or unexpected_contacts
        or coverage_errors or order_errors
    )
    return InterfaceExplanationResult(
        complete=complete,
        declared_interface_count=len(declared),
        discovered_interface_count=len(actual),
        missing_declared_interfaces=missing,
        unexplained_interfaces=unexplained,
        unexpected_contacts=unexpected_contacts,
        interval_coverage_errors=coverage_errors,
        interval_order_errors=order_errors,
        declared_coverage_ratio=float(declared_ratio),
        discovered_coverage_ratio=float(discovered_ratio),
        t_junction_consistent=not (coverage_errors or order_errors),
    )


def evaluate_geometry_admissibility(
    fully_placed_state: FullyPlacedState,
    profile: ObservationUncertaintyProfile,
    config: RuntimeConfig,
    stage_timer: Any | None = None,
) -> GeometryAdmissibilityResult:
    """Reject obviously wrong global layouts before expensive optimization."""

    reasons: list[str] = []
    poses = list(fully_placed_state.poses)
    if len(poses) != fully_placed_state.expected_piece_count:
        reasons.append("not_fully_placed")
    arrays = [
        np.asarray(pose.target_vertices_local, dtype=float) for pose in poses
    ]
    if any(array.ndim != 2 or array.shape[1:] != (2,) for array in arrays):
        reasons.append("invalid_polygon")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        reasons.append("nonfinite_transform")
    if not arrays or reasons:
        empty_explanation = InterfaceExplanationResult(False, 0, 0)
        return GeometryAdmissibilityResult(
            accepted=False,
            rejection_reasons=sorted(set(reasons)),
            target_rectangle_width_mm=0.0,
            target_rectangle_height_mm=0.0,
            rectangle_fill_ratio=0.0,
            union_rectangle_iou_estimate=0.0,
            overlap_area_mm2=math.inf,
            overlap_ratio=math.inf,
            nonadjacent_overlap_area_mm2=math.inf,
            uncovered_area_mm2=math.inf,
            uncovered_ratio=math.inf,
            outside_area_mm2=math.inf,
            outside_ratio=math.inf,
            boundary_rectangularity=0.0,
            boundary_complexity=math.inf,
            declared_interface_count=0,
            discovered_interface_count=0,
            interface_explanation_complete=False,
            max_declared_interface_residual_mm=math.inf,
            max_discovered_interface_residual_mm=math.inf,
            repairability="not_repairable",
            interface_explanation=empty_explanation,
        )
    polygons = [Polygon(array) for array in arrays]
    if any(
        not polygon.is_valid or polygon.is_empty or polygon.area <= 1e-6
        for polygon in polygons
    ):
        reasons.append("invalid_polygon")

    if not polygons or "invalid_polygon" in reasons:
        empty_explanation = InterfaceExplanationResult(False, 0, 0)
        return GeometryAdmissibilityResult(
            accepted=False,
            rejection_reasons=sorted(set(reasons)),
            target_rectangle_width_mm=0.0,
            target_rectangle_height_mm=0.0,
            rectangle_fill_ratio=0.0,
            union_rectangle_iou_estimate=0.0,
            overlap_area_mm2=math.inf,
            overlap_ratio=math.inf,
            nonadjacent_overlap_area_mm2=math.inf,
            uncovered_area_mm2=math.inf,
            uncovered_ratio=math.inf,
            outside_area_mm2=math.inf,
            outside_ratio=math.inf,
            boundary_rectangularity=0.0,
            boundary_complexity=math.inf,
            declared_interface_count=0,
            discovered_interface_count=0,
            interface_explanation_complete=False,
            max_declared_interface_residual_mm=math.inf,
            max_discovered_interface_residual_mm=math.inf,
            repairability="not_repairable",
            interface_explanation=empty_explanation,
        )

    rectangle_metrics_started = perf_counter()
    union = unary_union(polygons)
    minx, miny, maxx, maxy = union.bounds
    width, height = float(maxx - minx), float(maxy - miny)
    rectangle = box(minx, miny, maxx, maxy)
    dimension_relaxation = float(config.geometry_candidate_dimension_relaxation_mm)
    standard_dimensions_ok = bool(
        90.0 - dimension_relaxation <= width <= 120.0 + dimension_relaxation
        and 50.0 - dimension_relaxation
        <= height <= 90.0 + dimension_relaxation
    )
    repairable_relaxation = float(
        config.repairable_geometry_dimension_relaxation_mm
    )
    repairable_dimensions_ok = bool(
        90.0 - repairable_relaxation
        <= width <= 120.0 + repairable_relaxation
        and 50.0 - repairable_relaxation
        <= height <= 90.0 + repairable_relaxation
    )

    total_piece_area = float(sum(polygon.area for polygon in polygons))
    union_area = float(union.area)
    rectangle_area = max(float(rectangle.area), 1e-12)
    overlap = float(pairwise_overlap(polygons))
    overlap_ratio = overlap / max(total_piece_area, 1e-12)
    uncovered = float(rectangle.difference(union).area)
    outside = float(union.difference(rectangle).area)
    uncovered_ratio = uncovered / rectangle_area
    outside_ratio = outside / max(union_area, 1e-12)
    fill_ratio = union_area / rectangle_area
    intersection_area = float(union.intersection(rectangle).area)
    iou = intersection_area / max(
        union_area + rectangle_area - intersection_area, 1e-12
    )

    declared_pairs = {
        frozenset((item.first_piece, item.second_piece))
        for item in fully_placed_state.declared_interfaces
    }
    nonadjacent_overlap = 0.0
    for first_index, second_index in combinations(range(len(poses)), 2):
        pair_overlap = float(
            polygons[first_index].intersection(polygons[second_index]).area
        )
        pair = frozenset((
            poses[first_index].piece_id, poses[second_index].piece_id
        ))
        if pair not in declared_pairs:
            nonadjacent_overlap += pair_overlap

    standard_nonadjacent_limit = max(
        config.overlap_numerical_epsilon_mm2,
        config.maximum_geometry_candidate_nonadjacent_overlap_ratio
        * total_piece_area,
    )
    standard_metrics_ok = bool(
        standard_dimensions_ok
        and fill_ratio >= config.minimum_geometry_candidate_fill_ratio
        and iou >= config.minimum_geometry_candidate_rectangle_iou
        and overlap_ratio <= config.maximum_geometry_candidate_overlap_ratio
        and nonadjacent_overlap <= standard_nonadjacent_limit
        and uncovered_ratio
        <= config.maximum_geometry_candidate_uncovered_ratio
        and outside_ratio <= config.maximum_geometry_candidate_outside_ratio
    )
    # A deliberately wider, still bounded layer lets soft_l1 repair noisy
    # observations.  It remains well above the seed-12010 bad layout
    # (IoU≈0.63), and interface explanation is still mandatory below.
    repairable_metrics_ok = bool(
        repairable_dimensions_ok
        and iou >= config.minimum_repairable_geometry_iou
        and overlap_ratio
        <= config.maximum_repairable_geometry_overlap_ratio
        and nonadjacent_overlap
        <= config.maximum_repairable_geometry_overlap_ratio * total_piece_area
        and uncovered_ratio
        <= config.maximum_repairable_geometry_uncovered_ratio
        and outside_ratio <= config.maximum_geometry_candidate_outside_ratio
    )
    if not (standard_metrics_ok or repairable_metrics_ok):
        if not repairable_dimensions_ok:
            reasons.append("illegal_target_dimensions")
        if fill_ratio < config.minimum_geometry_candidate_fill_ratio:
            reasons.append("fill_ratio_too_low")
        if iou < config.minimum_repairable_geometry_iou:
            reasons.append("rectangle_iou_too_low")
        if (
            overlap_ratio
            > config.maximum_repairable_geometry_overlap_ratio
        ):
            reasons.append("excessive_overlap")
        if (
            nonadjacent_overlap
            > config.maximum_repairable_geometry_overlap_ratio
            * total_piece_area
        ):
            reasons.append("nonadjacent_overlap")
        if (
            uncovered_ratio
            > config.maximum_repairable_geometry_uncovered_ratio
        ):
            reasons.append("excessive_uncovered_area")
        if outside_ratio > config.maximum_geometry_candidate_outside_ratio:
            reasons.append("excessive_outside_area")
    if stage_timer is not None:
        stage_timer.add(
            "rectangle_metrics",
            perf_counter() - rectangle_metrics_started,
        )

    # Search placements make a declared seam geometrically coincident.  A
    # broad observation-error tolerance here would incorrectly rediscover
    # merely nearby outer edges as real contacts, especially around a T node.
    # Observation uncertainty belongs to later repair/risk stages; positive
    # contact rediscovery intentionally uses a tight geometric tolerance.
    discovery_tolerance = max(
        1e-5,
        min(
            1e-3,
            0.05 * float(profile.typical_sigma_mm or 0.02),
        ),
    )
    rediscovery_started = perf_counter()
    placed_vertices = {
        pose.piece_id: np.asarray(
            pose.target_vertices_local, dtype=float
        )
        for pose in poses
    }
    discovered = discover_interfaces_in_placed_state(
        placed_vertices,
        tolerance_mm=discovery_tolerance,
        length_epsilon_mm=max(1e-6, 0.05 * discovery_tolerance),
    )
    if stage_timer is not None:
        stage_timer.add(
            "interface_rediscovery",
            perf_counter() - rediscovery_started,
        )
    explanation_started = perf_counter()
    explanation = validate_interface_explanation(
        fully_placed_state.declared_interfaces,
        discovered,
        interval_tolerance=config.interface_explanation_interval_tolerance,
    )
    # A stage may declare only an atomic subinterval of a closure edge.  When
    # rediscovery finds the same piece/edge pair and the global union is
    # already an essentially exact, non-overlapping rectangle, the observed
    # positive-length closure safely completes that declaration.  A different
    # piece/edge pair, a missing declared seam, or an imperfect global layout
    # is never auto-completed.
    rediscovered_closure_completes_atomic_intervals = bool(
        explanation.unexplained_interfaces
        and not explanation.missing_declared_interfaces
        and not explanation.interval_coverage_errors
        and not explanation.interval_order_errors
        and iou >= 0.98
        and overlap_ratio <= 0.01
        and uncovered_ratio <= 0.02
        and outside_ratio <= 0.01
    )
    if rediscovered_closure_completes_atomic_intervals:
        explanation.complete = True
    if not explanation.complete:
        reasons.append("interface_explanation_incomplete")
    if (
        explanation.unexplained_interfaces
        and not rediscovered_closure_completes_atomic_intervals
    ):
        reasons.append("unexpected_contact")
    if not explanation.t_junction_consistent:
        reasons.append("t_junction_inconsistent")
    if stage_timer is not None:
        stage_timer.add(
            "interface_explanation_validation",
            perf_counter() - explanation_started,
        )

    exterior = union.convex_hull.exterior
    boundary_complexity = float(max(0, len(exterior.coords) - 5))
    boundary_rectangularity = float(
        union_area / max(float(union.convex_hull.area), 1e-12)
    )
    declared_residual = max(
        (
            float(item.transform_residual or 0.0)
            for item in fully_placed_state.declared_interfaces
        ),
        default=0.0,
    )
    discovered_residual = float(discovered.get("cycle_residual_mm", 0.0))

    directed_endpoint_context_valid = all(
        bool(getattr(item, "directed_endpoint_context_valid", True))
        and float(getattr(item, "directed_endpoint_cost", 0.0))
        <= float(config.directed_endpoint_hard_rejection_z)
        for item in fully_placed_state.declared_interfaces
    )
    opposite_side_valid = all(
        (
            getattr(item, "opposite_side_result", None) is None
            or bool(item.opposite_side_result.valid)
            or bool(item.opposite_side_result.ambiguous)
        )
        for item in fully_placed_state.declared_interfaces
    )
    if not directed_endpoint_context_valid:
        reasons.append("directed_endpoint_mismatch")
    if not opposite_side_valid:
        reasons.append("same_side_interface")

    excluded_edges = {
        (item.first_piece, int(item.first_segment.edge_index))
        for item in fully_placed_state.declared_interfaces
    } | {
        (item.second_piece, int(item.second_segment.edge_index))
        for item in fully_placed_state.declared_interfaces
    }
    # 高质量闭合布局中由实际正长度接触补出的接口同样是内拼缝，不能
    # 再参与矩形外边角色拟合。
    if rediscovered_closure_completes_atomic_intervals:
        for item in discovered.get("interfaces", []):
            excluded_edges.add((
                str(item["piece_a"]), int(item["edge_a"])
            ))
            excluded_edges.add((
                str(item["piece_b"]), int(item["edge_b"])
            ))
    source_by_id = {
        piece.piece_id: piece for piece in fully_placed_state.source_pieces
    }
    if config.enable_directed_role_admission:
        role_started = perf_counter()
        rectangle_frame, outer_roles, role_consistency = fit_rectangle_frame(
            placed_vertices,
            profile,
            excluded_piece_edges=excluded_edges,
            angle_model_floor_deg=config.angle_model_floor_deg,
            hard_rejection_z=config.rectangle_role_hard_rejection_z,
            minimum_side_coverage_ratio=(
                config.minimum_outer_side_coverage_ratio
            ),
        )
        # 精确/标准候选必须立即满足全局角色。只有在公开输入没有绝对
        # covariance 且候选仅处于 repairable 层时，角色残差才可作为
        # ambiguous 进入 soft_l1；它仍排在角色有效候选之后，优化后必须
        # 重新通过正式准入。这样不会让局部平行四边形压过标准正确解。
        post_optimization = (
            fully_placed_state.stage_name
            == "post_optimization_revalidation"
        )
        essentially_exact_layout = bool(
            iou >= 0.995
            and overlap_ratio <= 1e-4
            and uncovered_ratio <= 0.005
            and outside_ratio <= 1e-4
        )
        role_hard_required = bool(
            post_optimization
            or profile.absolute_measurement_sigma_known
            or (standard_metrics_ok and essentially_exact_layout)
        )
        if role_hard_required:
            for reason in rectangle_frame.rejection_reasons:
                reasons.append(reason)
        junction_closure = evaluate_junction_closure(
            placed_vertices,
            rectangle_frame,
            profile,
            source_pieces=source_by_id,
            cluster_tolerance_mm=config.junction_cluster_tolerance_mm,
            angle_model_floor_deg=config.angle_model_floor_deg,
            hard_rejection_z=config.junction_closure_hard_rejection_z,
        )
        junction_closure_valid = all(
            row.valid or row.ambiguous for row in junction_closure
        )
        if not junction_closure_valid and role_hard_required:
            reasons.append("junction_angle_inconsistent")
        if stage_timer is not None:
            stage_timer.add(
                "directed_role_admission", perf_counter() - role_started
            )
    else:
        rectangle_frame = None
        outer_roles = {}
        role_consistency = None
        junction_closure = []
        junction_closure_valid = True

    rectangle_frame_fit_valid = bool(
        rectangle_frame is None or rectangle_frame.valid
    )
    rectangle_role_consistency_valid = bool(
        role_consistency is None or role_consistency.valid
    )
    per_piece_outer_role_valid = bool(
        not outer_roles
        or all(not role.ambiguous for role in outer_roles.values())
    )
    accepted = not reasons
    repairability = (
        "admissible_exact"
        if accepted and standard_metrics_ok
        else (
            "admissible_repairable"
            if accepted and repairable_metrics_ok
            else "not_repairable"
        )
    )
    return GeometryAdmissibilityResult(
        accepted=accepted,
        rejection_reasons=sorted(set(reasons)),
        target_rectangle_width_mm=float(width),
        target_rectangle_height_mm=float(height),
        rectangle_fill_ratio=float(fill_ratio),
        union_rectangle_iou_estimate=float(iou),
        overlap_area_mm2=overlap,
        overlap_ratio=float(overlap_ratio),
        nonadjacent_overlap_area_mm2=float(nonadjacent_overlap),
        uncovered_area_mm2=uncovered,
        uncovered_ratio=float(uncovered_ratio),
        outside_area_mm2=outside,
        outside_ratio=float(outside_ratio),
        boundary_rectangularity=boundary_rectangularity,
        boundary_complexity=boundary_complexity,
        declared_interface_count=len(fully_placed_state.declared_interfaces),
        discovered_interface_count=len(discovered.get("interfaces", [])),
        interface_explanation_complete=explanation.complete,
        max_declared_interface_residual_mm=declared_residual,
        max_discovered_interface_residual_mm=discovered_residual,
        repairability=repairability,
        interface_explanation=explanation,
        directed_endpoint_context_valid=directed_endpoint_context_valid,
        opposite_side_valid=opposite_side_valid,
        rectangle_frame_fit_valid=rectangle_frame_fit_valid,
        rectangle_role_consistency_valid=(
            rectangle_role_consistency_valid
        ),
        junction_closure_valid=junction_closure_valid,
        per_piece_outer_role_valid=per_piece_outer_role_valid,
        rectangle_frame_fit=rectangle_frame,
        outer_roles=outer_roles,
        rectangle_role_consistency=role_consistency,
        junction_closure=junction_closure,
    )


def candidate_discrete_signature(candidate: Any) -> tuple:
    """Discrete interpretation signature used to preserve pool diversity."""

    interfaces = tuple(sorted(
        (
            item.first_piece,
            int(item.first_segment.edge_index),
            round(float(item.first_segment.t_start), 4),
            round(float(item.first_segment.t_end), 4),
            item.second_piece,
            int(item.second_segment.edge_index),
            round(float(item.second_segment.t_start), 4),
            round(float(item.second_segment.t_end), 4),
            item.composite_id,
        )
        for item in candidate.interfaces
    ))
    relative = tuple(sorted(
        (
            pose.piece_id,
            round(float(pose.angle) % 360.0, 3),
        )
        for pose in candidate.poses
    ))
    admission = getattr(candidate, "admissibility", None)
    outer_roles = tuple(sorted(
        (
            piece_id,
            tuple(role.supported_sides),
        )
        for piece_id, role in (
            getattr(admission, "outer_roles", {}) or {}
        ).items()
    ))
    junction_order = tuple(sorted(
        (
            row.junction_type,
            round(float(row.location[0]), 3),
            round(float(row.location[1]), 3),
            round(float(row.observed_angle_sum_deg), 2),
        )
        for row in (
            getattr(admission, "junction_closure", []) or []
        )
    ))
    return interfaces, relative, outer_roles, junction_order


class GeometryCandidatePool:
    """Bounded best-candidate pool that never acts as a search stop condition."""

    def __init__(
        self, capacity: int, max_variants_per_discrete_signature: int = 2
    ) -> None:
        self.capacity = max(1, int(capacity))
        self.max_variants_per_discrete_signature = max(
            1, int(max_variants_per_discrete_signature)
        )
        self._items: list[Any] = []
        self.replacements = 0
        self.duplicates = 0
        self.discarded = 0
        self.considered = 0
        self.ever_full = False
        self.considered_after_full = 0

    @staticmethod
    def _rank(candidate: Any) -> tuple:
        admission = candidate.admissibility
        return (
            *(admission.rank_key() if admission is not None else (1.0,)),
            float(candidate.total),
        )

    def _remove_identity(self, candidate: Any) -> None:
        for index, item in enumerate(self._items):
            if item is candidate:
                del self._items[index]
                return
        raise ValueError("candidate identity is not present in pool")

    def consider(self, candidate: Any) -> bool:
        self.considered += 1
        admission = getattr(candidate, "admissibility", None)
        if admission is None or not admission.accepted:
            self.discarded += 1
            return False
        if len(self._items) >= self.capacity:
            self.ever_full = True
            self.considered_after_full += 1
        signature = candidate_discrete_signature(candidate)
        siblings = [
            item for item in self._items
            if candidate_discrete_signature(item) == signature
        ]
        if len(siblings) >= self.max_variants_per_discrete_signature:
            worst_sibling = max(siblings, key=self._rank)
            if self._rank(candidate) >= self._rank(worst_sibling):
                self.duplicates += 1
                return False
            self._remove_identity(worst_sibling)
            self._items.append(candidate)
            self.replacements += 1
            return True
        if len(self._items) < self.capacity:
            self._items.append(candidate)
            if len(self._items) >= self.capacity:
                self.ever_full = True
            return True
        worst = max(self._items, key=self._rank)
        if self._rank(candidate) < self._rank(worst):
            self._remove_identity(worst)
            self._items.append(candidate)
            self.replacements += 1
            return True
        self.discarded += 1
        return False

    def items(self) -> list[Any]:
        return sorted(self._items, key=self._rank)

    def __len__(self) -> int:
        return len(self._items)
