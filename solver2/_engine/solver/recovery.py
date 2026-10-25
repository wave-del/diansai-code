"""Bounded, deterministic recovery for random puzzle candidates."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, replace
from math import radians, sin
from time import perf_counter
from typing import Any, Callable
from ..appearance.variant_pool import distinct_safe_variants

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..config import (
    TARGET_HEIGHT_MAX_MM,
    TARGET_HEIGHT_MIN_MM,
    TARGET_WIDTH_MAX_MM,
    TARGET_WIDTH_MIN_MM,
)
from ..planning.clearance_planner import (
    ClearancePlan,
)
from ..planning.motion_plan import build_moves, validate_transport_paths
from ..runtime_config import RuntimeConfig
from .optimizer import optimize_generic_candidate
from .placed_interface_discovery import discover_interfaces_in_placed_state
from .preprocess import ProcessedPiece
from .rectangle_fit import target_offset
from .search import (
    RandomCandidate,
    SearchRunStats,
    stream_random_candidates,
)
from .execution_gate import point_inside_a4, pick_place_closes
from .random_clearance import plan_random_clearance
from .random_safety import (
    RANDOM_TARGET_LONG_SIDE_MAX_MM,
    RANDOM_TARGET_SHORT_SIDE_MAX_MM,
    RandomCandidatePool,
    RandomSafetyAssessment,
    assess_random_assembly,
)
from .observation_uncertainty import (
    DEFAULT_HARD_ERROR_RADIUS_MM,
    ObservationUncertaintyProfile,
    vertex_distance_metrics,
)
from .progressive_search import (
    ProgressiveSearchStage,
    stages_from_config,
)


STAGE_MINIMUM_SECONDS = {
    "quick_geometry": 0.02,
    "soft_l1": 0.25,
    "clearance": 0.25,
    "motion": 0.10,
    "finalization": 1.20,
}


@dataclass
class StageCostPredictor:
    """Conservative per-stage cost predictor used by the cooperative deadline."""

    alpha: float = 0.35
    safety_factor: float = 1.75
    samples: dict[str, deque[float]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=10))
    )
    ema: dict[str, float] = field(default_factory=dict)
    longest_noninterruptible_call_seconds: float = 0.0

    def record(self, stage: str, elapsed: float) -> None:
        value = max(0.0, float(elapsed))
        self.samples[stage].append(value)
        previous = self.ema.get(stage, value)
        self.ema[stage] = self.alpha * value + (1.0 - self.alpha) * previous
        self.longest_noninterruptible_call_seconds = max(
            self.longest_noninterruptible_call_seconds, value
        )

    def predict(self, stage: str) -> float:
        values = list(self.samples.get(stage, ()))
        p90 = float(np.percentile(values, 90)) if values else 0.0
        ema = self.ema.get(stage, 0.0) * self.safety_factor
        return max(STAGE_MINIMUM_SECONDS.get(stage, 0.02), ema, p90)


@dataclass
class RecoveryCoordinator:
    started: float
    config: RuntimeConfig
    surface_mode: str = "white"
    monotonic_fn: Callable[[], float] = perf_counter
    predictor: StageCostPredictor = field(default_factory=StageCostPredictor)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    incumbent: "EvaluatedVariant | None" = None
    safe_candidates: list["EvaluatedVariant"] = field(default_factory=list)

    @property
    def total_deadline(self) -> float:
        if self.config.unlimited_search:
            return float("inf")
        return self.started + self.config.recovery_total_deadline_seconds

    @property
    def solve_deadline(self) -> float:
        appearance_reserve = (
            self.config.appearance_resolution_reserve_seconds
            if self.surface_mode == "patterned" else 0.0
        )
        return (
            self.total_deadline
            - self.config.recovery_finalization_reserve_seconds
            - appearance_reserve
        )

    @property
    def initial_stage_deadline(self) -> float:
        return min(
            self.solve_deadline,
            self.started + self.config.recovery_initial_stage_seconds,
        )

    def remaining_solve_seconds(self) -> float:
        return max(0.0, self.solve_deadline - self.monotonic_fn())

    def can_start(self, *stages: str) -> bool:
        required = sum(self.predictor.predict(stage) for stage in stages)
        return self.remaining_solve_seconds() >= required

    def consider(self, variant: "EvaluatedVariant") -> None:
        """完整合格变体优先；可执行诊断候选不能压过已通过全部检查的解。"""

        if not variant.machine_executable:
            return
        if (
            hasattr(variant, "candidate_id")
            and hasattr(variant, "variant_id")
        ):
            key = (variant.candidate_id, variant.variant_id)
            if all(
                (item.candidate_id, item.variant_id) != key
                for item in self.safe_candidates
            ):
                self.safe_candidates.append(variant)
                self.safe_candidates.sort(key=lambda item: (not item.safe, item.quality_key))
                self.safe_candidates = distinct_safe_variants(self.safe_candidates) + [
                    item for item in self.safe_candidates if not item.safe
                ]
                del self.safe_candidates[
                    self.config.max_appearance_candidates:
                ]
        if self.incumbent is None or (not variant.safe, variant.quality_key) < (
            not self.incumbent.safe, self.incumbent.quality_key
        ):
            self.incumbent = variant

    def should_continue_candidates(self) -> bool:
        # Shape-only solving is complete once the selected incumbent passed
        # geometry, clearance and motion validation. A merely executable
        # diagnostic candidate must never trigger this shortcut. Patterned
        # matching still needs alternatives for appearance disambiguation.
        safe_count = len(distinct_safe_variants(self.safe_candidates))
        if self.incumbent is not None and self.incumbent.safe and self.incumbent.machine_executable:
            safe_count = max(1,safe_count)
        if (
            self.incumbent is not None
            and self.incumbent.safe
            and self.incumbent.machine_executable
            and safe_count >= (
                max(2, self.config.minimum_safe_candidates)
                if self.surface_mode == "patterned" else self.config.minimum_safe_candidates
            )
        ):
            return False
        if self.monotonic_fn() >= self.solve_deadline:
            return False
        stages = ("quick_geometry", "clearance", "motion")
        if self.incumbent is None:
            return self.can_start(*stages)
        return self.can_start("quick_geometry", "soft_l1", "clearance", "motion")


@dataclass
class EvaluatedVariant:
    candidate: RandomCandidate
    candidate_id: str
    variant_id: str
    records: list[dict[str, Any]]
    width: float
    height: float
    optimization: dict[str, Any]
    clearance: ClearancePlan
    nominal_max_vertex_distance_mm: float
    estimated_risk_vertex_distance_mm: float
    robust_max_vertex_distance_mm: float
    piece_vertex_uncertainty_mm: dict[str, float]
    machine_executable: bool
    safe: bool
    block_reasons: list[str]
    quality_key: tuple[float, ...]
    geometry_admissibility: dict[str, Any] | None = None


def rank_candidate_for_execution_policy(
    *,
    candidate: RandomCandidate,
    variant_id: str,
    clearance: ClearancePlan,
    machine_executable: bool,
    topology_complete: bool = True,
    mirrored: bool = False,
    assessment: RandomSafetyAssessment | None = None,
) -> tuple[float, ...]:
    """为一次性执行稳定排序，理想几何优先但不淘汰退化候选。"""
    interface_gaps = [
        float(metric.nominal_normal_gap_mm)
        for metric in clearance.interface_metrics
    ]
    rectangle_violation = (
        max(0.0, TARGET_WIDTH_MIN_MM - float(candidate.width))
        + max(
            0.0,
            float(candidate.width) - RANDOM_TARGET_LONG_SIDE_MAX_MM,
        )
        + max(0.0, TARGET_HEIGHT_MIN_MM - float(candidate.height))
        + max(
            0.0,
            float(candidate.height) - RANDOM_TARGET_SHORT_SIDE_MAX_MM,
        )
    )
    admission = assessment or getattr(candidate, "admissibility", None)
    admission_rank = (
        admission.rank_key()
        if admission is not None
        else (1.0, 1.0, 1.0, 0.0, float("inf"), float("inf"), float("inf"))
    )
    return (
        *admission_rank,
        0.0 if topology_complete else 1.0,
        0.0 if not mirrored else 1.0,
        float(rectangle_violation),
        float(candidate.geometry_score),
        -float(candidate.appearance_score),
        0.0 if machine_executable else 1.0,
        float(clearance.estimated_physical_overlap_area_mm2),
        float(clearance.max_corresponding_vertex_distance_mm),
        max(interface_gaps, default=0.0),
        sum(interface_gaps),
        float(clearance.nonadjacent_robust_buffer_overlap_area_mm2),
        0.0 if clearance.feasible else 1.0,
        float(clearance.robust_buffer_overlap_area_mm2),
        float(clearance.total_translation_mm),
        float(candidate.total),
        0.0 if variant_id == "optimized" else 1.0,
    )


@dataclass
class RecoveryOutcome:
    selected: EvaluatedVariant | None
    diagnostic: EvaluatedVariant | None
    candidates: list[RandomCandidate]
    diagnostics: dict[str, Any]
    recovery: dict[str, Any]
    anchor_scores: list[dict[str, Any]]
    safe_candidates: list[EvaluatedVariant]


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low <= 1e-12:
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]


def compute_anchor_scores(pieces: list[ProcessedPiece]) -> list[dict[str, Any]]:
    """Return deterministic, normalized anchor quality components."""
    descriptors: list[np.ndarray] = []
    for piece in pieces:
        lengths = np.sort(piece.edge_lengths / max(np.sum(piece.edge_lengths), 1e-9))
        padded = np.pad(lengths, (0, max(0, 5 - len(lengths))))[:5]
        angles = np.sort(piece.internal_angles / 180.0)
        padded_angles = np.pad(angles, (0, max(0, 5 - len(angles))))[:5]
        descriptors.append(np.concatenate((padded, padded_angles)))
    distinctiveness = []
    for index, descriptor in enumerate(descriptors):
        distances = [
            float(np.linalg.norm(descriptor - other))
            for other_index, other in enumerate(descriptors)
            if other_index != index
        ]
        distinctiveness.append(min(distances, default=0.0))
    interface_margin = [
        float(np.std(piece.edge_lengths) / max(np.mean(piece.edge_lengths), 1e-9))
        for piece in pieces
    ]
    confidence = [
        float(np.clip(
            min(piece.piece_confidence, float(np.mean(piece.vertex_confidence))),
            0.0,
            1.0,
        ))
        for piece in pieces
    ]
    constraint_count = [
        float(np.clip((len(piece.vertices) - 2.0) / 3.0, 0.0, 1.0))
        for piece in pieces
    ]
    areas = [float(piece.area) for piece in pieces]
    asymmetry = []
    for piece in pieces:
        normalized = piece.edge_lengths / max(np.mean(piece.edge_lengths), 1e-9)
        repeats = sum(
            abs(float(first - second)) <= 0.03
            for i, first in enumerate(normalized)
            for second in normalized[i + 1:]
        )
        possible = max(1, len(normalized) * (len(normalized) - 1) // 2)
        asymmetry.append(1.0 - repeats / possible)
    normalized_components = {
        "shape_distinctiveness": _minmax(distinctiveness),
        "interface_margin": _minmax(interface_margin),
        "observation_confidence": confidence,
        "constraint_count": constraint_count,
        "area_stability": _minmax(areas),
        "asymmetry": [float(np.clip(value, 0.0, 1.0)) for value in asymmetry],
    }
    weights = {
        "shape_distinctiveness": 0.30,
        "interface_margin": 0.25,
        "observation_confidence": 0.15,
        "constraint_count": 0.10,
        "area_stability": 0.10,
        "asymmetry": 0.10,
    }
    rows = []
    for index, piece in enumerate(pieces):
        components = {
            name: float(values[index])
            for name, values in normalized_components.items()
        }
        rows.append({
            "anchor_index": index,
            "piece_id": piece.piece_id,
            "anchor_score": float(sum(
                weights[name] * components[name] for name in weights
            )),
            "anchor_score_components": components,
        })
    return sorted(
        rows,
        key=lambda row: (-row["anchor_score"], row["piece_id"]),
    )


def _raw_candidate_signature(candidate: RandomCandidate) -> tuple[Any, ...]:
    """Conservative raw-candidate key; variants are deliberately excluded."""
    poses = sorted(candidate.poses, key=lambda pose: pose.piece_id)
    base = np.mean(poses[0].target_vertices_local, axis=0)
    geometry = tuple(
        (
            pose.piece_id,
            round(float(pose.angle), 8),
            tuple(
                tuple(round(float(value), 8) for value in point - base)
                for point in pose.target_vertices_local
            ),
        )
        for pose in poses
    )
    interfaces = tuple(sorted(item.interface_id for item in candidate.interfaces))
    return geometry, interfaces


def _candidate_records(
    candidate: RandomCandidate,
    pieces: list[ProcessedPiece],
) -> list[dict[str, Any]]:
    by_id = {piece.piece_id: piece for piece in pieces}
    return [
        {
            "piece_id": pose.piece_id,
            "observed": by_id[pose.piece_id].vertices,
            "target": np.asarray(pose.target_vertices_local, dtype=float),
            "angle": float(pose.angle),
            "translation": np.asarray(pose.translation, dtype=float),
            "top_edge_index": pose.top_edge_index,
            "matched_prev_edge": pose.matched_prev_edge,
            "matched_next_edge": pose.matched_next_edge,
            "piece_confidence": by_id[pose.piece_id].piece_confidence,
        }
        for pose in candidate.poses
    ]


def _shifted_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    points = np.vstack([record["target"] for record in records])
    bounds = (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )
    offset = target_offset(bounds)
    shifted = []
    for record in records:
        item = dict(record)
        item["target"] = np.asarray(record["target"], dtype=float) + offset
        item["translation"] = (
            np.asarray(record["translation"], dtype=float) + offset
        )
        shifted.append(item)
    return shifted, np.asarray(offset, dtype=float)


def _revalidate_variant_geometry(
    candidate: RandomCandidate,
    records: list[dict[str, Any]],
    pieces: list[ProcessedPiece],
    profile: ObservationUncertaintyProfile,
    config: RuntimeConfig,
):
    """优化改变位姿后用随机专用新安全门重新审核完整几何。"""

    return assess_random_assembly(
        {
            str(record["piece_id"]): np.asarray(record["target"], dtype=float)
            for record in records
        },
        profile,
        config,
        expected_piece_count=len(pieces),
        declared_interface_count=len(candidate.interfaces),
        stage_name="post_optimization_revalidation",
    )


def _piece_uncertainties(
    records: list[dict[str, Any]],
    profile: ObservationUncertaintyProfile,
    edge_rms: float,
) -> dict[str, float]:
    result = {}
    rotation_std = max(0.05, float(edge_rms) / 20.0)
    translation_std = max(
        0.1,
        float(edge_rms),
        float(profile.typical_sigma_mm or 0.0),
    )
    for record in records:
        observed = np.asarray(record["observed"], dtype=float)
        center = np.mean(observed, axis=0)
        radius = max(
            (float(np.linalg.norm(point - center)) for point in observed),
            default=0.0,
        )
        rotation_displacement = radius * abs(sin(radians(rotation_std)))
        result[record["piece_id"]] = float(
            profile.hard_error_radius_mm
            + translation_std
            + rotation_displacement
        )
    return result


def evaluate_candidate_for_execution(
    coordinator: RecoveryCoordinator,
    candidate: RandomCandidate,
    candidate_id: str,
    variant_id: str,
    records: list[dict[str, Any]],
    width: float,
    height: float,
    optimization: dict[str, Any],
    uncertainty_profile: ObservationUncertaintyProfile,
    pieces: list[ProcessedPiece],
    *,
    validate_motion: bool = True,
) -> EvaluatedVariant:
    started = coordinator.monotonic_fn()
    block_reasons: list[str] = []
    if len(records) < 2 or any(
        not Polygon(record["target"]).is_valid for record in records
    ):
        block_reasons.append("invalid_candidate_geometry")
    if not (
        TARGET_WIDTH_MIN_MM - 1e-5
        <= width
        <= TARGET_WIDTH_MAX_MM + 1e-5
    ):
        block_reasons.append("invalid_target_width")
    if not (
        TARGET_HEIGHT_MIN_MM - 1e-5
        <= height
        <= TARGET_HEIGHT_MAX_MM + 1e-5
    ):
        block_reasons.append("invalid_target_height")
    geometry_admission = _revalidate_variant_geometry(
        candidate,
        records,
        pieces,
        uncertainty_profile,
        coordinator.config,
    )
    block_reasons.extend(geometry_admission.rejection_reasons)
    coordinator.predictor.record(
        "quick_geometry", coordinator.monotonic_fn() - started
    )

    shifted, _offset = _shifted_records(records)
    shifted_union = unary_union([
        Polygon(record["target"]) for record in shifted
    ])
    union_bounds = shifted_union.bounds
    bounding_area = max(
        1e-9,
        float(
            (union_bounds[2] - union_bounds[0])
            * (union_bounds[3] - union_bounds[1])
        ),
    )
    layout_fill_ratio = float(shifted_union.area / bounding_area)
    if layout_fill_ratio + 1e-9 < 0.95:
        block_reasons.append("low_rectangle_fill_ratio")
    clearance_started = coordinator.monotonic_fn()
    clearance = plan_random_clearance(
        {
            str(record["piece_id"]): np.asarray(record["target"], dtype=float)
            for record in shifted
        },
        geometry_admission.topology,
        target_gap_mm=coordinator.config.target_seam_gap_mm,
        minimum_gap_mm=coordinator.config.minimum_target_seam_gap_mm,
        maximum_gap_mm=coordinator.config.maximum_target_seam_gap_mm,
        maximum_vertex_distance_mm=coordinator.config.maximum_vertex_distance_mm,
        measurement_error_radius_mm=uncertainty_profile.hard_error_radius_mm,
        machine_position_margin_mm=coordinator.config.machine_position_margin_mm,
        overlap_tolerance_mm2=coordinator.config.overlap_numerical_epsilon_mm2,
    )
    coordinator.predictor.record(
        "clearance", coordinator.monotonic_fn() - clearance_started
    )
    uncertainty = _piece_uncertainties(
        shifted, uncertainty_profile, float(candidate.edge_rms)
    )
    distance_metrics = vertex_distance_metrics(
        clearance.max_corresponding_vertex_distance_mm,
        uncertainty_profile,
        target_gap_mm=clearance.requested_clearance_mm,
        machine_position_margin_mm=(
            coordinator.config.machine_position_margin_mm
        ),
        pose_residual_mm=float(candidate.edge_rms),
        competition_limit_mm=(
            coordinator.config.maximum_vertex_distance_mm
        ),
    )
    robust_max = distance_metrics.worst_case_robust_max_mm
    if not clearance.feasible:
        block_reasons.append("clearance_infeasible")
    if (
        clearance.estimated_physical_overlap_area_mm2
        > coordinator.config.overlap_numerical_epsilon_mm2
    ):
        block_reasons.append("post_clearance_overlap")
    # 缓冲区交叠只表示误差风险，不再否决名义几何已经无重叠的执行位姿。
    predicted_max = max(
        3.0 * float(candidate.edge_rms) + 2.0,
        2.0 * float(uncertainty_profile.typical_sigma_mm or 0.0),
    )
    pose_confidence = float(np.clip(
        1.0 - predicted_max / coordinator.config.maximum_vertex_distance_mm,
        0.0,
        1.0,
    ))
    if pose_confidence < 0.70:
        block_reasons.append("execution_confidence_too_low")

    planned_records = []
    for record, delta, planned in zip(
        shifted, clearance.translations_mm, clearance.planned_vertices
    ):
        item = dict(record)
        item["translation"] = (
            np.asarray(record["translation"], dtype=float)
            + np.asarray(delta, dtype=float)
        )
        item["target"] = np.asarray(planned, dtype=float)
        planned_records.append(item)
    if validate_motion:
        motion_started = coordinator.monotonic_fn()
        moves = build_moves(planned_records, confidence=1.0)
        planned_by_id = {
            record["piece_id"]: record for record in planned_records
        }
        for move in moves:
            if not point_inside_a4(move.pick_xy_a4_mm):
                block_reasons.append("invalid_pick_point")
            if not point_inside_a4(
                move.place_xy_a4_mm, lower_half=True
            ):
                block_reasons.append("invalid_place_point")
            if not pick_place_closes(
                move, planned_by_id[move.piece_id], tolerance_mm=1e-7
            ):
                block_reasons.append("pick_place_closure_failed")
        block_reasons.extend(validate_transport_paths(
            planned_records,
            moves,
            coordinator.config.transport_collision_mode,
            overlap_epsilon_mm2=(
                coordinator.config.overlap_numerical_epsilon_mm2
            ),
        ))
        coordinator.predictor.record(
            "motion", coordinator.monotonic_fn() - motion_started
        )
    machine_only_reasons = {
        "invalid_candidate_geometry",
        "invalid_pick_point",
        "invalid_place_point",
        "pick_place_closure_failed",
        "incomplete_motion_plan",
        "nonfinite_target_coordinate",
        "target_outside_machine_workspace",
        "transport_path_collision",
        "transport_workspace_violation",
    }
    machine_executable = not any(
        reason in machine_only_reasons for reason in block_reasons
    )
    quality = rank_candidate_for_execution_policy(
        candidate=candidate,
        variant_id=variant_id,
        clearance=clearance,
        machine_executable=machine_executable,
        topology_complete=len(records) == len(candidate.poses),
        assessment=geometry_admission,
    )
    return EvaluatedVariant(
        candidate=candidate,
        candidate_id=candidate_id,
        variant_id=variant_id,
        records=records,
        width=float(width),
        height=float(height),
        optimization=optimization,
        clearance=clearance,
        nominal_max_vertex_distance_mm=float(
            clearance.max_corresponding_vertex_distance_mm
        ),
        estimated_risk_vertex_distance_mm=float(
            distance_metrics.estimated_risk_max_mm
        ),
        robust_max_vertex_distance_mm=float(robust_max),
        piece_vertex_uncertainty_mm=uncertainty,
        machine_executable=machine_executable,
        safe=not block_reasons,
        block_reasons=sorted(set(block_reasons)),
        quality_key=quality,
        geometry_admissibility=geometry_admission.as_json(),
    )


def recover_random_candidate(
    pieces: list[ProcessedPiece],
    noise_mm: float,
    top_n: int,
    appearance_edge_matches: list[dict] | None,
    config: RuntimeConfig,
    *,
    started: float,
    stage_timer: Any | None = None,
    monotonic_fn: Callable[[], float] = perf_counter,
    surface_mode: str = "white",
    uncertainty_profile: ObservationUncertaintyProfile | None = None,
    rigid_backend=None,
) -> RecoveryOutcome:
    """Search progressively without using a generator noise label as control."""
    profile = uncertainty_profile or ObservationUncertaintyProfile(
        DEFAULT_HARD_ERROR_RADIUS_MM,
        None,
        None,
        None,
        None,
        "fallback_from_candidate_statistics",
        "medium",
        False,
    )
    coordinator = RecoveryCoordinator(
        started=started,
        config=config,
        surface_mode=surface_mode,
        monotonic_fn=monotonic_fn,
    )
    anchor_scores = compute_anchor_scores(pieces)
    default_anchor = 0
    anchor_order = [default_anchor] + [
        row["anchor_index"] for row in anchor_scores
        if row["anchor_index"] != default_anchor
    ]
    all_candidates: list[RandomCandidate] = []
    recovery_candidate_pool = RandomCandidatePool(
        config.recovery_candidate_limit,
        config.max_variants_per_discrete_signature,
    )
    candidate_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    per_anchor_stats: list[tuple[SearchRunStats, dict[str, Any]]] = []
    candidate_counter = 0
    best_diagnostic: EvaluatedVariant | None = None
    cache_hits = 0
    optimized_candidate_count = 0
    stage_optimized_count = 0
    current_stage = stages_from_config(config)[0]
    best_unsafe_key: tuple[float, ...] | None = None
    meaningful_improvement = False
    termination_reason = "search_exhausted"

    def evaluate(candidate: RandomCandidate) -> bool:
        nonlocal candidate_counter, best_diagnostic, cache_hits
        nonlocal optimized_candidate_count, stage_optimized_count
        nonlocal best_unsafe_key, meaningful_improvement, termination_reason
        nonlocal all_candidates
        signature = _raw_candidate_signature(candidate)
        cached = candidate_cache.get(signature)
        if cached is not None:
            cache_hits += 1
            if (
                not cached["optimized"]
                and stage_optimized_count
                < current_stage.optimize_candidate_limit
            ):
                candidate = cached["candidate"]
            else:
                return coordinator.should_continue_candidates()
        else:
            if not recovery_candidate_pool.consider(candidate):
                return coordinator.should_continue_candidates()
            candidate_counter += 1
            cached = {
                "candidate": candidate,
                "candidate_id": f"candidate_{candidate_counter:03d}",
                "raw": None,
                "optimized": False,
            }
            candidate_cache[signature] = cached
            all_candidates = recovery_candidate_pool.items()
        candidate_id = str(cached["candidate_id"])
        attempt_variants: list[dict[str, Any]] = []

        if cached["raw"] is None:
            if not coordinator.can_start(
                "quick_geometry", "clearance", "motion"
            ):
                termination_reason = "insufficient_validation_budget"
                return False
            raw_records = _candidate_records(candidate, pieces)
            raw_evaluated = evaluate_candidate_for_execution(
                coordinator,
                candidate,
                candidate_id,
                "raw",
                raw_records,
                float(candidate.width),
                float(candidate.height),
                {
                    "optimization_success": True,
                    "optimization_accepted": False,
                    "nfev": 0,
                    "skipped_reason": "raw_variant",
                },
                profile,
                pieces,
            )
            cached["raw"] = raw_evaluated
            coordinator.consider(raw_evaluated)
            attempt_variants.append({
                "variant_id": "raw",
                "safe": raw_evaluated.safe,
                "block_reasons": raw_evaluated.block_reasons,
                "geometry_admissibility": (
                    raw_evaluated.geometry_admissibility
                ),
                "nominal_max_vertex_distance_mm": (
                    raw_evaluated.nominal_max_vertex_distance_mm
                ),
                "robust_max_vertex_distance_mm": (
                    raw_evaluated.robust_max_vertex_distance_mm
                ),
            })
            if stage_timer is not None:
                stage_timer.add("raw_geometry_check", 0.0)
            if (
                best_diagnostic is None
                or raw_evaluated.quality_key
                < best_diagnostic.quality_key
            ):
                best_diagnostic = raw_evaluated
            unsafe_key = (
                float(len(raw_evaluated.block_reasons)),
                *raw_evaluated.quality_key,
            )
            if best_unsafe_key is None:
                best_unsafe_key = unsafe_key
            elif unsafe_key < best_unsafe_key:
                if (
                    best_unsafe_key[0] - unsafe_key[0] >= 1
                    or best_unsafe_key[1] - unsafe_key[1]
                    >= config.futility_min_candidate_improvement
                ):
                    meaningful_improvement = True
                best_unsafe_key = unsafe_key
            if raw_evaluated.safe and surface_mode == "white":
                termination_reason = "safe_raw_candidate"
        else:
            raw_evaluated = cached["raw"]

        # Optimization is quality-driven: a generator noise label never enables it.
        repairable_reasons = {
            "clearance_infeasible",
            "post_clearance_overlap",
            "nonadjacent_robust_buffer_overlap",
            "robust_vertex_distance_exceeded",
            "execution_confidence_too_low",
            "invalid_target_width",
            "invalid_target_height",
            "low_rectangle_fill_ratio",
            "rectangle_axis_inconsistent",
            "outer_side_role_mismatch",
            "junction_angle_inconsistent",
            "rectangle_iou_too_low",
            "excessive_uncovered_area",
            "illegal_target_dimensions",
            "piece_without_target_outer_edge",
        }
        near_safe = (
            not raw_evaluated.safe
            and bool(raw_evaluated.block_reasons)
            and set(raw_evaluated.block_reasons) <= repairable_reasons
        )
        if (
            near_safe
            and not cached["optimized"]
            and config.enable_global_optimization
            and stage_optimized_count
            < current_stage.optimize_candidate_limit
            and coordinator.can_start(
                "soft_l1", "quick_geometry", "clearance", "motion"
            )
        ):
            cached["optimized"] = True
            stage_optimized_count += 1
            optimized_candidate_count += 1
            optimization_started = monotonic_fn()
            raw_records = _candidate_records(candidate, pieces)
            typical_sigma = float(profile.typical_sigma_mm or 0.0)
            discovered = discover_interfaces_in_placed_state(
                {
                    record["piece_id"]: record["target"]
                    for record in raw_records
                },
                tolerance_mm=max(
                    1.0, profile.hard_error_radius_mm + typical_sigma
                ),
                length_epsilon_mm=max(
                    1e-6, min(profile.hard_error_radius_mm, 1.0)
                ),
                parallel_tolerance=max(
                    0.01, 0.01 + 0.01 * profile.hard_error_radius_mm
                ),
            )
            optimized, width, height, metadata = optimize_generic_candidate(
                raw_records,
                candidate.width,
                candidate.height,
                candidate.matches,
                candidate.interfaces,
                discovered.get("interfaces", []),
                dimension_bounds=((90.0, 50.0), (120.0, 90.0))
                    if config.random_competition_rules else None,
                max_nfev_override=max(
                    20,
                    min(
                        800,
                        800 if config.unlimited_search else int(400.0 * coordinator.remaining_solve_seconds()),
                    ),
                ),
            )
            elapsed_optimization = monotonic_fn() - optimization_started
            coordinator.predictor.record("soft_l1", elapsed_optimization)
            if stage_timer is not None:
                stage_timer.add("soft_l1", elapsed_optimization)
            optimized_evaluated = evaluate_candidate_for_execution(
                coordinator,
                candidate,
                candidate_id,
                "optimized",
                optimized,
                width,
                height,
                metadata,
                profile,
                pieces,
            )
            coordinator.consider(optimized_evaluated)
            attempt_variants.append({
                "variant_id": "optimized",
                "safe": optimized_evaluated.safe,
                "block_reasons": optimized_evaluated.block_reasons,
                "geometry_admissibility": (
                    optimized_evaluated.geometry_admissibility
                ),
                "nominal_max_vertex_distance_mm": (
                    optimized_evaluated.nominal_max_vertex_distance_mm
                ),
                "robust_max_vertex_distance_mm": (
                    optimized_evaluated.robust_max_vertex_distance_mm
                ),
            })
            if (
                best_diagnostic is None
                or optimized_evaluated.quality_key
                < best_diagnostic.quality_key
            ):
                best_diagnostic = optimized_evaluated
            if optimized_evaluated.safe and surface_mode == "white":
                termination_reason = "safe_optimized_candidate"

        if attempt_variants:
            coordinator.attempts.append({
                "candidate_id": candidate_id,
                "search_stage": current_stage.name,
                "variants": attempt_variants,
            })
        return coordinator.should_continue_candidates()

    progressive_stages = stages_from_config(config)
    # 新反向分割解析器不依赖锚片，重复 fast/expanded/多锚运行只会从头
    # 枚举同一批解析树并吃掉总时限。一次 recovery 级运行获得完整预算，
    # 其内部自行执行“严格边链 -> T 结点区间”的渐进开放。
    runs: list[tuple[ProgressiveSearchStage, int]] = [
        (progressive_stages[2], default_anchor)
    ]
    for run_index, (search_stage, anchor_index) in enumerate(runs):
        if monotonic_fn() >= coordinator.solve_deadline:
            termination_reason = "solve_deadline"
            break
        if coordinator.incumbent is not None and surface_mode == "white":
            break
        if (
            search_stage.name == "recovery"
            and coordinator.remaining_solve_seconds()
            < config.recovery_min_deep_search_seconds
        ):
            termination_reason = "deep_search_reserve_unavailable"
            break
        current_stage = search_stage
        stage_optimized_count = 0
        if search_stage.name == "fast":
            deadline = coordinator.initial_stage_deadline
        elif search_stage.name == "expanded":
            deadline = min(
                coordinator.solve_deadline,
                monotonic_fn()
                + max(
                    config.recovery_min_deep_search_seconds,
                    0.5 * coordinator.remaining_solve_seconds(),
                ),
            )
        else:
            # Anchor alternatives are already ordered by normalized potential.
            # Give the strongest one the remaining deep-search budget; a
            # shallow equal split would prevent every hard branch completing.
            deadline = coordinator.solve_deadline
        if config.unlimited_search:
            deadline = None
        diagnostics: dict[str, Any] = {}
        stats = stream_random_candidates(
            pieces,
            noise_mm,
            min(top_n, search_stage.candidate_limit),
            appearance_edge_matches,
            anchor_index=anchor_index,
            deadline=deadline,
            runtime_config=config,
            rigid_backend=rigid_backend,
            on_candidate=evaluate,
            diagnostics=diagnostics,
            stage_timer=stage_timer,
            uncertainty_profile=profile,
            progressive_stage=search_stage,
        )
        per_anchor_stats.append((stats, diagnostics))
        coordinator.attempts.append({
            "stage": search_stage.name,
            "anchor_index": anchor_index,
            "piece_id": pieces[anchor_index].piece_id,
            "candidate_count": stats.candidate_count,
            "search_nodes": stats.search_nodes,
            "hard_interface_pool_count": diagnostics.get(
                "hard_interface_pool_count", 0
            ),
            "opened_interface_count": diagnostics.get(
                "opened_interface_count", 0
            ),
            "budget_exhausted": stats.budget_exhausted,
            "wall_time_limited": stats.wall_time_limited,
            "elapsed_seconds": stats.elapsed_seconds,
        })
        if (
            config.enable_futility_detection
            and search_stage.name == "expanded"
            and coordinator.incumbent is None
            and not meaningful_improvement
            and not all_candidates
            and profile.quality_level == "low"
        ):
            termination_reason = "futility_detected"
            break
        if (
            coordinator.incumbent is not None
            and (
                surface_mode == "white"
                or len(coordinator.safe_candidates)
                >= config.max_appearance_candidates
            )
        ):
            break

    elapsed = monotonic_fn() - started
    selected = coordinator.incumbent
    status = (
        "recovered"
        if selected is not None
        else (
            "deadline_exhausted"
            if monotonic_fn() >= coordinator.solve_deadline
            else "no_safe_candidate"
        )
    )
    combined_diagnostics: dict[str, Any] = {
        "recovery_enabled": True,
        "uncertainty_profile": profile.as_json(),
        "legacy_noise_label_present": noise_mm is not None,
        "legacy_noise_label_used_by_solver": False,
        "progressive_stage_count": len(per_anchor_stats),
        "anchor_scores": anchor_scores,
        "anchor_attempt_count": len(per_anchor_stats),
        "search_nodes": sum(stats.search_nodes for stats, _ in per_anchor_stats),
        "candidate_count": len(all_candidates),
        "search_statistics": {
            "search_state_count": sum(
                int(diagnostics.get("search_state_count", 0))
                for _, diagnostics in per_anchor_stats
            ),
            "fully_placed_state_count": sum(
                int(diagnostics.get("fully_placed_state_count", 0))
                for _, diagnostics in per_anchor_stats
            ),
            "fully_placed_rejected_count": sum(
                int(diagnostics.get("fully_placed_rejected_count", 0))
                for _, diagnostics in per_anchor_stats
            ),
            "geometry_candidate_count": sum(
                int(diagnostics.get("geometry_candidate_count", 0))
                for _, diagnostics in per_anchor_stats
            ),
            "geometry_candidate_pool_size": len(all_candidates),
            "optimized_candidate_count": optimized_candidate_count,
            "safe_candidate_count": sum(
                item.safe for item in coordinator.safe_candidates
            ),
            "machine_executable_candidate_count": len(
                coordinator.safe_candidates
            ),
            "pool_replacements": (
                recovery_candidate_pool.replacements
            ),
            "pool_duplicates": recovery_candidate_pool.duplicates,
            "pool_ever_full": bool(
                recovery_candidate_pool.ever_full
                or any(
                    diagnostics.get("candidate_pool_ever_full", False)
                    for _, diagnostics in per_anchor_stats
                )
            ),
            "search_continued_after_pool_full": bool(
                recovery_candidate_pool.considered_after_full > 0
                or any(
                    diagnostics.get(
                        "search_continued_after_pool_full", False
                    )
                    for _, diagnostics in per_anchor_stats
                )
            ),
        },
        "geometry_candidate_rejection_reasons": dict(sorted(
            sum(
                (
                    Counter(
                        diagnostics.get(
                            "geometry_candidate_rejection_reasons", {}
                        )
                    )
                    for _, diagnostics in per_anchor_stats
                ),
                Counter(),
            ).items()
        )),
        "geometry_candidate_rejection_samples": [
            sample
            for _, diagnostics in per_anchor_stats
            for sample in diagnostics.get(
                "geometry_candidate_rejection_samples", []
            )
        ][:config.diagnostic_rejection_sample_limit],
        "optimized_candidate_count": optimized_candidate_count,
        "candidate_cache_hits": cache_hits,
        "termination_reason": termination_reason,
        "search_budget_limited": bool(
            any(
                stats.budget_exhausted or stats.wall_time_limited
                for stats, _ in per_anchor_stats
            )
        ),
        "search_level_reached": max(
            (
                int(diagnostics.get("search_level_reached", 0))
                for _, diagnostics in per_anchor_stats
            ),
            default=0,
        ),
        "stage_runs": [
            {
                "anchor_index": stats.anchor_index,
                "candidate_count": stats.candidate_count,
                "elapsed_seconds": stats.elapsed_seconds,
                "stage": diagnostics.get("progressive_stage", {}).get(
                    "name", "unknown"
                ),
                "hard_interface_pool_count": diagnostics.get(
                    "hard_interface_pool_count", 0
                ),
                "opened_interface_count": diagnostics.get(
                    "opened_interface_count", 0
                ),
            }
            for stats, diagnostics in per_anchor_stats
        ],
        "generic_partial_fallback_used": any(
            bool(diagnostics.get("generic_partial_fallback_used", False))
            for _, diagnostics in per_anchor_stats
        ),
        "expanded_states": sum(
            int(diagnostics.get("expanded_states", stats.search_nodes))
            for stats, diagnostics in per_anchor_stats
        ),
        "generated_states": sum(
            int(diagnostics.get("generated_states", 0))
            for _, diagnostics in per_anchor_stats
        ),
        "duplicate_pruned": sum(
            int(diagnostics.get("duplicate_pruned", 0))
            for _, diagnostics in per_anchor_stats
        ),
        "rectangle_pruned": sum(
            int(diagnostics.get("rectangle_pruned", 0))
            for _, diagnostics in per_anchor_stats
        ),
        "overlap_pruned": sum(
            int(diagnostics.get("overlap_pruned", 0))
            for _, diagnostics in per_anchor_stats
        ),
        "cycle_pruned": sum(
            int(diagnostics.get("cycle_pruned", 0))
            for _, diagnostics in per_anchor_stats
        ),
        "interface_conflict_pruned": sum(
            int(diagnostics.get("interface_conflict_pruned", 0))
            for _, diagnostics in per_anchor_stats
        ),
    }
    interface_candidates: dict[str, dict[str, Any]] = {}
    for _, diagnostics in per_anchor_stats:
        for item in diagnostics.get("interface_candidates", []):
            interface_candidates[str(item.get("interface_id"))] = item
    combined_diagnostics["interface_candidates"] = list(
        interface_candidates.values()
    )
    recovery = {
        "enabled": True,
        "triggered": len(per_anchor_stats) > 1 or len(all_candidates) > 1,
        "status": status,
        "termination_reason": termination_reason,
        "deadline_seconds": None if config.unlimited_search else config.recovery_total_deadline_seconds,
        "solve_deadline_seconds": (
            None if config.unlimited_search else config.recovery_total_deadline_seconds
            - config.recovery_finalization_reserve_seconds
            - (
                config.appearance_resolution_reserve_seconds
                if surface_mode == "patterned" else 0.0
            )
        ),
        "elapsed_seconds": float(elapsed),
        "deadline_overshoot_seconds": float(max(
            0.0, monotonic_fn() - coordinator.total_deadline
        )),
        "longest_noninterruptible_call_seconds": (
            coordinator.predictor.longest_noninterruptible_call_seconds
        ),
        "attempt_count": len(coordinator.attempts),
        "selected_attempt": (
            selected.candidate_id if selected is not None else None
        ),
        "selected_variant": (
            selected.variant_id if selected is not None else None
        ),
        # 保留旧字段名以兼容诊断消费者；池内现表示“机械可执行且已按
        # 几何质量排序”，每项的 safe 字段说明是否达到理想门槛。
        "safe_candidate_pool": [
            {
                "candidate_id": item.candidate_id,
                "variant_id": item.variant_id,
                "quality_key": list(item.quality_key),
                "safe": item.safe,
                "machine_executable": item.machine_executable,
            }
            for item in coordinator.safe_candidates
        ],
        "attempts": coordinator.attempts,
    }
    return RecoveryOutcome(
        selected=selected,
        diagnostic=best_diagnostic,
        candidates=list(all_candidates),
        diagnostics=combined_diagnostics,
        recovery=recovery,
        anchor_scores=anchor_scores,
        safe_candidates=list(coordinator.safe_candidates),
    )
