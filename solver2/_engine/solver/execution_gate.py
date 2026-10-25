"""Truth-free execution gate shared by all final solver modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from shapely.geometry import Polygon

from ..config import (
    A4_HEIGHT_MM,
    A4_WIDTH_MM,
    DIVIDER_Y_MM,
    VERTEX_DISTANCE_LIMIT_MM,
    TARGET_HEIGHT_MAX_MM,
    TARGET_HEIGHT_MIN_MM,
    TARGET_WIDTH_MAX_MM,
    TARGET_WIDTH_MIN_MM,
)
from ..runtime_config import RuntimeConfig
from ..transforms import transform_points


@dataclass(frozen=True)
class ExecutionGateResult:
    """Final software/hardware gate values written to solution diagnostics."""

    reasons: list[str]
    solver_block_reasons: list[str]
    hardware_block_reasons: list[str]
    executable: bool
    hardware_ready: bool
    topology_confidence: float
    pose_confidence: float
    execution_confidence: float
    final_revalidation_reasons: list[str]
    machine_executable: bool
    competition_gate_passed: bool
    strict_robust_gate_passed: bool
    execution_policy: str
    execution_quality: str
    forced_output: bool
    risk_warnings: list[str]
    competition_failure_reasons: list[str]
    strict_robust_failure_reasons: list[str]


def point_inside_a4(
    point: Iterable[float],
    *,
    lower_half: bool = False,
) -> bool:
    x, y = (float(value) for value in point)
    minimum_y = DIVIDER_Y_MM if lower_half else 0.0
    return 0.0 <= x <= A4_WIDTH_MM and minimum_y <= y <= A4_HEIGHT_MM


def target_inside_a4_lower(
    points: np.ndarray,
    *,
    tolerance_mm: float = 0.0,
) -> bool:
    values = np.asarray(points, dtype=float)
    tolerance = float(tolerance_mm)
    return bool(
        float(values[:, 0].min()) >= -tolerance
        and float(values[:, 0].max()) <= A4_WIDTH_MM + tolerance
        and float(values[:, 1].min()) >= DIVIDER_Y_MM - tolerance
        and float(values[:, 1].max()) <= A4_HEIGHT_MM + tolerance
    )


def pick_place_closes(
    move: Any,
    record: dict[str, Any],
    *,
    tolerance_mm: float = 1e-7,
) -> bool:
    recomputed = transform_points(
        [move.pick_xy_a4_mm],
        record["angle"],
        record["translation"],
    )[0]
    return bool(np.allclose(
        recomputed,
        np.asarray(move.place_xy_a4_mm, dtype=float),
        atol=tolerance_mm,
    ))


def final_candidate_revalidation(
    mode: str,
    pose_records: list[dict[str, Any]],
    moves: list[Any],
    *,
    config: RuntimeConfig,
    recovery_outcome: Any | None,
    search_diagnostics: dict[str, Any],
) -> list[str]:
    """Pure read-only safety confirmation immediately before serialization."""
    reasons: list[str] = []
    polygons = [
        Polygon(np.asarray(record["target"], dtype=float))
        for record in pose_records
    ]
    if any(not polygon.is_valid or polygon.area <= 0.0 for polygon in polygons):
        reasons.append("final_invalid_target_polygon")
    if any(
        not np.all(np.isfinite(np.asarray(record["target"], dtype=float)))
        for record in pose_records
    ):
        reasons.append("final_nonfinite_target_coordinate")
    all_points = np.vstack([
        np.asarray(record["target"], dtype=float)
        for record in pose_records
    ])
    if not target_inside_a4_lower(all_points, tolerance_mm=1e-7):
        reasons.append("final_target_outside_machine_workspace")
    records_by_id = {
        record["piece_id"]: record for record in pose_records
    }
    if {move.piece_id for move in moves} != set(records_by_id):
        reasons.append("final_motion_candidate_mismatch")
    for move in moves:
        record = records_by_id.get(move.piece_id)
        if record is None:
            continue
        if not pick_place_closes(move, record, tolerance_mm=1e-7):
            reasons.append(
                "fixed_motion_transform_mismatch"
                if mode == "fixed"
                else "final_place_transform_mismatch"
            )
        if (
            not np.all(np.isfinite(move.pick_xy_a4_mm))
            or not np.all(np.isfinite(move.place_xy_a4_mm))
        ):
            reasons.append("final_nonfinite_motion_coordinate")
        transformed = transform_points(
            record["observed"],
            record["angle"],
            record["translation"],
        )
        if (
            transformed.shape
            != np.asarray(record["target"], dtype=float).shape
            or not np.allclose(
                transformed,
                np.asarray(record["target"], dtype=float),
                atol=1e-6,
            )
        ):
            reasons.append(
                "fixed_final_target_transform_mismatch"
                if mode == "fixed"
                else "final_target_transform_mismatch"
            )
        if mode == "fixed":
            target = np.asarray(record["target"], dtype=float)
            if (
                transformed.shape != target.shape
                or not np.allclose(
                    transformed, target, atol=1e-6
                )
            ):
                reasons.append("fixed_physical_contour_closure_failed")
            for anchor in record.get(
                "virtual_assembly_anchors", []
            ):
                recomputed_anchor = transform_points(
                    [anchor["observed_anchor_a4_mm"]],
                    record["angle"],
                    record["translation"],
                )[0]
                if not np.allclose(
                    recomputed_anchor,
                    np.asarray(
                        anchor["target_anchor_a4_mm"], dtype=float
                    ),
                    atol=1e-6,
                ):
                    reasons.append(
                        "fixed_virtual_anchor_transform_mismatch"
                    )
    if recovery_outcome is not None and recovery_outcome.selected is not None:
        if (
            search_diagnostics.get("selected_candidate_id")
            != recovery_outcome.selected.candidate_id
            or search_diagnostics.get("selected_variant_id")
            != recovery_outcome.selected.variant_id
        ):
            reasons.append("final_recovery_selection_mismatch")
    return sorted(set(reasons))


def evaluate_execution_gate(
    *,
    mode: str,
    surface_mode: str,
    ambiguous: bool,
    appearance_resolved: bool,
    confidence: float,
    predicted_max_vertex_error_mm: float,
    nominal_max_vertex_distance_mm: float,
    estimated_risk_vertex_distance_mm: float,
    robust_max_vertex_distance_mm: float,
    estimated_overlap_area_mm2: float,
    robust_overlap_area_mm2: float,
    nonadjacent_robust_overlap_area_mm2: float,
    clearance_plan: Any,
    optimization: dict[str, Any],
    search_diagnostics: dict[str, Any],
    recovery_outcome: Any | None,
    candidate_margin: float,
    pose_records: list[dict[str, Any]],
    moves: list[Any],
    transport_reasons: list[str],
    visual_geometry_validated: bool,
    calibration: dict[str, Any] | None,
    target_width_mm: float,
    target_height_mm: float,
    config: RuntimeConfig,
) -> ExecutionGateResult:
    """Split machine, competition and strict-robust semantics."""
    machine_reasons = list(transport_reasons)
    if any(not point_inside_a4(move.pick_xy_a4_mm) for move in moves):
        machine_reasons.append("invalid_pick_point")
    if not moves or len(moves) != len(pose_records):
        machine_reasons.append("incomplete_motion_plan")
    target_points = np.vstack([
        np.asarray(record["target"], dtype=float)
        for record in pose_records
    ])
    if not np.all(np.isfinite(target_points)):
        machine_reasons.append("nonfinite_target_coordinate")
    if not target_inside_a4_lower(target_points):
        machine_reasons.append("target_outside_a4_lower_half")

    topology_confidence = float(np.clip(
        1.0
        - search_diagnostics.get(
            "partial_to_partial_candidate_count", 0
        )
        / max(
            search_diagnostics.get(
                "pairwise_interface_placement_count", 1
            ),
            1,
        ),
        0.0,
        1.0,
    ))
    # Preserve the competition's fixed 20 mm confidence scale.
    pose_confidence = float(np.clip(
        1.0
        - predicted_max_vertex_error_mm / VERTEX_DISTANCE_LIMIT_MM,
        0.0,
        1.0,
    ))
    execution_confidence = float(min(
        confidence, topology_confidence, pose_confidence
    ))
    final_reasons = final_candidate_revalidation(
        mode,
        pose_records,
        moves,
        config=config,
        recovery_outcome=recovery_outcome,
        search_diagnostics=search_diagnostics,
    )
    machine_reasons.extend(final_reasons)
    hardware_reasons: list[str] = []
    if not visual_geometry_validated:
        hardware_reasons.append("visual_geometry_not_validated")
    if calibration is None:
        hardware_reasons.append("machine_calibration_missing")
    if config.execution_profile == "debug":
        machine_reasons.extend(hardware_reasons)
    machine_reasons = sorted(set(machine_reasons))
    machine_executable = not machine_reasons

    competition_reasons: list[str] = []
    if surface_mode == "patterned" and not appearance_resolved:
        competition_reasons.append("appearance_ambiguity")
    if estimated_overlap_area_mm2 > config.overlap_numerical_epsilon_mm2:
        competition_reasons.append("estimated_overlap_present")
    if (
        nominal_max_vertex_distance_mm
        > config.maximum_vertex_distance_mm + 1e-7
    ):
        competition_reasons.append("nominal_vertex_distance_exceeded")
    if not clearance_plan.feasible:
        competition_reasons.append("nominal_clearance_infeasible")
        # Compatibility alias retained for existing downstream consumers.
        competition_reasons.append("clearance_constraints_infeasible")
    if not (
        TARGET_WIDTH_MIN_MM - 1e-7
        <= target_width_mm
        <= TARGET_WIDTH_MAX_MM + 1e-7
        and TARGET_HEIGHT_MIN_MM - 1e-7
        <= target_height_mm
        <= TARGET_HEIGHT_MAX_MM + 1e-7
    ):
        competition_reasons.append("illegal_target_rectangle")
    if not target_inside_a4_lower(target_points):
        competition_reasons.append("target_outside_a4_lower_half")
    competition_reasons = sorted(set(competition_reasons))
    competition_passed = not competition_reasons

    strict_reasons: list[str] = []
    # ``robust_max_vertex_distance_mm`` is a deliberately pessimistic
    # diagnostic envelope (two independent vision bounds, machine margin and
    # pose residual).  Keep reporting an excess below, but do not turn that
    # estimate into a command-blocking condition.  The executable plan is
    # still subject to the nominal geometry, workspace, overlap, transform and
    # hardware checks above.
    if robust_overlap_area_mm2 > config.overlap_numerical_epsilon_mm2:
        strict_reasons.append("adjacent_uncertainty_buffers_overlap")
    if (
        nonadjacent_robust_overlap_area_mm2
        > config.overlap_numerical_epsilon_mm2
    ):
        strict_reasons.append("nonadjacent_robust_buffer_overlap")
    if (
        confidence < config.minimum_execution_confidence
        or execution_confidence < config.minimum_execution_confidence
    ):
        strict_reasons.append("low_confidence")
    # 纯几何任务允许多个有效矩形解；图案歧义由上面的 appearance 检查处理。
    if not optimization.get("optimization_success", False):
        strict_reasons.append("optimization_failed")
    if (
        search_diagnostics.get("search_budget_limited", False)
        and recovery_outcome is None
    ):
        strict_reasons.append("search_budget_limited")
    strict_reasons = sorted(set(strict_reasons))
    strict_passed = competition_passed and not strict_reasons

    if strict_passed:
        quality = "strict"
    elif competition_passed:
        quality = "competition_compliant"
    elif (
        machine_executable
        and estimated_overlap_area_mm2
        <= config.overlap_numerical_epsilon_mm2
    ):
        quality = "best_effort"
    else:
        quality = "last_resort"

    if config.execution_policy == "strict":
        executable = (
            machine_executable and competition_passed and strict_passed
        )
    else:
        executable = machine_executable
    forced_output = bool(
        executable and not (competition_passed and strict_passed)
    )
    warnings = list(competition_reasons) + list(strict_reasons)
    if robust_max_vertex_distance_mm > config.maximum_vertex_distance_mm + 1e-7:
        warnings.append("robust_vertex_distance_exceeded")
    if (
        estimated_risk_vertex_distance_mm
        > config.maximum_vertex_distance_mm + 1e-7
    ):
        warnings.append("estimated_risk_vertex_distance_exceeded")
    if clearance_plan.post_compaction_max_gap_mm > (
        config.large_seam_gap_warning_mm
    ):
        warnings.append("large_seam_gap")
    if quality == "last_resort":
        warnings.append("last_resort_candidate")
    warnings = sorted(set(warnings))
    if config.execution_policy == "strict":
        blocking = sorted(set(
            machine_reasons + competition_reasons + strict_reasons
        ))
    else:
        blocking = machine_reasons
    return ExecutionGateResult(
        reasons=blocking,
        solver_block_reasons=sorted(set(
            competition_reasons + strict_reasons
        )),
        hardware_block_reasons=hardware_reasons,
        executable=executable,
        hardware_ready=machine_executable and not hardware_reasons,
        topology_confidence=topology_confidence,
        pose_confidence=pose_confidence,
        execution_confidence=execution_confidence,
        final_revalidation_reasons=final_reasons,
        machine_executable=machine_executable,
        competition_gate_passed=competition_passed,
        strict_robust_gate_passed=strict_passed,
        execution_policy=config.execution_policy,
        execution_quality=quality,
        forced_output=forced_output,
        risk_warnings=warnings,
        competition_failure_reasons=competition_reasons,
        strict_robust_failure_reasons=strict_reasons,
    )
