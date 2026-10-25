"""随机拼图专用的最终执行安全门。"""

from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import Polygon

from ..config import TARGET_HEIGHT_MIN_MM, TARGET_WIDTH_MIN_MM
from ..runtime_config import RuntimeConfig
from ..transforms import transform_points
from .execution_gate import (
    ExecutionGateResult,
    point_inside_a4,
    target_inside_a4_lower,
)
from .random_safety import (
    RANDOM_TARGET_LONG_SIDE_MAX_MM,
    RANDOM_TARGET_SHORT_SIDE_MAX_MM,
    RandomSafetyAssessment,
)


def _final_motion_reasons(
    pose_records: list[dict[str, Any]], moves: list[Any]
) -> list[str]:
    """只验证最终刚体变换、有限坐标和工作区，不重复旧几何门。"""

    reasons: list[str] = []
    records_by_id = {
        str(record["piece_id"]): record for record in pose_records
    }
    if len(moves) != len(pose_records) or {
        str(move.piece_id) for move in moves
    } != set(records_by_id):
        reasons.append("incomplete_motion_plan")
    for move in moves:
        record = records_by_id.get(str(move.piece_id))
        if record is None:
            continue
        target = np.asarray(record["target"], dtype=float)
        transformed = transform_points(
            record["observed"], record["angle"], record["translation"]
        )
        if (
            target.shape != transformed.shape
            or not np.allclose(target, transformed, atol=1e-6)
        ):
            reasons.append("final_target_transform_mismatch")
        if (
            not np.all(np.isfinite(target))
            or not np.all(np.isfinite(move.pick_xy_a4_mm))
            or not np.all(np.isfinite(move.place_xy_a4_mm))
        ):
            reasons.append("final_nonfinite_motion_coordinate")
        if not point_inside_a4(move.pick_xy_a4_mm):
            reasons.append("invalid_pick_point")
        if not point_inside_a4(move.place_xy_a4_mm, lower_half=True):
            reasons.append("invalid_place_point")
    return sorted(set(reasons))


def evaluate_random_execution_gate(
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
    random_assessment: RandomSafetyAssessment | None,
) -> ExecutionGateResult:
    """评估随机结果质量；只有机械无效条件可以阻止执行。"""

    del mode, predicted_max_vertex_error_mm, search_diagnostics
    del recovery_outcome, candidate_margin, optimization
    machine_reasons = list(transport_reasons)
    machine_reasons.extend(_final_motion_reasons(pose_records, moves))
    targets = [
        np.asarray(record["target"], dtype=float) for record in pose_records
    ]
    polygons = [Polygon(vertices) for vertices in targets]
    if any(
        not polygon.is_valid or polygon.is_empty or polygon.area <= 0.0
        for polygon in polygons
    ):
        machine_reasons.append("final_invalid_target_polygon")
    if targets and not target_inside_a4_lower(np.vstack(targets)):
        machine_reasons.append("target_outside_a4_lower_half")
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
    if random_assessment is None or not random_assessment.accepted:
        competition_reasons.extend(
            ["random_geometry_not_admitted"]
            if random_assessment is None
            else random_assessment.rejection_reasons
        )
    if random_assessment is not None and random_assessment.rectangle_fill_ratio + 1e-9 < 0.95:
        competition_reasons.append("rectangle_fill_below_minimum")
    if not clearance_plan.feasible:
        competition_reasons.append("nominal_clearance_infeasible")
    if estimated_overlap_area_mm2 > config.overlap_numerical_epsilon_mm2:
        competition_reasons.append("estimated_overlap_present")
    if nominal_max_vertex_distance_mm > config.maximum_vertex_distance_mm + 1e-7:
        competition_reasons.append("nominal_vertex_distance_exceeded")
    if not (
        TARGET_WIDTH_MIN_MM - 1e-7
        <= target_width_mm
        <= RANDOM_TARGET_LONG_SIDE_MAX_MM + 1e-7
        and TARGET_HEIGHT_MIN_MM - 1e-7
        <= target_height_mm
        <= RANDOM_TARGET_SHORT_SIDE_MAX_MM + 1e-7
    ):
        competition_reasons.append("illegal_target_rectangle")
    competition_reasons = sorted(set(competition_reasons))
    competition_passed = not competition_reasons

    warnings: list[str] = []
    if robust_overlap_area_mm2 > config.overlap_numerical_epsilon_mm2:
        warnings.append("adjacent_uncertainty_buffers_overlap")
    if nonadjacent_robust_overlap_area_mm2 > config.overlap_numerical_epsilon_mm2:
        warnings.append("nonadjacent_uncertainty_buffers_overlap")
    if robust_max_vertex_distance_mm > config.maximum_vertex_distance_mm + 1e-7:
        warnings.append("robust_vertex_distance_exceeded")
    if estimated_risk_vertex_distance_mm > config.maximum_vertex_distance_mm + 1e-7:
        warnings.append("estimated_risk_vertex_distance_exceeded")
    if confidence < config.minimum_execution_confidence:
        warnings.append("low_confidence")
    if ambiguous and surface_mode == "white":
        warnings.append("multiple_valid_white_geometry_solutions")
    strict_passed = competition_passed
    executable = bool(machine_executable and competition_passed and confidence >= config.minimum_execution_confidence)
    topology_confidence = float(
        0.0 if random_assessment is None
        else np.clip(random_assessment.union_rectangle_iou_estimate, 0.0, 1.0)
    )
    pose_confidence = float(np.clip(
        1.0 - nominal_max_vertex_distance_mm
        / max(config.maximum_vertex_distance_mm, 1e-9),
        0.0,
        1.0,
    ))
    execution_confidence = float(min(
        confidence, topology_confidence, pose_confidence
    ))
    if competition_passed:
        quality = "strict"
    elif estimated_overlap_area_mm2 <= config.overlap_numerical_epsilon_mm2:
        quality = "best_effort"
    else:
        quality = "last_resort"
    warnings.extend(competition_reasons)
    if quality == "last_resort":
        warnings.append("last_resort_candidate")
    blocking = sorted(set(machine_reasons + competition_reasons + (["low_confidence"] if confidence < config.minimum_execution_confidence else [])))
    return ExecutionGateResult(
        reasons=blocking,
        solver_block_reasons=competition_reasons,
        hardware_block_reasons=hardware_reasons,
        executable=executable,
        hardware_ready=machine_executable and not hardware_reasons,
        topology_confidence=topology_confidence,
        pose_confidence=pose_confidence,
        execution_confidence=execution_confidence,
        final_revalidation_reasons=_final_motion_reasons(
            pose_records, moves
        ),
        machine_executable=machine_executable,
        competition_gate_passed=competition_passed,
        strict_robust_gate_passed=strict_passed,
        execution_policy=config.execution_policy,
        execution_quality=quality,
        forced_output=bool(executable and not competition_passed),
        risk_warnings=sorted(set(warnings)),
        competition_failure_reasons=competition_reasons,
        strict_robust_failure_reasons=[],
    )
