"""失败案例的可复核分类。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


KNOWN_SEARCH_FAILURES = {
    "no_complete_search_candidate",
    "unsupported_partial_edge_matching",
    "shared_interface_candidate_missing",
    "interface_graph_disconnected",
    "disconnected_positive_length_adjacency",
    "cycle_inconsistency",
    "continuous_optimization_failed",
    "final_geometry_invalid",
    "reflection_error",
    "same_side_interface",
    "directed_endpoint_mismatch",
    "piece_role_swap",
    "outer_side_role_mismatch",
    "parallelogram_layout",
    "rectangle_axis_inconsistent",
    "junction_angle_inconsistent",
    "physically_equivalent_permutation",
    "timeout",
}


def classify_failure(
    solution: dict[str, Any],
    evaluation: dict[str, Any],
    target_area: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not solution.get("success"):
        explicit = str(solution.get("failure_type", "")).strip()
        text = str(solution.get("failure_reason", "")).lower()
        if explicit in KNOWN_SEARCH_FAILURES:
            reasons.append(explicit)
        elif "timeout" in text:
            reasons.append("timeout")
        else:
            reasons.append("no_complete_search_candidate")
    elif evaluation.get("competition_white_success"):
        # 白色几何已成功时，绝对 truth 差异只能作为研究诊断。
        if evaluation.get("equivalent_global_rotation"):
            reasons.append("equivalent_global_rotation")
        if solution.get("ambiguous"):
            reasons.append("ambiguous_geometry")
        if evaluation.get("piece_correspondence_accuracy", 1.0) < 1.0:
            reasons.append("physically_equivalent_permutation")
    else:
        if not evaluation.get("no_reflection", True):
            reasons.append("reflection_error")
        if not evaluation.get(
            "positive_length_adjacency_connected", True
        ):
            reasons.append("disconnected_positive_length_adjacency")
        if evaluation.get("piece_correspondence_accuracy", 0.0) < 1.0:
            reasons.append("wrong_piece_assignment")
            if evaluation.get("no_reflection", True):
                reasons.append("piece_role_swap")
        if evaluation.get("adjacency_accuracy", 0.0) < 1.0:
            reasons.append("wrong_edge_matching")
        if (evaluation.get("rotation_error_deg_mean") or 0.0) > 10.0:
            reasons.append("rotation_error")
        if (evaluation.get("translation_error_mm_mean") or 0.0) > 20.0:
            reasons.append("translation_error")
        if (
            evaluation.get(
                "ground_truth_actual_overlap_area_mm2",
                evaluation.get("overlap_area_mm2", 0.0),
            )
            > 1e-5
        ):
            reasons.append("excessive_overlap")
        if evaluation.get("gap_area_mm2", 0.0) > 0.10 * target_area:
            reasons.append("excessive_gap")
        if evaluation.get("union_rectangle_iou", 0.0) < 0.90:
            reasons.append("low_iou")
        if not evaluation.get("connected_with_tolerance_10mm", False):
            reasons.append("disconnected_with_tolerance")
        elif not evaluation.get("exact_connected", False):
            reasons.append("disconnected_exact_only")
        if solution.get("ambiguous"):
            reasons.append("ambiguous_geometry")
        rejection_counts = (
            solution.get("search_diagnostics", {})
            .get("geometry_candidate_rejection_reasons", {})
        )
        for name in (
            "same_side_interface",
            "directed_endpoint_mismatch",
            "outer_side_role_mismatch",
            "parallelogram_layout",
            "rectangle_axis_inconsistent",
            "junction_angle_inconsistent",
        ):
            if int(rejection_counts.get(name, 0)) > 0:
                reasons.append(name)
        if evaluation.get("geometry_success") is False and not reasons:
            reasons.append("final_geometry_invalid")
    primary_order = [
        "evaluator_coordinate_mismatch",
        "timeout",
        "disconnected_positive_length_adjacency",
        "interface_graph_disconnected",
        "unsupported_partial_edge_matching",
        "shared_interface_candidate_missing",
        "no_complete_search_candidate",
        "cycle_inconsistency",
        "continuous_optimization_failed",
        "final_geometry_invalid",
        "reflection_error",
        "same_side_interface",
        "directed_endpoint_mismatch",
        "parallelogram_layout",
        "rectangle_axis_inconsistent",
        "outer_side_role_mismatch",
        "junction_angle_inconsistent",
        "piece_role_swap",
        "wrong_piece_assignment",
        "wrong_edge_matching",
        "rotation_error",
        "translation_error",
        "excessive_overlap",
        "excessive_gap",
        "low_iou",
        "disconnected_with_tolerance",
        "disconnected_exact_only",
        "equivalent_global_rotation",
        "physically_equivalent_permutation",
        "ambiguous_geometry",
    ]
    primary = next((name for name in primary_order if name in reasons), None)
    return {"primary_failure_type": primary, "all_failure_types": reasons}


def write_failure_classification(
    case_dir: str | Path,
    solution: dict[str, Any],
    evaluation: dict[str, Any],
    target_area: float,
) -> dict[str, Any]:
    classification = classify_failure(solution, evaluation, target_area)
    path = Path(case_dir) / "solution" / "failure_type.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(classification, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return classification
