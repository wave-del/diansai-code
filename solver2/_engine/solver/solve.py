"""只读取指定 observations.json 的独立求解入口。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from solver2.acceleration import ClearanceScanBackend

from ..appearance.candidate_resolver import resolve_candidates_by_appearance
from ..appearance.config import (
    APPEARANCE_WINNER_MARGIN,
    GEOMETRY_AMBIGUITY_THRESHOLD,
)
from ..appearance.models import (
    AppearanceResolutionResult,
    GeometryAmbiguityResult,
)
from ..appearance.policy import (
    default_appearance_resolution,
    default_geometry_ambiguity,
    normalize_appearance_observation,
    resolve_patterned_variants,
)
from ..config import COORDINATE_SYSTEM
from ..generator.fixed_generator import (
    FIXED_HEIGHT_MM,
    FIXED_TARGET_SEAM_GAP_MM,
    FIXED_WIDTH_MM,
    fixed_final_target_transform,
)
from ..geometry import pairwise_overlap
from ..logging_config import configure_logging
from ..models import Score
from ..planning.machine_transform import load_calibration
from ..planning.motion_plan import (
    build_moves,
    validate_transport_paths,
)
from ..planning.clearance_planner import (
    pairwise_positive_overlap_area,
    plan_clearance,
)
from ..profiling import MemoryTracker, StageTimer
from ..runtime_config import RuntimeConfig, resolve_runtime_config
from ..transforms import compose_transform, transform_points
from .fixed_template_solver import solve_fixed
from .preprocess import preprocess_observations
from .observation_uncertainty import (
    build_observation_uncertainty,
    vertex_distance_metrics,
)
from .optimizer import optimize_generic_candidate, optimize_template_poses
from .exact_candidate_scoring import (
    exact_geometry_metrics,
    exact_geometry_objective,
)
from .parallel_search import solve_random_top_level_processes
from .placed_interface_discovery import discover_interfaces_in_placed_state
from .rectangle_fit import target_offset
from .scoring import confidence_from_candidates
from .search import solve_random
from .recovery import recover_random_candidate
from .execution_gate import evaluate_execution_gate
from .random_clearance import plan_random_clearance
from .random_execution_gate import evaluate_random_execution_gate
from .random_safety import assess_random_assembly
from .output_quality import MINIMUM_OUTPUT_FILL_RATIO, rectangle_fill_ratio
from .solution_output import (
    write_failure_artifacts,
    write_solution_artifacts,
)


def _failure(
    output: Path,
    reason: str,
    elapsed: float,
    logger: logging.Logger,
    *,
    failure_type: str = "no_complete_search_candidate",
    search_diagnostics: dict[str, Any] | None = None,
    recovery: dict[str, Any] | None = None,
    recommended_action: str = "rescan",
    mode: str | None = None,
    surface_mode: str = "white",
    geometry_ambiguity: GeometryAmbiguityResult | None = None,
    appearance_resolution: AppearanceResolutionResult | None = None,
) -> dict:
    result = {
        "schema_version": "1.0",
        "mode": mode,
        "surface_mode": surface_mode,
        "success": False,
        "ambiguous": True,
        "confidence": 0.0,
        "executable": False,
        "machine_executable": False,
        "competition_gate_passed": False,
        "strict_robust_gate_passed": False,
        "execution_quality": "last_resort",
        "forced_output": False,
        "recommended_action": recommended_action,
        "execution_block_reasons": ["solver_failed"],
        "failure_reason": reason,
        "failure_type": failure_type,
        "search_nodes": int((search_diagnostics or {}).get("search_nodes", 0)),
        "search_budget_limited": bool(
            (search_diagnostics or {}).get(
                "search_budget_limited",
                (search_diagnostics or {}).get("budget_exhausted", False),
            )
        ),
        "search_level_reached": int(
            (search_diagnostics or {}).get("search_level_reached", 1)
        ),
        "search_diagnostics": search_diagnostics or {},
        "elapsed_seconds": elapsed,
        "moves": [],
        "geometry_ambiguity": (
            geometry_ambiguity.as_json()
            if geometry_ambiguity is not None
            else default_geometry_ambiguity(
                GEOMETRY_AMBIGUITY_THRESHOLD
            ).as_json()
        ),
        "appearance_resolution": (
            appearance_resolution.as_json()
            if appearance_resolution is not None
            else default_appearance_resolution(
                APPEARANCE_WINNER_MARGIN
            ).as_json()
        ),
        "recovery": recovery or {
            "enabled": False,
            "triggered": False,
            "status": "nonrecoverable_failure",
        },
    }
    write_failure_artifacts(output, result, search_diagnostics or {})
    logger.error("求解失败：%s", reason)
    return result


def _recommended_action_for_exception(exc: Exception) -> str:
    message = str(exc).lower()
    operator_keywords = (
        "schema",
        "calibration",
        "runtime config",
        "machine",
        "work range",
        "transport z",
        "collision_mode",
        "json",
    )
    return (
        "operator_check"
        if any(keyword in message for keyword in operator_keywords)
        else "rescan"
    )


def _geometry_score(targets: list[np.ndarray], width: float, height: float, base_error: float) -> Score:
    polygons = [Polygon(vertices) for vertices in targets]
    union = unary_union(polygons)
    minx, miny, maxx, maxy = union.bounds
    rect = box(minx, miny, minx + width, miny + height)
    overlap = pairwise_overlap(polygons)
    uncovered = float(rect.difference(union).area)
    outside = float(union.difference(rect).area)
    fill = float(union.intersection(rect).area / rect.area) if rect.area else 0.0
    connectivity = 1 if union.geom_type == "Polygon" else len(getattr(union, "geoms", []))
    return Score(
        total=float(base_error + overlap + (uncovered + outside) / 100.0),
        matched_edge_rms_mm=float(base_error),
        matched_vertex_rms_mm=float(base_error),
        outer_edge_rms_mm=float((uncovered + outside) / max(2 * (width + height), 1.0)),
        overlap_area_mm2=float(overlap),
        uncovered_area_mm2=uncovered,
        outside_area_mm2=outside,
        rectangle_fill_ratio=fill,
        union_connectivity=connectivity,
        width_mm=float(width),
        height_mm=float(height),
    )


def _relative_assembly_signature(candidate: Any) -> tuple:
    """消除全局 SE(2) 后比较物理解；全局旋转不算未解决多解。"""
    poses = sorted(candidate.poses, key=lambda pose: pose.piece_id)
    centroids = {
        pose.piece_id: np.mean(pose.target_vertices_local, axis=0)
        for pose in poses
    }
    rows = []
    for index, first in enumerate(poses):
        for second in poses[index + 1:]:
            rows.append((
                first.piece_id,
                second.piece_id,
                round(float(np.linalg.norm(
                    centroids[first.piece_id] - centroids[second.piece_id]
                )), 4),
                round(float((second.angle - first.angle) % 360.0), 4),
            ))
    return tuple(rows)


def _choose_fixed_candidate(
    candidates: list[Any],
    pieces: list[Any],
    data: dict[str, Any],
    config: RuntimeConfig,
    started: float,
    clearance_backend: ClearanceScanBackend | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Try fixed-template alternatives without changing template geometry."""
    width, height = FIXED_WIDTH_MM, FIXED_HEIGHT_MM
    offset = target_offset((0.0, 0.0, width, height))
    radius = float(data.get(
        "measurement_error_radius_mm",
        data.get("noise_bound_mm", 0.0),
    ))
    attempts: list[dict[str, Any]] = []
    ranked: list[tuple[int, float, Any]] = []
    limit = (
        min(len(candidates), config.recovery_candidate_limit)
        if config.enable_recovery else 1
    )
    for index, candidate in enumerate(candidates[:limit]):
        if (
            config.enable_recovery
            and not config.unlimited_search
            and perf_counter() - started
            >= config.recovery_total_deadline_seconds
            - config.recovery_finalization_reserve_seconds
        ):
            break
        records = []
        for pose in candidate.poses:
            source = next(
                piece for piece in pieces
                if piece.piece_id == pose.piece_id
            )
            records.append({
                "piece_id": pose.piece_id,
                "observed": source.vertices,
                "target": np.asarray(
                    pose.template_vertices_local, dtype=float
                ) + offset,
                "angle": pose.angle,
                "translation": np.asarray(pose.translation, dtype=float)
                + offset,
            })
        clearance = plan_clearance(
            [record["target"] for record in records],
            radius,
            machine_position_margin_mm=config.machine_position_margin_mm,
            minimum_clearance_mm=config.minimum_clearance_mm,
            maximum_vertex_distance_mm=config.maximum_vertex_distance_mm,
            overlap_tolerance_mm2=config.overlap_numerical_epsilon_mm2,
            clearance_backend=clearance_backend,
        )
        reasons = []
        if not clearance.feasible:
            reasons.append("clearance_infeasible")
        reasons.extend(validate_transport_paths(
            records,
            build_moves(records, 1.0),
            config.transport_collision_mode,
            overlap_epsilon_mm2=config.overlap_numerical_epsilon_mm2,
        ))
        candidate_id = f"fixed_candidate_{index + 1:03d}"
        attempts.append({
            "candidate_id": candidate_id,
            "safe": not reasons,
            "block_reasons": sorted(set(reasons)),
        })
        ranked.append((len(set(reasons)), float(candidate.total), candidate))
        if not reasons:
            break
    if not ranked:
        return candidates[0], {
            "enabled": bool(config.enable_recovery),
            "triggered": False,
            "status": "deadline_exhausted",
            "attempt_count": 0,
            "selected_attempt": "fixed_candidate_001",
            "attempts": [],
        }
    selected = min(ranked, key=lambda item: (item[0], item[1]))[2]
    selected_index = next(
        index for index, candidate in enumerate(candidates)
        if candidate is selected
    )
    selected_attempt = f"fixed_candidate_{selected_index + 1:03d}"
    safe = next(
        item["safe"] for item in attempts
        if item["candidate_id"] == selected_attempt
    )
    return selected, {
        "enabled": bool(config.enable_recovery),
        "triggered": len(attempts) > 1,
        "status": (
            "recovered"
            if safe and len(attempts) > 1
            else ("not_needed" if safe else "no_safe_candidate")
        ),
        "attempt_count": len(attempts),
        "selected_attempt": selected_attempt,
        "attempts": attempts,
    }


def solve_observations(
    input_path: str | Path,
    output: str | Path,
    machine_calibration: str | Path | None = None,
    verbose: bool = False,
    print_plan: bool = True,
    runtime_config: RuntimeConfig | dict[str, Any] | str | Path | None = None,
    diagnostics_level: str | None = None,
    surface_mode: str | None = None,
    rigid_backend=None,
    clearance_backend: ClearanceScanBackend | None = None,
    appearance_scorer=None,
) -> dict:
    """求解器唯一数据入口；不会搜索输入文件所在目录。"""
    started = perf_counter()
    implicit_runtime_config = runtime_config is None
    config = (
        RuntimeConfig()
        if runtime_config is None
        else resolve_runtime_config(runtime_config)
    )
    if diagnostics_level is not None:
        config = RuntimeConfig.from_mapping({
            **config.as_dict(), "diagnostics_level": diagnostics_level,
        })
    config.apply_blas_environment()
    timer = StageTimer()
    memory = MemoryTracker(
        enabled=True,
        track_python_allocations=config.execution_profile == "debug",
    )
    memory.start()
    input_file = Path(input_path)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir / "solver.log", verbose)
    try:
        read_started = perf_counter()
        data = json.loads(input_file.read_text(encoding="utf-8"))
        timer.add("input_read", perf_counter() - read_started)
        preprocess_started = perf_counter()
        pieces = preprocess_observations(data)
        timer.add("preprocess", perf_counter() - preprocess_started)
        uncertainty_started = perf_counter()
        observation_uncertainty = build_observation_uncertainty(
            data,
            pieces,
            default_hard_error_radius_mm=(
                config.default_hard_error_radius_mm
            ),
        )
        timer.add(
            "observation_uncertainty",
            perf_counter() - uncertainty_started,
        )
        mode = data.get("mode")
        # Preserve the historical fixed-mode debug artifacts and conservative
        # visual gate.  Random defaults to summary diagnostics because its
        # progressive hard pool can be large; this changes diagnostics only.
        if implicit_runtime_config and mode == "fixed":
            config = RuntimeConfig(
                execution_profile="debug", diagnostics_level="full"
            )
        effective_surface_mode = (
            surface_mode or data.get("surface_mode", "white")
        )
        if effective_surface_mode not in {"white", "patterned"}:
            raise ValueError(
                "surface_mode must be either 'white' or 'patterned'"
            )
        geometry_ambiguity = default_geometry_ambiguity(
            config.geometry_ambiguity_threshold
        )
        appearance_resolution = default_appearance_resolution(
            config.appearance_winner_margin
        )
        pose_records: list[dict] = []
        candidate_rows: list[dict] = []
        search_diagnostics: dict[str, Any] = {}
        search_started = perf_counter()
        if mode == "fixed":
            candidates = solve_fixed(
                pieces,
                max(
                    float(
                        observation_uncertainty.typical_sigma_mm or 0.0
                    ),
                    float(
                        observation_uncertainty.hard_error_radius_mm
                    ),
                ),
                **({"rigid_backend": rigid_backend} if rigid_backend is not None else {}),
                full_diagnostics=config.diagnostics_level == "full",
            )
            if not candidates:
                return _failure(output_dir, "没有可用的固定模板候选", perf_counter() - started, logger)
            best, fixed_recovery = _choose_fixed_candidate(
                candidates, pieces, data, config, started, clearance_backend
            )
            scores = [candidate.total for candidate in candidates]
            width, height = FIXED_WIDTH_MM, FIXED_HEIGHT_MM
            offset = target_offset((0.0, 0.0, width, height))
            pose_uncertainty = {}
            for pose in best.poses:
                observed = next(piece.vertices for piece in pieces if piece.piece_id == pose.piece_id)
                pose_records.append(
                    {
                        "piece_id": pose.piece_id,
                        "template_id": pose.template_id,
                        "observed": observed,
                        "target": pose.target_vertices_local,
                        "template_target": pose.template_vertices_local,
                        "ideal_template_target": (
                            pose.template_vertices_local
                        ),
                        "structural_observed_corners": (
                            pose.physical_observation.structural_corners
                        ),
                        "structural_target": (
                            pose.structural_target_vertices_local
                        ),
                        "edge_sample_groups": (
                            pose.physical_observation.edge_sample_groups
                        ),
                        "physical_contour_diagnostics": (
                            pose.physical_observation
                            .simplification_diagnostics
                        ),
                        "support_line_rms_mm": pose.support_line_rms_mm,
                        "angle": pose.angle,
                        "translation": pose.translation,
                        "edges": next(piece.edges for piece in pieces if piece.piece_id == pose.piece_id),
                    }
                )
                pose_uncertainty[pose.piece_id] = {
                    "rotation_std_deg": pose.rotation_std_deg,
                    "translation_std_mm": pose.translation_std_mm,
                    "predicted_max_vertex_error_mm": pose.predicted_max_vertex_error_mm,
                }
            # 固定模式只匹配模板身份；最终落位来自预定义的静态坐标。
            geometric_fill = rectangle_fill_ratio(record["target"] for record in pose_records)
            optimization = {
                "optimization_success": True,
                "optimization_accepted": True,
                "before_rms_mm": 0.0,
                "after_rms_mm": 0.0,
                "improvement_mm": 0.0,
                "nfev": 0,
                "residual_semantics": "fixed_static_target_template",
                "virtual_anchor_point_correspondences_used": False,
            }
            for record in pose_records:
                static_angle, static_translation = fixed_final_target_transform(
                    record["template_id"]
                )
                angle, translation = compose_transform(
                    float(record["angle"]),
                    np.asarray(record["translation"], dtype=float),
                    static_angle,
                    static_translation,
                )
                record["angle"] = angle
                record["translation"] = translation
                record["target"] = transform_points(
                    record["observed"], angle, translation
                )
                record["structural_target"] = transform_points(
                    record["structural_observed_corners"], angle, translation
                )
                record["estimated_physical_target"] = (
                    np.asarray(record["target"], dtype=float).copy()
                )
                static_template = transform_points(
                    record["template_target"],
                    static_angle,
                    static_translation,
                )
                record["template_target"] = static_template
                record["ideal_template_target"] = static_template.copy()
                record["virtual_assembly_anchors"] = []
            fixed_target_points = np.vstack([
                np.asarray(record["ideal_template_target"], dtype=float)
                for record in pose_records
            ])
            fixed_bounds = (
                float(fixed_target_points[:, 0].min()),
                float(fixed_target_points[:, 1].min()),
                float(fixed_target_points[:, 0].max()),
                float(fixed_target_points[:, 1].max()),
            )
            width = fixed_bounds[2] - fixed_bounds[0]
            height = fixed_bounds[3] - fixed_bounds[1]
            for rank, candidate in enumerate(candidates, 1):
                candidate_rows.append(
                    {
                        "rank": rank,
                        "total_score": candidate.total,
                        "mapping": {p.piece_id: p.template_id for p in candidate.poses},
                        "matched_vertex_rms_mm": candidate.total,
                        "piece_assignment_score": candidate.piece_assignment_score,
                        "vertex_fit_score": candidate.vertex_fit_score,
                        "edge_direction_score": candidate.edge_direction_score,
                        "edge_length_ratio_score": candidate.edge_length_ratio_score,
                        "joint_pose_score": candidate.joint_pose_score,
                        "predicted_execution_uncertainty": candidate.predicted_execution_uncertainty,
                        "estimators": {p.piece_id: p.estimator for p in candidate.poses},
                        "estimator_rms_mm": {p.piece_id: p.estimator_rms for p in candidate.poses},
                        "physical_contour": {
                            p.piece_id: (
                                p.physical_observation
                                .simplification_diagnostics
                            )
                            for p in candidate.poses
                        },
                        "support_line_rms_mm": {
                            p.piece_id: p.support_line_rms_mm
                            for p in candidate.poses
                        },
                        "optimization": (
                            optimization if candidate is best else None
                        ),
                        "selected_after_optimization": candidate is best,
                    }
                )
            targets = [np.asarray(record["estimated_physical_target"]) for record in pose_records]
            score = _geometry_score(targets, width, height, best.total)
            confidence, ambiguous = confidence_from_candidates(scores, best.total, score.rectangle_fill_ratio)
            if effective_surface_mode == "patterned":
                appearance_resolution = resolve_candidates_by_appearance(
                    ["fixed_candidate_001"],
                    normalize_appearance_observation(data),
                    config,
                )
        elif mode == "random":
            if rigid_backend is not None and config.parallel_mode != "off":
                from dataclasses import replace
                config = replace(config, parallel_mode="off", worker_count=1)
            recovery_outcome = None
            if config.enable_recovery:
                recovery_outcome = recover_random_candidate(
                    pieces,
                    float(data.get("noise_bound_mm", 0.0)),
                    min(int(data.get("search_top_n", 100)), config.top_n),
                    data.get("appearance_edge_matches"),
                    config,
                    started=started,
                    stage_timer=timer,
                    rigid_backend=rigid_backend,
                    surface_mode=effective_surface_mode,
                    uncertainty_profile=observation_uncertainty,
                )
                candidates = recovery_outcome.candidates
                search_diagnostics.update(recovery_outcome.diagnostics)
                recovery_choice = recovery_outcome.selected
                # 保留诊断布局供预览；最终几何、留缝和执行门独立复核。
                if (
                    recovery_choice is None
                    and recovery_outcome.diagnostic is not None
                    and recovery_outcome.diagnostic.machine_executable
                ):
                    recovery_choice = recovery_outcome.diagnostic
                if recovery_choice is None:
                    return _failure(
                        output_dir,
                        "recovery exhausted without a machine-executable random candidate",
                        perf_counter() - started,
                        logger,
                        failure_type=recovery_outcome.recovery["status"],
                        search_diagnostics=search_diagnostics,
                        recovery=recovery_outcome.recovery,
                        recommended_action=(
                            "rescan"
                            if config.execution_policy == "strict"
                            else "operator_check"
                        ),
                        mode=mode,
                        surface_mode=effective_surface_mode,
                        geometry_ambiguity=geometry_ambiguity,
                        appearance_resolution=appearance_resolution,
                    )
                if effective_surface_mode == "patterned":
                    surface_selection = resolve_patterned_variants(
                        data,
                        recovery_outcome.safe_candidates,
                        recovery_choice,
                        config,
                        appearance_scorer=appearance_scorer,
                    )
                    geometry_ambiguity = (
                        surface_selection.geometry_ambiguity
                    )
                    appearance_resolution = (
                        surface_selection.appearance_resolution
                    )
                    recovery_choice = (
                        surface_selection.selected_variant
                    )
                    if (
                        appearance_resolution.resolved
                        and recovery_choice is not None
                    ):
                        recovery_outcome.selected = recovery_choice
                        recovery_outcome.recovery[
                            "selected_attempt"
                        ] = recovery_choice.candidate_id
                        recovery_outcome.recovery[
                            "selected_variant"
                        ] = recovery_choice.variant_id
            elif (
                config.parallel_mode == "top_level_processes"
                and config.worker_count > 1
            ):
                candidates = solve_random_top_level_processes(
                    pieces,
                    float(data.get("noise_bound_mm", 0.0)),
                    min(int(data.get("search_top_n", 100)), config.top_n),
                    data.get("appearance_edge_matches"),
                    None,
                    search_diagnostics,
                    config,
                    uncertainty_profile=observation_uncertainty,
                )
            else:
                candidates = solve_random(
                    pieces,
                    float(data.get("noise_bound_mm", 0.0)),
                    min(int(data.get("search_top_n", 100)), config.top_n),
                    data.get("appearance_edge_matches"),
                    diagnostics=search_diagnostics,
                    runtime_config=config,
                    stage_timer=timer,
                    rigid_backend=rigid_backend,
                    uncertainty_profile=observation_uncertainty,
                )
            if not candidates:
                if config.failure_diagnostics_level == "full":
                    write_global_search_diagnostics(
                        output_dir,
                        pieces,
                        search_diagnostics,
                        None,
                        float(data.get("noise_bound_mm", 0.0)),
                        observation_uncertainty,
                    )
                return _failure(
                    output_dir,
                    "随机模式界面图搜索后没有合法矩形候选",
                    perf_counter() - started,
                    logger,
                    failure_type=search_diagnostics.get(
                        "failure_type", "no_complete_search_candidate"
                    ),
                    search_diagnostics=search_diagnostics,
                    recommended_action=(
                        "rescan"
                        if config.execution_policy == "strict"
                        else "operator_check"
                    ),
                )
            best = (
                recovery_choice.candidate
                if recovery_outcome is not None
                else candidates[0]
            )
            optimized_results: dict[int, tuple] = {}
            optimized_candidate_count = 0

            def candidate_records(candidate: Any) -> list[dict]:
                result_records = []
                for candidate_pose in candidate.poses:
                    source_piece = next(
                        piece for piece in pieces
                        if piece.piece_id == candidate_pose.piece_id
                    )
                    result_records.append({
                        "piece_id": candidate_pose.piece_id,
                        "observed": source_piece.vertices,
                        "target": candidate_pose.target_vertices_local,
                        "angle": candidate_pose.angle,
                        "translation": candidate_pose.translation,
                        "top_edge_index": candidate_pose.top_edge_index,
                        "matched_prev_edge": candidate_pose.matched_prev_edge,
                        "matched_next_edge": candidate_pose.matched_next_edge,
                        "piece_confidence": source_piece.piece_confidence,
                    })
                return result_records

            if recovery_outcome is not None:
                selected_variant = recovery_choice
                pose_records = selected_variant.records
                width = selected_variant.width
                height = selected_variant.height
                optimization = selected_variant.optimization
                optimized_candidate_count = int(
                    selected_variant.variant_id == "optimized"
                )
                search_diagnostics["candidate_rank_before"] = (
                    next(
                        (
                            index
                            for index, item in enumerate(candidates, 1)
                            if item is best
                        ),
                        1,
                    )
                )
                search_diagnostics["candidate_rank_after"] = 1
                search_diagnostics["selected_candidate_id"] = (
                    selected_variant.candidate_id
                )
                search_diagnostics["selected_variant_id"] = (
                    selected_variant.variant_id
                )
            elif config.enable_global_optimization:
                optimization_started = perf_counter()
                limit = min(
                    len(candidates),
                    config.maximum_optimized_candidates,
                    config.optimized_candidate_limit,
                )
                ranked_optimized = []
                for candidate_index, candidate in enumerate(candidates[:limit]):
                    records = candidate_records(candidate)
                    discovery_started = perf_counter()
                    discovered = discover_interfaces_in_placed_state(
                        {
                            record["piece_id"]: record["target"]
                            for record in records
                        },
                        tolerance_mm=max(
                            1.0,
                            observation_uncertainty.hard_error_radius_mm
                            + float(
                                observation_uncertainty.typical_sigma_mm
                                or 0.0
                            ),
                        ),
                        length_epsilon_mm=max(
                            1e-6,
                            min(
                                observation_uncertainty
                                .hard_error_radius_mm,
                                1.0,
                            ),
                        ),
                        parallel_tolerance=max(
                            0.01,
                            0.01
                            + 0.01
                            * observation_uncertainty
                            .hard_error_radius_mm,
                        ),
                    )
                    timer.add(
                        "placed_interface_discovery",
                        perf_counter() - discovery_started,
                    )
                    (
                        updated_records,
                        updated_width,
                        updated_height,
                        candidate_optimization,
                    ) = optimize_generic_candidate(
                        records,
                        candidate.width,
                        candidate.height,
                        candidate.matches,
                        candidate.interfaces,
                        discovered.get("interfaces", []),
                        dimension_bounds=((90.0, 50.0), (120.0, 90.0))
                            if config.random_competition_rules else None,
                    )
                    metrics = exact_geometry_metrics(
                        [record["target"] for record in updated_records],
                        updated_width,
                        updated_height,
                    )
                    optimized_assessment = assess_random_assembly(
                        {
                            str(record["piece_id"]): np.asarray(
                                record["target"], dtype=float
                            )
                            for record in updated_records
                        },
                        observation_uncertainty,
                        config,
                        expected_piece_count=len(updated_records),
                        declared_interface_count=len(candidate.interfaces),
                        stage_name="post_optimization_ranking",
                    )
                    rank_key = (
                        *optimized_assessment.rank_key(),
                        exact_geometry_objective(metrics),
                        float(candidate_optimization.get(
                            "after_rms_mm", float("inf")
                        )),
                        float(candidate.total),
                        candidate_index,
                    )
                    optimized_results[candidate_index] = (
                        updated_records, updated_width, updated_height,
                        candidate_optimization, metrics, rank_key,
                    )
                    ranked_optimized.append(
                        (rank_key, candidate_index)
                    )
                optimized_candidate_count = len(ranked_optimized)
                _rank_key, selected_index = min(ranked_optimized)
                best = candidates[selected_index]
                (
                    pose_records, width, height, optimization,
                    _selected_metrics, _selected_rank_key,
                ) = optimized_results[selected_index]
                timer.add(
                    "global_optimization",
                    perf_counter() - optimization_started,
                )
                search_diagnostics["candidate_rank_before"] = (
                    selected_index + 1
                )
                search_diagnostics["candidate_rank_after"] = 1
            else:
                pose_records = candidate_records(best)
                width, height = best.width, best.height
                optimization = {
                    "optimization_success": True,
                    "optimization_accepted": False,
                    "before_rms_mm": best.edge_rms,
                    "after_rms_mm": best.edge_rms,
                    "improvement_mm": 0.0,
                    "nfev": 0,
                    "skipped_reason": "global_optimization_disabled",
                }
            search_diagnostics["selected_interface_ids"] = [
                item.interface_id for item in best.interfaces
            ]
            search_diagnostics["selected_composite_ids"] = sorted(
                {
                    item.composite_id
                    for item in best.interfaces
                    if item.composite_id is not None
                }
            )
            search_diagnostics[
                "correct_candidate_entered_combination_search"
            ] = bool(best.interfaces)
            search_diagnostics["fully_placed_state_selected"] = True
            search_diagnostics["geometry_candidate_selected"] = bool(
                best.admissibility is not None
                and best.admissibility.accepted
            )
            search_diagnostics["assessed_candidate_selected"] = bool(
                best.admissibility is not None
            )
            search_diagnostics["selected_geometry_admissibility"] = (
                recovery_choice.geometry_admissibility
                if recovery_outcome is not None
                and recovery_choice.geometry_admissibility is not None
                else (
                    best.admissibility.as_json()
                    if best.admissibility is not None
                    else None
                )
            )
            # 完整摆放与质量准入分开记录；未达理想门槛的完整摆放仍可
            # 作为一次性比赛动作的最优退化输出。
            search_diagnostics["complete_assembly_constructed"] = True
            comparison_candidates = [best] + [
                candidate for candidate in candidates if candidate is not best
            ]
            scores = [
                candidate.total for candidate in comparison_candidates
            ]
            all_local = np.vstack([record["target"] for record in pose_records])
            bounds = (
                float(all_local[:, 0].min()), float(all_local[:, 1].min()),
                float(all_local[:, 0].max()), float(all_local[:, 1].max()),
            )
            offset = target_offset(bounds)
            for record in pose_records:
                record["translation"] = record["translation"] + offset
                record["target"] = record["target"] + offset
                record["estimated_physical_target"] = record["target"].copy()
            for rank, candidate in enumerate(candidates, 1):
                candidate_rows.append(
                    {
                        "rank": rank,
                        "total_score": candidate.total,
                        "piece_order": [p.piece_id for p in candidate.poses],
                        "matched_edges": [
                            {"piece_a": a, "edge_a": ea, "piece_b": b, "edge_b": eb}
                            for a, ea, b, eb in candidate.matches
                        ],
                        "shared_interfaces": [
                            interface.as_json()
                            for interface in candidate.interfaces
                        ],
                        "cycle_residual_mm": candidate.cycle_residual_mm,
                        "width_mm": candidate.width,
                        "height_mm": candidate.height,
                        "rectangle_fill_ratio": candidate.fill_ratio,
                        "overlap_area_mm2": candidate.overlap,
                        "uncovered_area_mm2": candidate.uncovered,
                        "outside_area_mm2": candidate.outside,
                        "geometry_valid": candidate.geometry_valid,
                        "geometry_admissibility": (
                            candidate.admissibility.as_json()
                            if candidate.admissibility is not None
                            else None
                        ),
                        "geometry_score": candidate.geometry_score,
                        "appearance_score": candidate.appearance_score,
                        "legal_complete_candidate_count": candidate.legal_complete_candidate_count,
                        "matched_edge_rms_mm": candidate.edge_rms,
                        "optimization": (
                            optimized_results[rank - 1][3]
                            if rank - 1 in optimized_results
                            else (optimization if candidate is best else None)
                        ),
                        "exact_geometry_after": (
                            optimized_results[rank - 1][4]
                            if rank - 1 in optimized_results else None
                        ),
                        "selected_after_optimization": candidate is best,
                    }
                )
            targets = [np.asarray(record["target"]) for record in pose_records]
            score = _geometry_score(targets, width, height, best.edge_rms)
            confidence, _score_ambiguous = confidence_from_candidates(
                scores, best.edge_rms, best.fill_ratio
            )
            ambiguous = bool(
                len(comparison_candidates) > 1
                and abs(
                    comparison_candidates[1].total - best.total
                ) <= 1e-6
                and _relative_assembly_signature(comparison_candidates[1])
                != _relative_assembly_signature(best)
            )
            if _score_ambiguous and not ambiguous:
                confidence = max(confidence, 0.85)
            pose_uncertainty = {
                record["piece_id"]: {
                    "rotation_std_deg": float(max(0.05, best.edge_rms / 20.0)),
                    "translation_std_mm": float(
                        max(
                            0.1,
                            best.edge_rms,
                            observation_uncertainty.typical_sigma_mm
                            or 0.0,
                        )
                    ),
                    "predicted_max_vertex_error_mm": float(
                        max(
                            3.0 * best.edge_rms + 2.0,
                            observation_uncertainty
                            .hard_error_radius_mm,
                        )
                    ),
                }
                for record in pose_records
            }
            if (
                effective_surface_mode == "patterned"
                and recovery_outcome is None
            ):
                appearance_resolution = resolve_candidates_by_appearance(
                    ["candidate_001"],
                    normalize_appearance_observation(data),
                    config,
                )
        else:
            return _failure(
                output_dir,
                f"未知 mode: {mode!r}",
                perf_counter() - started,
                logger,
                recommended_action="operator_check",
            )

        if mode == "random":
            geometric_fill = rectangle_fill_ratio(record["target"] for record in pose_records)
        if geometric_fill + 1e-9 < MINIMUM_OUTPUT_FILL_RATIO:
            search_diagnostics["rectangle_fill_ratio"] = geometric_fill
            search_diagnostics["minimum_output_fill_ratio"] = MINIMUM_OUTPUT_FILL_RATIO
            failed = _failure(
                output_dir, f"矩形填充率 {geometric_fill:.2%}，未达到 95% 输出门槛",
                perf_counter() - started, logger,
                failure_type="rectangle_fill_below_minimum",
                search_diagnostics=search_diagnostics, mode=mode,
            )
            failed.update(rectangle_fill_ratio=geometric_fill,
                          minimum_output_fill_ratio=MINIMUM_OUTPUT_FILL_RATIO)
            write_failure_artifacts(output_dir, failed, search_diagnostics)
            return failed
        timer.add("candidate_search_and_pose", perf_counter() - search_started)
        clearance_started = perf_counter()
        measurement_error_radius = float(
            observation_uncertainty.hard_error_radius_mm
            if mode == "random"
            else data.get(
                "measurement_error_radius_mm",
                data.get("noise_bound_mm", 0.0),
            )
        )
        id_to_index = {
            record["piece_id"]: index
            for index, record in enumerate(pose_records)
        }
        explicit_adjacency = None
        if mode == "random":
            explicit_adjacency = {
                tuple(sorted((
                    id_to_index[item.first_piece],
                    id_to_index[item.second_piece],
                )))
                for item in best.interfaces
                if (
                    item.first_piece in id_to_index
                    and item.second_piece in id_to_index
                )
            }
            if not explicit_adjacency:
                explicit_adjacency = None
        for record in pose_records:
            record["estimated_physical_before_clearance"] = np.asarray(
                record.get("estimated_physical_target", record["target"]),
                dtype=float,
            ).copy()
        random_assessment = None
        if mode == "random":
            random_assessment = assess_random_assembly(
                {
                    str(record["piece_id"]): np.asarray(
                        record["estimated_physical_before_clearance"],
                        dtype=float,
                    )
                    for record in pose_records
                },
                observation_uncertainty,
                config,
                expected_piece_count=len(pose_records),
                declared_interface_count=len(best.interfaces),
                stage_name="final_pre_clearance",
            )
            clearance_plan = plan_random_clearance(
                {
                    str(record["piece_id"]): np.asarray(
                        record["estimated_physical_before_clearance"],
                        dtype=float,
                    )
                    for record in pose_records
                },
                random_assessment.topology,
                target_gap_mm=config.target_seam_gap_mm,
                minimum_gap_mm=config.minimum_target_seam_gap_mm,
                maximum_gap_mm=config.maximum_target_seam_gap_mm,
                maximum_vertex_distance_mm=config.maximum_vertex_distance_mm,
                measurement_error_radius_mm=measurement_error_radius,
                machine_position_margin_mm=config.machine_position_margin_mm,
                overlap_tolerance_mm2=config.physical_overlap_tolerance_mm2,
            )
        else:
            clearance_plan = plan_clearance(
                [
                    record["estimated_physical_before_clearance"]
                    for record in pose_records
                ],
                measurement_error_radius,
                machine_position_margin_mm=config.machine_position_margin_mm,
                minimum_clearance_mm=config.minimum_clearance_mm,
                maximum_vertex_distance_mm=config.maximum_vertex_distance_mm,
                base_correspondence_error_mm=0.0,
                adjacency_pairs=explicit_adjacency,
                overlap_tolerance_mm2=config.physical_overlap_tolerance_mm2,
                strategy="interface_local_compaction",
                target_seam_gap_mm=FIXED_TARGET_SEAM_GAP_MM,
                minimum_target_seam_gap_mm=FIXED_TARGET_SEAM_GAP_MM,
                maximum_target_seam_gap_mm=FIXED_TARGET_SEAM_GAP_MM,
                uncertainty_multiplier=config.seam_uncertainty_multiplier,
                typical_uncertainty_mm=(
                    observation_uncertainty.typical_sigma_mm
                ),
                interface_constraints=None,
            )
        # 固定模板只保留 clearance 诊断；不得把诊断规划的位移写回目标。
        if mode == "random" and clearance_plan.feasible:
            for record, delta, planned in zip(
                pose_records,
                clearance_plan.translations_mm,
                clearance_plan.planned_vertices,
            ):
                record["translation"] = (
                    np.asarray(record["translation"], dtype=float)
                    + np.asarray(delta, dtype=float)
                )
                record["target"] = np.asarray(planned, dtype=float)
                record["estimated_physical_target"] = np.asarray(
                    planned, dtype=float
                )
                if "structural_target" in record:
                    record["structural_target"] = (
                        np.asarray(record["structural_target"], dtype=float)
                        + np.asarray(delta, dtype=float)
                    )
                if "ideal_template_target" in record:
                    record["ideal_template_target"] = (
                        np.asarray(
                            record["ideal_template_target"], dtype=float
                        )
                        + np.asarray(delta, dtype=float)
                    )
                for anchor in record.get(
                    "virtual_assembly_anchors", []
                ):
                    anchor["target_anchor_a4_mm"] = (
                        np.asarray(
                            anchor["target_anchor_a4_mm"], dtype=float
                        )
                        + np.asarray(delta, dtype=float)
                    ).tolist()
                    anchor["ideal_template_anchor_a4_mm"] = (
                        np.asarray(
                            anchor["ideal_template_anchor_a4_mm"],
                            dtype=float,
                        )
                        + np.asarray(delta, dtype=float)
                    ).tolist()
        timer.add(
            "clearance_planning", perf_counter() - clearance_started
        )

        motion_started = perf_counter()
        calibration = load_calibration(machine_calibration)
        moves = build_moves(pose_records, confidence, calibration)
        transport_reasons = validate_transport_paths(
            pose_records,
            moves,
            config.transport_collision_mode,
            overlap_epsilon_mm2=config.overlap_numerical_epsilon_mm2,
        )
        timer.add("motion_planning", perf_counter() - motion_started)
        if mode == "fixed":
            final_target_points = np.vstack([
                np.asarray(record["ideal_template_target"], dtype=float)
                for record in pose_records
            ])
            target_center = [
                float((final_target_points[:, 0].min() + final_target_points[:, 0].max()) / 2.0),
                float((final_target_points[:, 1].min() + final_target_points[:, 1].max()) / 2.0),
            ]
        else:
            target_center = [
                float(offset[0] + bounds[0] + width / 2),
                float(offset[1] + bounds[1] + height / 2),
            ]
        target_info = {"width_mm": float(width), "height_mm": float(height), "center_a4_mm": target_center}
        elapsed = perf_counter() - started
        polygons = [Polygon(record["target"]) for record in pose_records]
        buffered = unary_union([polygon.buffer(5.0) for polygon in polygons])
        connected_10mm = buffered.geom_type == "Polygon"
        observed_final_polygons = [
            Polygon(transform_points(
                record["observed"],
                record["angle"],
                record["translation"],
            ))
            for record in pose_records
        ]
        observed_overlap_area = pairwise_positive_overlap_area(
            observed_final_polygons
        )
        estimated_overlap_area = (
            clearance_plan.estimated_physical_overlap_area_mm2
        )
        robust_overlap_area = (
            clearance_plan.robust_buffer_overlap_area_mm2
        )
        predicted_max = max(
            item["predicted_max_vertex_error_mm"] for item in pose_uncertainty.values()
        )
        nominal_max_vertex_distance = float(
            clearance_plan.max_corresponding_vertex_distance_mm
        )
        piece_vertex_uncertainty = {}
        for record in pose_records:
            uncertainty = pose_uncertainty[record["piece_id"]]
            observed = np.asarray(record["observed"], dtype=float)
            center = np.mean(observed, axis=0)
            radius = max(
                (
                    float(np.linalg.norm(point - center))
                    for point in observed
                ),
                default=0.0,
            )
            piece_vertex_uncertainty[record["piece_id"]] = float(
                measurement_error_radius
                + uncertainty["translation_std_mm"]
                + radius
                * abs(np.sin(np.deg2rad(
                    uncertainty["rotation_std_deg"]
                )))
            )
        distance_metrics = vertex_distance_metrics(
            nominal_max_vertex_distance,
            observation_uncertainty,
            target_gap_mm=clearance_plan.requested_clearance_mm,
            machine_position_margin_mm=(
                config.machine_position_margin_mm
            ),
            pose_residual_mm=(
                0.0 if mode == "fixed" else float(best.edge_rms)
            ),
            competition_limit_mm=config.maximum_vertex_distance_mm,
        )
        estimated_risk_vertex_distance = float(
            distance_metrics.estimated_risk_max_mm
        )
        robust_max_vertex_distance = float(
            distance_metrics.worst_case_robust_max_mm
        )
        rotation_std = max(
            item["rotation_std_deg"] for item in pose_uncertainty.values()
        )
        translation_std = max(
            item["translation_std_mm"] for item in pose_uncertainty.values()
        )
        pick_std = translation_std
        place_std = float(
            np.hypot(translation_std, predicted_max / 3.0)
        )
        candidate_margin = (
            float(
                comparison_candidates[1].total
                - comparison_candidates[0].total
            )
            if mode == "random" and len(comparison_candidates) > 1
            else (
                float(candidates[1].total - candidates[0].total)
                if len(candidates) > 1 else float("inf")
            )
        )
        active_recovery = (
            recovery_outcome
            if mode == "random"
            and locals().get("recovery_outcome") is not None
            else None
        )
        final_validation_started = perf_counter()
        gate_evaluator = (
            evaluate_random_execution_gate
            if mode == "random"
            else evaluate_execution_gate
        )
        gate = gate_evaluator(
            mode=mode,
            surface_mode=effective_surface_mode,
            ambiguous=bool(ambiguous),
            appearance_resolved=appearance_resolution.resolved,
            confidence=float(confidence),
            predicted_max_vertex_error_mm=predicted_max,
            nominal_max_vertex_distance_mm=(
                nominal_max_vertex_distance
            ),
            estimated_risk_vertex_distance_mm=(
                estimated_risk_vertex_distance
            ),
            robust_max_vertex_distance_mm=robust_max_vertex_distance,
            estimated_overlap_area_mm2=estimated_overlap_area,
            robust_overlap_area_mm2=robust_overlap_area,
            nonadjacent_robust_overlap_area_mm2=(
                clearance_plan.nonadjacent_robust_buffer_overlap_area_mm2
            ),
            clearance_plan=clearance_plan,
            optimization=optimization,
            search_diagnostics=search_diagnostics,
            recovery_outcome=active_recovery,
            candidate_margin=candidate_margin,
            pose_records=pose_records,
            moves=moves,
            transport_reasons=transport_reasons,
            visual_geometry_validated=bool(
                data.get("visual_geometry_validated", False)
            ),
            calibration=calibration,
            target_width_mm=float(width),
            target_height_mm=float(height),
            config=config,
            **(
                {"random_assessment": random_assessment}
                if mode == "random" else {}
            ),
        )
        timer.add("final_revalidation", perf_counter() - final_validation_started)
        reasons = gate.reasons
        solver_block_reasons = gate.solver_block_reasons
        geometric_block_reasons = [
            reason for reason in solver_block_reasons
            if reason != "appearance_ambiguity"
        ]
        if (
            mode == "random"
            and locals().get("recovery_outcome") is not None
            and geometric_block_reasons
            and recovery_outcome.recovery.get("status") == "recovered"
        ):
            recovery_outcome.recovery["status"] = "best_effort_selected"
        topology_confidence = gate.topology_confidence
        pose_confidence = gate.pose_confidence
        execution_confidence = gate.execution_confidence
        final_revalidation_reasons = (
            gate.final_revalidation_reasons
        )
        hardware_block_reasons = gate.hardware_block_reasons
        executable = gate.executable
        hardware_ready = gate.hardware_ready
        move_json = []
        records_by_id = {
            record["piece_id"]: record for record in pose_records
        }
        for move in moves:
            item = move.as_json()
            item["pose_uncertainty"] = pose_uncertainty[move.piece_id]
            source_record = records_by_id[move.piece_id]
            item[
                "estimated_physical_vertices_before_clearance_a4_mm"
            ] = np.asarray(
                source_record["estimated_physical_before_clearance"],
                dtype=float,
            ).tolist()
            item["estimated_physical_vertices_a4_mm"] = np.asarray(
                source_record.get(
                    "estimated_physical_target",
                    source_record["target"],
                ),
                dtype=float,
            ).tolist()
            item["planned_final_vertices_a4_mm"] = np.asarray(
                source_record["target"], dtype=float
            ).tolist()
            if mode == "fixed":
                item["observed_physical_contour"] = np.asarray(
                    source_record["observed"], dtype=float
                ).tolist()
                item["structural_observed_corners"] = np.asarray(
                    source_record["structural_observed_corners"],
                    dtype=float,
                ).tolist()
                item[
                    "transformed_observed_physical_contour"
                ] = np.asarray(
                    source_record["target"], dtype=float
                ).tolist()
                item["ideal_template_physical_contour"] = np.asarray(
                    source_record["ideal_template_target"], dtype=float
                ).tolist()
                item["virtual_assembly_anchors"] = list(
                    source_record.get("virtual_assembly_anchors", [])
                )
                item["physical_contour_diagnostics"] = dict(
                    source_record.get(
                        "physical_contour_diagnostics", {}
                    )
                )
            item["uncertainty_buffer_radius_mm"] = (
                clearance_plan.uncertainty_buffer_radius_mm
            )
            move_json.append(item)
        result = {
            "schema_version": "1.0",
            "mode": mode,
            "surface_mode": effective_surface_mode,
            "success": True,
            "rectangle_fill_ratio": geometric_fill,
            "minimum_output_fill_ratio": MINIMUM_OUTPUT_FILL_RATIO,
            "fill_ratio_basis": "before_execution_clearance",
            "planned_rectangle_fill_ratio": rectangle_fill_ratio(record["target"] for record in pose_records),
            "ambiguous": bool(
                ambiguous or geometry_ambiguity.detected
            ),
            "confidence": float(confidence),
            "executable": executable,
            "geometry_gate_passed": bool(gate.machine_executable and not [
                reason for reason in gate.reasons if reason != "appearance_ambiguity"
            ]),
            "machine_executable": gate.machine_executable,
            "competition_gate_passed": (
                gate.competition_gate_passed
            ),
            "strict_robust_gate_passed": (
                gate.strict_robust_gate_passed
            ),
            "execution_policy": gate.execution_policy,
            "execution_quality": gate.execution_quality,
            "forced_output": gate.forced_output,
            "risk_warnings": gate.risk_warnings,
            "recommended_action": (
                "execute"
                if executable
                else (
                    "operator_check"
                    if (
                        hardware_block_reasons
                        or config.execution_policy != "strict"
                    )
                    else "rescan"
                )
            ),
            "execution_block_reasons": sorted(set(reasons)),
            "geometry_ambiguity": geometry_ambiguity.as_json(),
            "appearance_resolution": appearance_resolution.as_json(),
            "hardware_ready": hardware_ready,
            "hardware_block_reasons": hardware_block_reasons,
            "predicted_max_vertex_error_mm": predicted_max,
            "nominal_max_vertex_distance_mm": nominal_max_vertex_distance,
            "estimated_actual_risk_vertex_distance_mm": (
                estimated_risk_vertex_distance
            ),
            "robust_max_vertex_distance_mm": robust_max_vertex_distance,
            "vertex_distance_limit_mm": config.maximum_vertex_distance_mm,
            "piece_vertex_uncertainty_mm": piece_vertex_uncertainty,
            "vertex_distance": distance_metrics.as_json(),
            "rotation_std_deg": rotation_std,
            "translation_std_mm": translation_std,
            "pick_std_mm": pick_std,
            "place_std_mm": place_std,
            "topology_confidence": topology_confidence,
            "pose_confidence": pose_confidence,
            "execution_confidence": execution_confidence,
            "search_candidate_valid": True,
            "final_candidate_revalidation": {
                "passed": not final_revalidation_reasons,
                "reasons": final_revalidation_reasons,
            },
            "physically_executable": executable,
            "measurement_error_radius_mm": measurement_error_radius,
            "machine_position_margin_mm": (
                config.machine_position_margin_mm
            ),
            "transport_collision_mode": config.transport_collision_mode,
            "safe_z_clearance_verified": (
                config.safe_z_clearance_verified
            ),
            "planned_clearance_mm": (
                clearance_plan.requested_clearance_mm
            ),
            "clearance_target_mm": (
                clearance_plan.requested_clearance_mm
            ),
            "minimum_pairwise_clearance_mm": (
                clearance_plan.achieved_minimum_clearance_mm
            ),
            "maximum_pairwise_clearance_mm": (
                clearance_plan.maximum_pairwise_clearance_mm
            ),
            "max_corresponding_vertex_distance_mm": (
                clearance_plan.max_corresponding_vertex_distance_mm
            ),
            "clearance_violation_mm": float(max(
                0.0,
                clearance_plan.requested_clearance_mm
                - clearance_plan.achieved_minimum_clearance_mm,
            )),
            "clearance_feasible": clearance_plan.feasible,
            "clearance": {
                "strategy": clearance_plan.strategy,
                "default_target_gap_mm": config.target_seam_gap_mm,
                "target_gap_mm": clearance_plan.requested_clearance_mm,
                "pre_compaction_max_gap_mm": (
                    clearance_plan.pre_compaction_max_gap_mm
                ),
                "post_compaction_max_gap_mm": (
                    clearance_plan.post_compaction_max_gap_mm
                ),
                "pre_compaction_nominal_vertex_distance_mm": (
                    clearance_plan.pre_compaction_max_gap_mm
                ),
                "post_compaction_nominal_vertex_distance_mm": (
                    nominal_max_vertex_distance
                ),
                "total_translation_mm": (
                    clearance_plan.total_translation_mm
                ),
                "interface_metrics": [
                    item.as_json()
                    for item in clearance_plan.interface_metrics
                ],
            },
            "observed_overlap_area_mm2": observed_overlap_area,
            "estimated_physical_overlap_area_mm2": (
                estimated_overlap_area
            ),
            "robust_buffer_overlap_area_mm2": robust_overlap_area,
            "adjacent_robust_buffer_overlap_area_mm2": (
                clearance_plan.adjacent_robust_buffer_overlap_area_mm2
            ),
            "nonadjacent_robust_buffer_overlap_area_mm2": (
                clearance_plan.nonadjacent_robust_buffer_overlap_area_mm2
            ),
            "physical_overlap_tolerance_mm2": (
                config.overlap_numerical_epsilon_mm2
            ),
            "candidate_score_margin": candidate_margin,
            "search_budget_limited": bool(
                search_diagnostics.get("search_budget_limited", False)
            ),
            "search_level_reached": int(
                search_diagnostics.get("search_level_reached", 1)
            ),
            "optimized_candidate_count": int(
                locals().get("optimized_candidate_count", int(
                    bool(optimization.get("nfev", 0))
                ))
            ),
            "optimization": optimization,
            "pose_uncertainty": pose_uncertainty,
            "coordinate_system": {k: v for k, v in COORDINATE_SYSTEM.items() if k != "angle_range"},
            "target_rectangle": target_info,
            "score": vars(score),
            "elapsed_seconds": elapsed,
            "search_nodes": int(search_diagnostics.get("search_nodes", 0)),
            "search_diagnostics": search_diagnostics,
            "effective_runtime_config": config.as_dict(),
            "performance_profile": {
                "execution_profile": config.execution_profile,
                "diagnostics_level": config.diagnostics_level,
                "search_strategy": config.search_strategy,
                "worker_count": config.worker_count,
                "stage_times": timer.snapshot(),
                "memory": memory.snapshot(),
            },
            "moves": move_json,
            "recovery": (
                recovery_outcome.recovery
                if mode == "random"
                and locals().get("recovery_outcome") is not None
                else (
                    fixed_recovery
                    if mode == "fixed"
                    else {
                        "enabled": bool(config.enable_recovery),
                        "triggered": False,
                        "status": "not_needed",
                    }
                )
            ),
        }
        if mode == "random" and (
            config.diagnostics_level != "off"
            and (
                config.diagnostics_level == "full"
                or search_diagnostics.get(
                    "generic_partial_fallback_used", False
                )
                or int(search_diagnostics.get("expanded_states", 0)) >= 50
            )
        ):
            # 全局搜索图仅用于诊断，避免生产控制路径依赖 matplotlib。
            try:
                from .global_search_diagnostics import (
                    write_global_search_diagnostics,
                )
            except ModuleNotFoundError as error:
                # matplotlib 只服务离线诊断；生产求解和既有 JSON 输出不能
                # 因可选绘图库缺失而从成功结果退化成求解失败。
                if error.name != "matplotlib":
                    raise
                logger.warning("缺少 matplotlib，跳过随机搜索图")
            else:
                write_global_search_diagnostics(
                    output_dir,
                    pieces,
                    search_diagnostics,
                    best,
                    float(data.get("noise_bound_mm", 0.0)),
                    observation_uncertainty,
                )
        if config.execution_profile == "debug":
            # 可视化是诊断用途的可选依赖；生产控制路径不加载它。
            from ..visualization.render_solution import render_solution

            visualization_started = perf_counter()
            render_solution(
                output_dir,
                moves,
                target_info,
                score,
                confidence,
                safety=result,
            )
            timer.add(
                "visualization", perf_counter() - visualization_started
            )
        write_solution_artifacts(
            output_dir,
            result,
            moves=moves,
            confidence=confidence,
            executable=executable,
            diagnostics_level=config.diagnostics_level,
            candidate_rows=candidate_rows,
            search_diagnostics=search_diagnostics,
            block_reasons=reasons,
            timer=timer,
            solver_started=started,
        )
        logger.info("求解成功，mode=%s, confidence=%.3f, elapsed=%.3fs", mode, confidence, elapsed)
        if print_plan:
            print_motion_plan(result)
        memory.stop()
        return result
    except Exception as exc:
        logger.exception("求解器异常")
        memory.stop()
        return _failure(
            output_dir,
            str(exc),
            perf_counter() - started,
            logger,
            failure_type=(
                "continuous_optimization_failed"
                if "least_squares" in str(exc)
                else "no_complete_search_candidate"
            ),
            search_diagnostics=locals().get("search_diagnostics", {}),
            recommended_action=_recommended_action_for_exception(exc),
            mode=locals().get("mode"),
            surface_mode=locals().get(
                "effective_surface_mode", "white"
            ),
            geometry_ambiguity=locals().get("geometry_ambiguity"),
            appearance_resolution=locals().get(
                "appearance_resolution"
            ),
        )


def print_motion_plan(solution: dict) -> None:
    if not solution.get("success"):
        print(f"求解失败：{solution.get('failure_reason', '未知原因')}")
        return
    target = solution["target_rectangle"]
    print("求解成功")
    print(f"目标矩形：{target['width_mm']:.2f} mm × {target['height_mm']:.2f} mm")
    print(f"置信度：{solution['confidence']:.3f}")
    print(f"是否存在多解：{'是' if solution['ambiguous'] else '否'}")
    if not solution.get("executable", False):
        print("禁止下发写字机：" + ", ".join(solution.get("execution_block_reasons", [])))
    for move in solution["moves"]:
        print(f"\n[{move['order']}] {move['piece_id']}")
        print(f"    Pick A4   = ({move['pick_xy_a4_mm'][0]:.2f}, {move['pick_xy_a4_mm'][1]:.2f}) mm")
        print(f"    Clearance = {move['pick_clearance_mm']:.2f} mm")
        print(f"    Rotate    = {move['rotation_deg_clockwise']:+.2f} deg clockwise")
        print(f"    Place A4  = ({move['place_xy_a4_mm'][0]:.2f}, {move['place_xy_a4_mm'][1]:.2f}) mm")
        if move.get("machine_coordinates"):
            mc = move["machine_coordinates"]
            print(f"    Pick Machine  = ({mc['pick_xy_machine'][0]:.3f}, {mc['pick_xy_machine'][1]:.3f})")
            print(f"    Place Machine = ({mc['place_xy_machine'][0]:.3f}, {mc['place_xy_machine'][1]:.3f})")
