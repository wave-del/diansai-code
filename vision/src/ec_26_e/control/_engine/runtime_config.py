"""统一的求解运行配置。

配置包括资源、诊断、可选算法路径与显式的输入边长门槛。默认使用
benchmark profile；production 本身仅关闭非必要输出，不放宽安全门槛。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import json
import math
import os
from pathlib import Path
from typing import Any

from .appearance.config import (
    APPEARANCE_WINNER_MARGIN,
    GEOMETRY_AMBIGUITY_THRESHOLD,
    MAX_APPEARANCE_CANDIDATES,
    MINIMUM_INTERFACE_APPEARANCE_SCORE,
)
from .config import (
    OVERLAP_NUMERICAL_EPSILON_MM2,
    VERTEX_DISTANCE_LIMIT_MM,
)
from .planning.clearance_planner import (
    DEFAULT_TARGET_SEAM_GAP_MM,
    LARGE_SEAM_GAP_WARNING_MM,
    MAX_TARGET_SEAM_GAP_MM,
    MIN_TARGET_SEAM_GAP_MM,
)

@dataclass(frozen=True)
class RuntimeConfig:
    execution_profile: str = "benchmark"
    diagnostics_level: str = "summary"
    search_strategy: str = "legacy_deterministic"
    max_search_nodes: int = 1_000_000
    fallback_search_nodes: int = 20_000
    max_search_seconds: float = 60.0
    top_n: int = 100
    complete_candidate_limit: int = 24
    optimized_candidate_limit: int = 100
    parallel_mode: str = "off"
    worker_count: int = 1
    blas_threads: int = 1
    cache_geometry: bool = True
    incremental_union: bool = True
    precompute_relative_transforms: bool = True
    enable_local_optimization: bool = True
    enable_global_optimization: bool = True
    enable_expensive_exact_scoring: bool = True
    deterministic: bool = True
    random_seed: int = 0
    machine_position_margin_mm: float = 1.0
    minimum_clearance_mm: float | None = None
    maximum_vertex_distance_mm: float = VERTEX_DISTANCE_LIMIT_MM
    minimum_execution_confidence: float = 0.6
    physical_overlap_tolerance_mm2: float = (
        OVERLAP_NUMERICAL_EPSILON_MM2
    )
    enable_recovery: bool = True
    recovery_total_deadline_seconds: float = 30.0
    recovery_finalization_reserve_seconds: float = 1.2
    recovery_initial_stage_seconds: float = 12.0
    recovery_min_deep_search_seconds: float = 1.5
    recovery_widened_node_multiplier: float = 2.0
    recovery_candidate_limit: int = 24
    overlap_numerical_epsilon_mm2: float = (
        OVERLAP_NUMERICAL_EPSILON_MM2
    )
    transport_collision_mode: str = "safe_z_clearance"
    safe_z_clearance_verified: bool = True
    safe_transport_z_mm: float | None = None
    maximum_piece_height_mm: float | None = None
    suction_tool_clearance_margin_mm: float | None = None
    geometry_ambiguity_threshold: float = GEOMETRY_AMBIGUITY_THRESHOLD
    appearance_winner_margin: float = APPEARANCE_WINNER_MARGIN
    max_appearance_candidates: int = MAX_APPEARANCE_CANDIDATES
    minimum_interface_appearance_score: float | None = (
        MINIMUM_INTERFACE_APPEARANCE_SCORE
    )
    appearance_resolution_reserve_seconds: float = 1.5
    patterned_require_appearance_for_unique: bool = True
    exact_initial_nodes: int = 1_000_000
    exact_fallback_nodes: int = 20_000
    noisy_initial_nodes: int = 200_000
    noisy_fallback_nodes: int = 40_000
    completion_limit: int = 24
    maximum_complete_candidates: int = 100
    maximum_optimized_candidates: int = 100
    normal_topology_candidates: int = 50
    complex_topology_candidates: int = 200
    wall_time_guard_seconds: float = 60.0
    failure_diagnostics_level: str = "full"
    enable_cprofile: bool = False
    default_hard_error_radius_mm: float = 3.0
    progressive_search_enabled: bool = True
    fast_max_matches_per_edge: int = 2
    expanded_max_matches_per_edge: int = 3
    recovery_max_matches_per_edge: int = 32
    fast_candidate_limit: int = 4
    expanded_candidate_limit: int = 8
    fast_optimize_candidate_limit: int = 1
    expanded_optimize_candidate_limit: int = 3
    fast_node_limit: int = 4_000
    expanded_node_limit: int = 30_000
    fast_beam_width: int = 24
    expanded_beam_width: int = 96
    recovery_beam_width: int = 512
    # 失败诊断仅保留有限个完整落位反例，避免常规运行写出过大 JSON；
    # 离线排障可显式提高该值以渲染每个被拒绝布局。
    diagnostic_rejection_sample_limit: int = 24
    # Truth-free hard gate between a fully placed search state and an
    # optimizable geometry candidate.  These are admission thresholds, not
    # competition/evaluator thresholds.
    minimum_geometry_candidate_fill_ratio: float = 0.85
    minimum_geometry_candidate_rectangle_iou: float = 0.82
    maximum_geometry_candidate_overlap_ratio: float = 0.02
    maximum_geometry_candidate_nonadjacent_overlap_ratio: float = 0.002
    maximum_geometry_candidate_uncovered_ratio: float = 0.15
    maximum_geometry_candidate_outside_ratio: float = 0.05
    geometry_candidate_dimension_relaxation_mm: float = 3.0
    minimum_repairable_geometry_iou: float = 0.70
    maximum_repairable_geometry_overlap_ratio: float = 0.04
    maximum_repairable_geometry_uncovered_ratio: float = 0.30
    repairable_geometry_dimension_relaxation_mm: float = 10.0
    interface_explanation_interval_tolerance: float = 0.08
    max_variants_per_discrete_signature: int = 2
    enable_directed_role_admission: bool = True
    angle_model_floor_deg: float = 1.0
    directed_endpoint_hard_rejection_z: float = 12.0
    rectangle_role_hard_rejection_z: float = 3.0
    minimum_outer_side_coverage_ratio: float = 0.72
    junction_closure_hard_rejection_z: float = 4.0
    junction_cluster_tolerance_mm: float = 1e-3
    enable_futility_detection: bool = True
    futility_min_candidate_improvement: float = 1e-3
    clearance_strategy: str = "interface_local_compaction"
    # 随机拼图的结构解保持严丝合缝；发往执行器前仅用平移生成 5 mm
    # 公共拼缝。3～6 mm 是留给视觉和运动误差的可调安全范围。
    target_seam_gap_mm: float = 5.0
    minimum_target_seam_gap_mm: float = 3.0
    maximum_target_seam_gap_mm: float = 6.0
    large_seam_gap_warning_mm: float = LARGE_SEAM_GAP_WARNING_MM
    seam_uncertainty_multiplier: float = 2.0
    execution_policy: str = "strict"
    random_competition_rules: bool = False
    minimum_piece_edge_mm: float = 20.0
    unlimited_search: bool = False
    minimum_safe_candidates: int = 1
    rectangle_corner_hypotheses: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum_piece_edge_mm) or self.minimum_piece_edge_mm <= 0:
            raise ValueError("minimum_piece_edge_mm 必须是有限正数")
        if type(self.unlimited_search) is not bool:
            raise TypeError("unlimited_search 必须为布尔值")
        if type(self.minimum_safe_candidates) is not int or self.minimum_safe_candidates < 1:
            raise ValueError("minimum_safe_candidates 必须为正整数")
        if self.execution_profile not in {"debug", "benchmark", "production"}:
            raise ValueError("execution_profile 必须为 debug/benchmark/production")
        if self.diagnostics_level not in {"off", "summary", "full"}:
            raise ValueError("diagnostics_level 必须为 off/summary/full")
        if self.search_strategy not in {"legacy_deterministic", "best_first_bnb"}:
            raise ValueError("未知 search_strategy")
        if self.parallel_mode not in {"off", "top_level_processes"}:
            raise ValueError("未知 parallel_mode")
        if self.machine_position_margin_mm < 0.0:
            raise ValueError("machine_position_margin_mm 不得为负数")
        if (
            self.minimum_clearance_mm is not None
            and self.minimum_clearance_mm < 0.0
        ):
            raise ValueError("minimum_clearance_mm 不得为负数")
        if self.maximum_vertex_distance_mm <= 0.0:
            raise ValueError("maximum_vertex_distance_mm 必须为正数")
        if not 0.0 <= self.minimum_execution_confidence <= 1.0:
            raise ValueError(
                "minimum_execution_confidence 必须位于 [0, 1]"
            )
        if self.physical_overlap_tolerance_mm2 < 0.0:
            raise ValueError("physical_overlap_tolerance_mm2 不得为负数")
        if self.recovery_total_deadline_seconds <= 0.0:
            raise ValueError("recovery_total_deadline_seconds must be positive")
        if not (
            0.0 <= self.recovery_finalization_reserve_seconds
            < self.recovery_total_deadline_seconds
        ):
            raise ValueError("invalid recovery finalization reserve")
        if self.recovery_initial_stage_seconds <= 0.0:
            raise ValueError("recovery_initial_stage_seconds must be positive")
        if self.recovery_min_deep_search_seconds < 0.0:
            raise ValueError("recovery_min_deep_search_seconds cannot be negative")
        if self.recovery_widened_node_multiplier < 1.0:
            raise ValueError("recovery_widened_node_multiplier must be >= 1")
        if self.overlap_numerical_epsilon_mm2 < 0.0:
            raise ValueError("overlap_numerical_epsilon_mm2 cannot be negative")
        for name in (
            "minimum_geometry_candidate_fill_ratio",
            "minimum_geometry_candidate_rectangle_iou",
            "maximum_geometry_candidate_overlap_ratio",
            "maximum_geometry_candidate_nonadjacent_overlap_ratio",
            "maximum_geometry_candidate_uncovered_ratio",
            "maximum_geometry_candidate_outside_ratio",
            "minimum_repairable_geometry_iou",
            "maximum_repairable_geometry_overlap_ratio",
            "maximum_repairable_geometry_uncovered_ratio",
            "interface_explanation_interval_tolerance",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.geometry_candidate_dimension_relaxation_mm < 0.0:
            raise ValueError(
                "geometry_candidate_dimension_relaxation_mm cannot be negative"
            )
        if self.repairable_geometry_dimension_relaxation_mm < 0.0:
            raise ValueError(
                "repairable_geometry_dimension_relaxation_mm cannot be negative"
            )
        if self.max_variants_per_discrete_signature < 1:
            raise ValueError(
                "max_variants_per_discrete_signature must be positive"
            )
        for name in (
            "angle_model_floor_deg",
            "directed_endpoint_hard_rejection_z",
            "rectangle_role_hard_rejection_z",
            "junction_closure_hard_rejection_z",
            "junction_cluster_tolerance_mm",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.minimum_outer_side_coverage_ratio <= 1.0:
            raise ValueError(
                "minimum_outer_side_coverage_ratio must be within [0, 1]"
            )
        if self.default_hard_error_radius_mm <= 0.0:
            raise ValueError("default_hard_error_radius_mm must be positive")
        if self.futility_min_candidate_improvement < 0.0:
            raise ValueError(
                "futility_min_candidate_improvement cannot be negative"
            )
        if self.clearance_strategy not in {
            "legacy_centroid_expansion", "interface_local_compaction",
        }:
            raise ValueError("unknown clearance_strategy")
        if self.execution_policy not in {
            "strict",
            "competition_best_effort",
            "competition_force_output",
        }:
            raise ValueError("unknown execution_policy")
        if not (
            0.0 <= self.minimum_target_seam_gap_mm
            <= self.target_seam_gap_mm
            <= self.maximum_target_seam_gap_mm
        ):
            raise ValueError("invalid target seam gap range")
        if (
            self.large_seam_gap_warning_mm
            < self.maximum_target_seam_gap_mm
        ):
            raise ValueError(
                "large seam warning must not be below maximum target gap"
            )
        if self.seam_uncertainty_multiplier < 0.0:
            raise ValueError("seam_uncertainty_multiplier cannot be negative")
        if self.transport_collision_mode not in {
            "safe_z_clearance", "swept_polygon", "head_only",
        }:
            raise ValueError("unknown transport_collision_mode")
        if self.transport_collision_mode == "safe_z_clearance":
            values = (
                self.safe_transport_z_mm,
                self.maximum_piece_height_mm,
                self.suction_tool_clearance_margin_mm,
            )
            if (
                not self.safe_z_clearance_verified
                and any(value is None for value in values)
            ):
                raise ValueError(
                    "unverified safe_z_clearance requires safe transport "
                    "height, maximum piece height and suction clearance margin"
                )
            if (
                not any(value is None for value in values)
                and float(self.safe_transport_z_mm) < (
                    float(self.maximum_piece_height_mm)
                    + float(self.suction_tool_clearance_margin_mm)
                )
            ):
                raise ValueError("configured safe transport Z is insufficient")
        if not 0.0 <= self.geometry_ambiguity_threshold <= 1.0:
            raise ValueError("geometry_ambiguity_threshold must be in [0, 1]")
        if not 0.0 <= self.appearance_winner_margin <= 1.0:
            raise ValueError("appearance_winner_margin must be in [0, 1]")
        if (
            self.minimum_interface_appearance_score is not None
            and not 0.0 <= self.minimum_interface_appearance_score <= 1.0
        ):
            raise ValueError(
                "minimum_interface_appearance_score must be in [0, 1]"
            )
        if self.appearance_resolution_reserve_seconds < 0.0:
            raise ValueError(
                "appearance_resolution_reserve_seconds cannot be negative"
            )
        if self.worker_count < 1:
            raise ValueError("worker_count 必须 >= 1")
        if self.blas_threads < 1:
            raise ValueError("blas_threads 必须 >= 1")
        for name in (
            "max_search_nodes", "fallback_search_nodes", "top_n",
            "complete_candidate_limit", "optimized_candidate_limit",
            "exact_initial_nodes", "exact_fallback_nodes",
            "noisy_initial_nodes", "noisy_fallback_nodes",
            "completion_limit", "maximum_complete_candidates",
            "maximum_optimized_candidates",
            "normal_topology_candidates", "complex_topology_candidates",
            "recovery_candidate_limit",
            "max_appearance_candidates",
            "fast_max_matches_per_edge",
            "expanded_max_matches_per_edge",
            "recovery_max_matches_per_edge",
            "fast_candidate_limit",
            "expanded_candidate_limit",
            "fast_optimize_candidate_limit",
            "expanded_optimize_candidate_limit",
            "fast_node_limit",
            "expanded_node_limit",
            "fast_beam_width",
            "expanded_beam_width",
            "recovery_beam_width",
            "diagnostic_rejection_sample_limit",
            "max_variants_per_discrete_signature",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} 必须 >= 1")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_profile_defaults(self) -> "RuntimeConfig":
        if self.execution_profile == "debug":
            return replace(self, diagnostics_level="full")
        if self.execution_profile == "production":
            return replace(self, diagnostics_level="off")
        return self

    def apply_blas_environment(self) -> None:
        value = str(self.blas_threads)
        for name in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        ):
            os.environ[name] = value

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "RuntimeConfig":
        if not value:
            return cls().with_profile_defaults()
        valid = {field.name for field in fields(cls)}
        unknown = sorted(set(value) - valid)
        if unknown:
            raise ValueError(f"RuntimeConfig 未知字段: {', '.join(unknown)}")
        return cls(**value).with_profile_defaults()

    @classmethod
    def load(cls, path: str | Path | None) -> "RuntimeConfig":
        if path is None:
            return cls().with_profile_defaults()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("runtime config 顶层必须为 JSON object")
        return cls.from_mapping(data)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def resolve_runtime_config(
    value: RuntimeConfig | dict[str, Any] | str | Path | None,
) -> RuntimeConfig:
    if isinstance(value, RuntimeConfig):
        return value.with_profile_defaults()
    if isinstance(value, dict):
        return RuntimeConfig.from_mapping(value)
    return RuntimeConfig.load(value)
