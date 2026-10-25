"""整边、部分边及一对多公共界面候选生成。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations, permutations, product
import math
from typing import Any

import numpy as np

from ..transforms import estimate_rigid
from .atomic_segments import (
    AtomicSegment,
    IntervalEndpointSource,
    make_atomic_segment,
)
from .preprocess import ProcessedPiece
from .directed_roles import (
    DirectedInterfaceContext,
    OppositeSideResult,
    build_directed_interface_context,
    directed_endpoint_context_cost,
    evaluate_opposite_side,
)
from .observation_uncertainty import (
    DEFAULT_HARD_ERROR_RADIUS_MM,
    NUMERICAL_EPSILON,
    ObservationUncertaintyProfile,
)
from .uncertain_geometry import UncertainPieceGeometry


@dataclass(frozen=True)
class InterfaceMatchQuality:
    score: float
    normalized_length_error: float
    angle_error: float
    context_error: float
    source_confidence: float
    ambiguity: float
    directed_endpoint_cost: float = 0.0
    ordered_adjacent_length_z: float = 0.0
    multi_model_angle_z: float = 0.0
    signed_turn_z: float = 0.0


@dataclass
class SharedInterfaceCandidate:
    interface_id: str
    piece_a: str
    piece_b: str | tuple[str, ...]
    segments_a: list[AtomicSegment]
    segments_b: list[AtomicSegment]
    interface_type: str
    support_line: list[tuple[float, float]]
    direction: tuple[float, float]
    total_length_a: float
    total_length_b: float
    coverage_ratio: float
    line_rms: float
    endpoint_rms: float
    interval_gap_mm: float
    opposite_side_valid: bool
    score: float
    composite_id: str | None = None
    constraint_rank: int = 2
    determines_full_pose: bool = True
    remaining_dof: int = 0
    endpoint_sources: tuple[str, ...] = ()
    junction_ids: tuple[str, ...] = ()
    geometric_explanation_score: float = 1.0
    outer_edge_risk: float = 0.0
    ambiguity_penalty: float = 0.0
    length_residual_zscore: float = 0.0
    direction_residual_zscore: float = 0.0
    support_line_residual_zscore: float = 0.0
    endpoint_residual_zscore: float = 0.0
    coverage_probability: float = 1.0
    junction_probability: float = 1.0
    opposite_side_probability: float = 1.0
    outer_edge_probability: float = 0.5
    total_log_likelihood: float = 0.0
    match_quality: InterfaceMatchQuality | None = None
    directed_context_a: DirectedInterfaceContext | None = None
    directed_context_b: DirectedInterfaceContext | None = None
    directed_endpoint_context_valid: bool = True
    directed_endpoint_cost: float = 0.0
    opposite_side_result: OppositeSideResult | None = None

    def as_json(self) -> dict[str, Any]:
        result = asdict(self)
        result["segments_a"] = [item.as_json() for item in self.segments_a]
        result["segments_b"] = [item.as_json() for item in self.segments_b]
        return result


@dataclass(frozen=True)
class PairwiseInterfacePlacement:
    interface_id: str
    first_piece: str
    first_segment: AtomicSegment
    second_piece: str
    second_segment: AtomicSegment
    interface_type: str
    composite_id: str | None
    score: float
    constraint_rank: int = 2
    determines_full_pose: bool = True
    remaining_dof: int = 0
    geometric_explanation_score: float = 1.0
    outer_edge_risk: float = 0.0
    ambiguity_penalty: float = 0.0
    relative_angle: float | None = None
    relative_translation: tuple[float, float] | None = None
    transform_residual: float | None = None
    deterministic_pose_signature: tuple | None = None
    quality_score: float = 0.0
    directed_context_a: DirectedInterfaceContext | None = None
    directed_context_b: DirectedInterfaceContext | None = None
    directed_endpoint_context_valid: bool = True
    directed_endpoint_cost: float = 0.0
    opposite_side_result: OppositeSideResult | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "interface_id": self.interface_id,
            "piece_a": self.first_piece,
            "piece_b": self.second_piece,
            "segments_a": [self.first_segment.as_json()],
            "segments_b": [self.second_segment.as_json()],
            "interface_type": self.interface_type,
            "composite_id": self.composite_id,
            "score": self.score,
            "constraint_rank": self.constraint_rank,
            "determines_full_pose": self.determines_full_pose,
            "remaining_dof": self.remaining_dof,
            "geometric_explanation_score": self.geometric_explanation_score,
            "outer_edge_risk": self.outer_edge_risk,
            "ambiguity_penalty": self.ambiguity_penalty,
            "relative_angle": self.relative_angle,
            "relative_translation": self.relative_translation,
            "transform_residual": self.transform_residual,
            "deterministic_pose_signature": self.deterministic_pose_signature,
            "directed_context_a": (
                self.directed_context_a.as_json()
                if self.directed_context_a is not None else None
            ),
            "directed_context_b": (
                self.directed_context_b.as_json()
                if self.directed_context_b is not None else None
            ),
            "directed_endpoint_context_valid": (
                self.directed_endpoint_context_valid
            ),
            "directed_endpoint_cost": self.directed_endpoint_cost,
            "opposite_side_result": (
                self.opposite_side_result.as_json()
                if self.opposite_side_result is not None else None
            ),
        }

    def reversed(self) -> "PairwiseInterfacePlacement":
        source = np.asarray(
            [self.first_segment.start_point, self.first_segment.end_point],
            dtype=np.float64,
        )
        target = np.asarray(
            [self.second_segment.end_point, self.second_segment.start_point],
            dtype=np.float64,
        )
        angle, translation, residual = estimate_rigid(source, target)
        reversed_side = (
            None
            if self.opposite_side_result is None
            else OppositeSideResult(
                valid=self.opposite_side_result.valid,
                ambiguous=self.opposite_side_result.ambiguous,
                signed_distance_a_mm=(
                    self.opposite_side_result.signed_distance_b_mm
                ),
                signed_distance_b_mm=(
                    self.opposite_side_result.signed_distance_a_mm
                ),
                probability=self.opposite_side_result.probability,
                reason=self.opposite_side_result.reason,
            )
        )
        return PairwiseInterfacePlacement(
            interface_id=self.interface_id,
            first_piece=self.second_piece,
            first_segment=self.second_segment,
            second_piece=self.first_piece,
            second_segment=self.first_segment,
            interface_type=self.interface_type,
            composite_id=self.composite_id,
            score=self.score,
            constraint_rank=self.constraint_rank,
            determines_full_pose=self.determines_full_pose,
            remaining_dof=self.remaining_dof,
            geometric_explanation_score=self.geometric_explanation_score,
            outer_edge_risk=self.outer_edge_risk,
            ambiguity_penalty=self.ambiguity_penalty,
            relative_angle=float(angle),
            relative_translation=(
                float(translation[0]), float(translation[1])
            ),
            transform_residual=float(residual),
            deterministic_pose_signature=(
                round(float(angle), 10),
                round(float(translation[0]), 10),
                round(float(translation[1]), 10),
            ),
            quality_score=self.quality_score,
            directed_context_a=self.directed_context_b,
            directed_context_b=self.directed_context_a,
            directed_endpoint_context_valid=(
                self.directed_endpoint_context_valid
            ),
            directed_endpoint_cost=self.directed_endpoint_cost,
            opposite_side_result=reversed_side,
        )


def _edge_refs(
    pieces: list[ProcessedPiece],
) -> list[tuple[ProcessedPiece, int, float]]:
    return [
        (piece, edge_index, float(length))
        for piece in pieces
        for edge_index, length in enumerate(piece.edge_lengths)
    ]


def _ranking_sigma(
    profile: ObservationUncertaintyProfile,
    length_mm: float,
) -> float:
    """返回仅用于排序的尺度，绝不作为接口候选的毫米硬门槛。

    随机拼图无法在求解前知道视觉或裁切误差。这里采用边自身长度
    的无量纲比例把不同长度的残差放到同一排序中；即使分数很差，
    候选仍会被保留给后续的全局几何评分。
    """

    del profile
    return float(max(0.35, 0.015 * length_mm))


def _outer_edge_risk(
    piece: ProcessedPiece,
    edge_index: int,
    profile: ObservationUncertaintyProfile,
) -> float:
    """该边被用作内拼缝时的外框风险，仅作软排序。

    一条边两端越接近直角，越可能属于目标矩形的外框；把它优先
    消耗为内部拼缝会让搜索很早失去正确的外框拓扑。这里返回的是
    *风险*（而不是旧实现中误命名的“外框概率”），不会据此删除候选。
    """

    context = build_directed_interface_context(
        piece,
        make_atomic_segment(piece.piece_id, piece.vertices, edge_index),
        profile,
    )
    outer_likelihood = 0.5 * (
        context.start.possible_outer_corner_probability
        + context.end.possible_outer_corner_probability
    )
    # ``placement_priority`` 会优先选择较小的风险；因此这里必须直接
    # 返回外框可能性，而非它的补值。补值会反过来优先消耗疑似外边。
    return float(outer_likelihood)


def _event_intervals(
    refs: list[tuple[ProcessedPiece, int, float]],
    profile: ObservationUncertaintyProfile,
) -> list[AtomicSegment]:
    """由物理顶点或其他物理边顶点投影事件生成可解释区间。"""
    intervals: list[AtomicSegment] = []
    for piece, edge_index, length in refs:
        events: dict[float, list[IntervalEndpointSource]] = {
            0.0: [
                IntervalEndpointSource(
                    "physical_vertex", f"{piece.piece_id}:vertex_{edge_index}"
                )
            ],
            1.0: [
                IntervalEndpointSource(
                    "physical_vertex",
                    f"{piece.piece_id}:vertex_{(edge_index + 1) % len(piece.vertices)}",
                )
            ],
        }
        for source_piece, source_edge, source_length in refs:
            if source_piece.piece_id == piece.piece_id:
                continue
            # 只有较短物理边才会在长边上产生内部终点。这里不以未知
            # 观测误差扩大或缩小候选池，长度残差仅留给软排序处理。
            if source_length >= length - 1e-9:
                continue
            ratio = float(source_length / max(length, 1e-12))
            if ratio < 0.05 or ratio > 0.95:
                continue
            source_id = (
                f"{source_piece.piece_id}:edge_{source_edge}"
                f"->projection_on:{piece.piece_id}:edge_{edge_index}"
            )
            for parameter in (ratio, 1.0 - ratio):
                key = round(parameter, 7)
                events.setdefault(key, []).append(
                    IntervalEndpointSource(
                        "projected_vertex",
                        source_id,
                        confidence=float(np.clip(
                            min(
                                piece.piece_confidence,
                                source_piece.piece_confidence,
                            ),
                            0.0,
                            1.0,
                        )),
                    )
                )
        parameters = sorted(events)
        # Keep a linear-size, source-explainable interval pool.  Enumerating
        # every pair of projected events is quadratic per edge and mostly
        # creates unrelated hypotheses.  Prefixes, suffixes and elementary
        # adjacent intervals cover full↔partial and T-junction termination.
        parameter_pairs = {
            (parameters[index], parameters[index + 1])
            for index in range(len(parameters) - 1)
        }
        parameter_pairs.update(
            (0.0, parameter)
            for parameter in parameters[1:-1]
        )
        parameter_pairs.update(
            (parameter, 1.0)
            for parameter in parameters[1:-1]
        )
        parameter_pairs.add((0.0, 1.0))
        for start, end in sorted(parameter_pairs):
                segment_length = (end - start) * length
                if segment_length < 5.0:
                    continue
                intervals.append(
                    make_atomic_segment(
                        piece.piece_id,
                        piece.vertices,
                        edge_index,
                        start,
                        end,
                        start_sources=events[start],
                        end_sources=events[end],
                        confidence=min(
                            source.confidence
                            for source in events[start] + events[end]
                        ),
                        require_sources=True,
                    )
                )
    return intervals


def _direction(segment: AtomicSegment) -> tuple[float, float]:
    delta = np.asarray(segment.end_point) - np.asarray(segment.start_point)
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-12:
        return (0.0, 0.0)
    return (float(delta[0] / norm), float(delta[1] / norm))


def _candidate(
    interface_id: str,
    piece_a: str,
    piece_b: str | tuple[str, ...],
    segments_a: list[AtomicSegment],
    segments_b: list[AtomicSegment],
    interface_type: str,
    residual: float,
    composite_id: str | None = None,
    *,
    constraint_rank: int = 2,
    determines_full_pose: bool = True,
    remaining_dof: int = 0,
    geometric_explanation_score: float = 1.0,
    outer_edge_risk: float = 0.0,
    ambiguity_penalty: float = 0.0,
    junction_ids: tuple[str, ...] = (),
    uncertainty_profile: ObservationUncertaintyProfile | None = None,
) -> SharedInterfaceCandidate:
    total_a = float(sum(item.length_mm for item in segments_a))
    total_b = float(sum(item.length_mm for item in segments_b))
    support = [segments_a[0].start_point, segments_a[-1].end_point]
    denominator = max(total_a, total_b, 1e-12)
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
    sigma = _ranking_sigma(profile, denominator)
    z_length = float(abs(residual) / sigma)
    coverage = float(np.exp(-0.5 * z_length * z_length))
    return SharedInterfaceCandidate(
        interface_id=interface_id,
        piece_a=piece_a,
        piece_b=piece_b,
        segments_a=segments_a,
        segments_b=segments_b,
        interface_type=interface_type,
        support_line=support,
        direction=_direction(segments_a[0]),
        total_length_a=total_a,
        total_length_b=total_b,
        coverage_ratio=float(min(total_a, total_b) / denominator),
        line_rms=0.0,
        endpoint_rms=float(abs(residual)),
        interval_gap_mm=float(abs(residual)),
        opposite_side_valid=False,
        score=z_length,
        composite_id=composite_id,
        constraint_rank=constraint_rank,
        determines_full_pose=determines_full_pose,
        remaining_dof=remaining_dof,
        endpoint_sources=tuple(
            sorted(
                {
                    source.source_type
                    for segment in segments_a + segments_b
                    for source in segment.start_sources + segment.end_sources
                }
            )
        ),
        junction_ids=junction_ids,
        geometric_explanation_score=geometric_explanation_score,
        outer_edge_risk=outer_edge_risk,
        ambiguity_penalty=ambiguity_penalty,
        length_residual_zscore=z_length,
        endpoint_residual_zscore=z_length,
        coverage_probability=coverage,
        junction_probability=float(np.exp(-ambiguity_penalty)),
        outer_edge_probability=float(np.clip(
            1.0 - outer_edge_risk, 0.0, 1.0
        )),
        total_log_likelihood=float(
            -0.5 * z_length * z_length - ambiguity_penalty
        ),
        opposite_side_probability=0.5,
        opposite_side_result=OppositeSideResult(
            False,
            True,
            0.0,
            0.0,
            0.5,
            "not_evaluated_for_composite_summary",
        ),
    )


def build_shared_interface_candidates(
    pieces: list[ProcessedPiece],
    noise_mm: float = 0.0,
    *,
    include_generic_partial: bool = False,
    diagnostics_level: str = "full",
    uncertainty_profile: ObservationUncertaintyProfile | None = None,
) -> tuple[
    list[SharedInterfaceCandidate],
    list[PairwiseInterfacePlacement],
    dict[str, Any],
]:
    """生成完整边匹配及由一条长边和两条短边组成的虚拟 T 界面。"""
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
    refs = _edge_refs(pieces)
    pieces_by_id = {piece.piece_id: piece for piece in pieces}
    uncertain_by_id = {
        piece.piece_id: UncertainPieceGeometry.from_piece(
            piece, float(profile.typical_sigma_mm or 0.35)
        )
        for piece in pieces
    }
    candidates: list[SharedInterfaceCandidate] = []
    placements: list[PairwiseInterfacePlacement] = []
    seen_pairwise: set[tuple] = set()
    counter = 0

    def match_quality(
        first: AtomicSegment,
        second: AtomicSegment,
        residual: float,
        outer_edge_risk: float,
        ambiguity_penalty: float,
        first_context: DirectedInterfaceContext,
        second_context: DirectedInterfaceContext,
    ) -> InterfaceMatchQuality:
        first_piece = pieces_by_id[first.piece_id]
        second_piece = pieces_by_id[second.piece_id]
        length_scale = max(
            first.length_mm, second.length_mm, NUMERICAL_EPSILON
        )
        normalized_length = min(
            2.5,
            abs(float(residual))
            / max(_ranking_sigma(profile, length_scale), NUMERICAL_EPSILON),
        )
        directed_cost, directed_components = directed_endpoint_context_cost(
            first_context, second_context
        )
        angle_error = float(1.0 - math.exp(
            -0.5 * min(directed_components["multi_model_angle_z"], 8.0) ** 2
        ))
        context_error = float(1.0 - math.exp(
            -0.5 * min(
                directed_components["ordered_adjacent_length_z"], 8.0
            ) ** 2
        ))
        source_confidence = float(np.clip(min(
            first.confidence,
            second.confidence,
            first_piece.piece_confidence,
            second_piece.piece_confidence,
        ), 0.0, 1.0))
        ambiguity = float(np.clip(
            0.5 * outer_edge_risk + ambiguity_penalty,
            0.0,
            1.0,
        ))
        cost = (
            0.38 * normalized_length
            + 0.30 * min(directed_cost / 4.0, 1.0)
            + 0.12 * (1.0 - source_confidence)
            + 0.12 * ambiguity
        )
        return InterfaceMatchQuality(
            score=float(np.clip(1.0 - cost, 0.0, 1.0)),
            normalized_length_error=float(normalized_length),
            angle_error=float(angle_error),
            context_error=float(context_error),
            source_confidence=source_confidence,
            ambiguity=ambiguity,
            directed_endpoint_cost=float(directed_cost),
            ordered_adjacent_length_z=float(
                directed_components["ordered_adjacent_length_z"]
            ),
            multi_model_angle_z=float(
                directed_components["multi_model_angle_z"]
            ),
            signed_turn_z=float(directed_components["signed_turn_z"]),
        )

    def add_pairwise(
        first: AtomicSegment,
        second: AtomicSegment,
        kind: str,
        composite_id: str | None,
        residual: float,
        *,
        constraint_rank: int = 2,
        determines_full_pose: bool = True,
        remaining_dof: int = 0,
        geometric_explanation_score: float = 1.0,
        outer_edge_risk: float = 0.0,
        ambiguity_penalty: float = 0.0,
        junction_ids: tuple[str, ...] = (),
    ) -> None:
        nonlocal counter
        key = (
            first.piece_id,
            first.edge_index,
            round(first.t_start, 9),
            round(first.t_end, 9),
            second.piece_id,
            second.edge_index,
            round(second.t_start, 9),
            round(second.t_end, 9),
            composite_id,
        )
        reverse_key = (
            key[4], key[5], key[6], key[7],
            key[0], key[1], key[2], key[3],
            key[8],
        )
        if key in seen_pairwise or reverse_key in seen_pairwise:
            return
        seen_pairwise.add(key)
        first_context = build_directed_interface_context(
            pieces_by_id[first.piece_id],
            first,
            profile,
            uncertain_geometry=uncertain_by_id[first.piece_id],
        )
        second_context = build_directed_interface_context(
            pieces_by_id[second.piece_id],
            second,
            profile,
            uncertain_geometry=uncertain_by_id[second.piece_id],
        )
        directed_cost, _directed_components = directed_endpoint_context_cost(
            first_context, second_context
        )
        relative = (
            estimate_rigid(
                np.asarray(
                    [second.start_point, second.end_point],
                    dtype=np.float64,
                ),
                np.asarray(
                    [first.end_point, first.start_point],
                    dtype=np.float64,
                ),
            )
            if determines_full_pose else None
        )
        opposite_side = (
            evaluate_opposite_side(
                pieces_by_id[first.piece_id],
                first,
                pieces_by_id[second.piece_id],
                second,
                float(relative[0]),
                relative[1],
            )
            if relative is not None
            else OppositeSideResult(
                False, True, 0.0, 0.0, 0.5,
                "remaining_degree_of_freedom",
            )
        )
        # 一对二的长边半段含有一个虚拟分割端点，不能套用完整物理边
        # 的双端角色硬门槛。复合组会在搜索中原子开放并由全局外框、
        # 拼缝覆盖和 T 节点闭合统一裁决；这里仍保留该结果用于软排序。
        if (
            composite_id is None
            and not opposite_side.valid
            and not opposite_side.ambiguous
        ):
            return
        counter += 1
        interface_id = f"interface_{counter:05d}"
        candidate = _candidate(
            interface_id,
            first.piece_id,
            second.piece_id,
            [first],
            [second],
            kind,
            residual,
            composite_id,
            constraint_rank=constraint_rank,
            determines_full_pose=determines_full_pose,
            remaining_dof=remaining_dof,
            geometric_explanation_score=geometric_explanation_score,
            outer_edge_risk=outer_edge_risk,
            ambiguity_penalty=ambiguity_penalty,
            junction_ids=junction_ids,
            uncertainty_profile=profile,
        )
        quality = match_quality(
            first,
            second,
            residual,
            outer_edge_risk,
            ambiguity_penalty,
            first_context,
            second_context,
        )
        candidate.match_quality = quality
        candidate.directed_context_a = first_context
        candidate.directed_context_b = second_context
        candidate.directed_endpoint_cost = float(directed_cost)
        candidate.directed_endpoint_context_valid = bool(
            directed_cost <= 12.0
        )
        candidate.opposite_side_valid = bool(opposite_side.valid)
        candidate.opposite_side_probability = float(
            opposite_side.probability
        )
        candidate.opposite_side_result = opposite_side
        candidate.score = float(
            abs(residual)
            / max(
                _ranking_sigma(
                    profile,
                    max(first.length_mm, second.length_mm),
                ),
                NUMERICAL_EPSILON,
            )
        )
        candidates.append(candidate)
        relative_angle = float(relative[0]) if relative is not None else None
        relative_translation = (
            (float(relative[1][0]), float(relative[1][1]))
            if relative is not None else None
        )
        transform_residual = (
            float(relative[2]) if relative is not None else None
        )
        placements.append(PairwiseInterfacePlacement(
            interface_id=interface_id,
            first_piece=first.piece_id,
            first_segment=first,
            second_piece=second.piece_id,
            second_segment=second,
            interface_type=kind,
            composite_id=composite_id,
            score=candidate.score,
            constraint_rank=constraint_rank,
            determines_full_pose=determines_full_pose,
            remaining_dof=remaining_dof,
            geometric_explanation_score=geometric_explanation_score,
            outer_edge_risk=outer_edge_risk,
            ambiguity_penalty=ambiguity_penalty,
            relative_angle=relative_angle,
            relative_translation=relative_translation,
            transform_residual=transform_residual,
            deterministic_pose_signature=(
                (
                    round(relative_angle, 10),
                    round(relative_translation[0], 10),
                    round(relative_translation[1], 10),
                )
                if relative_angle is not None
                and relative_translation is not None else None
            ),
            quality_score=quality.score,
            directed_context_a=first_context,
            directed_context_b=second_context,
            directed_endpoint_context_valid=bool(directed_cost <= 12.0),
            directed_endpoint_cost=float(directed_cost),
            opposite_side_result=opposite_side,
        ))

    # 标准整边 ↔ 整边。
    for (first_piece, first_edge, first_length), (
        second_piece,
        second_edge,
        second_length,
    ) in combinations(refs, 2):
        if first_piece.piece_id == second_piece.piece_id:
            continue
        residual = first_length - second_length
        # 不用“长度差小于某个默认毫米数”提前删除接口。每一对完整边
        # 都是有限的拓扑假设；后续按相对残差排序，并由完整布局裁决。
        add_pairwise(
            make_atomic_segment(
                first_piece.piece_id, first_piece.vertices, first_edge
            ),
            make_atomic_segment(
                second_piece.piece_id, second_piece.vertices, second_edge
            ),
            "one_to_one",
            None,
            residual,
            outer_edge_risk=(
                _outer_edge_risk(first_piece, first_edge, profile)
                + _outer_edge_risk(second_piece, second_edge, profile)
            )
            / 2.0,
        )

    # 一条长边 ↔ 两条短边。noise>0 仍保留该结构，使用概率残差排序；
    # 不能因噪声删除真实 T 节点。
    composite_count = 0
    composite_candidates: list[SharedInterfaceCandidate] = []
    for long_piece, long_edge, long_length in refs:
        short_refs = [
            item
            for item in refs
            if item[0].piece_id != long_piece.piece_id
            and item[2] < long_length - 1e-9
        ]
        for first_short, second_short in combinations(short_refs, 2):
            first_piece, first_edge, first_length = first_short
            second_piece, second_edge, second_length = second_short
            if (
                first_piece.piece_id == second_piece.piece_id
            ):
                continue
            residual = long_length - first_length - second_length
            for order in ((first_short, second_short), (second_short, first_short)):
                composite_count += 1
                composite_id = f"composite_{composite_count:05d}"
                left_piece, left_edge, left_length = order[0]
                right_piece, right_edge, right_length = order[1]
                # 观测长度不可能严格守恒。按两条短边的总长归一化分割，
                # 将残差均匀摊到整个拼缝，避免全部误差落在最后一片。
                split_t = float(
                    left_length / max(left_length + right_length, 1e-12)
                )
                long_left = make_atomic_segment(
                    long_piece.piece_id,
                    long_piece.vertices,
                    long_edge,
                    0.0,
                    split_t,
                )
                long_right = make_atomic_segment(
                    long_piece.piece_id,
                    long_piece.vertices,
                    long_edge,
                    split_t,
                    1.0,
                )
                short_left = make_atomic_segment(
                    left_piece.piece_id, left_piece.vertices, left_edge
                )
                short_right = make_atomic_segment(
                    right_piece.piece_id, right_piece.vertices, right_edge
                )
                composite = _candidate(
                    composite_id,
                    long_piece.piece_id,
                    (left_piece.piece_id, right_piece.piece_id),
                    [long_left, long_right],
                    [short_left, short_right],
                    "one_to_two",
                    residual,
                    composite_id,
                    uncertainty_profile=profile,
                )
                composite_candidates.append(composite)
                add_pairwise(
                    long_left,
                    short_left,
                    "one_to_two_part",
                    composite_id,
                    residual,
                )
                add_pairwise(
                    long_right,
                    short_right,
                    "one_to_two_part",
                    composite_id,
                    residual,
                )

    # 一条长边 ↔ 三条短边。四片赛题中这是合法且必要的拓扑：一片的
    # 内边可能依次邻接其余三片。只按每条长边的相对长度残差保留有限个
    # 最优离散组合，再将入选组原子展开为三个定位约束；最终仍由全局
    # 矩形、无重叠、外边角色与节点闭合裁决。
    if len(pieces) == 4 and include_generic_partial:
        triple_hypothesis_limit_per_long_edge = 36
        for long_piece, long_edge, long_length in refs:
            other_piece_ids = sorted(
                piece.piece_id for piece in pieces
                if piece.piece_id != long_piece.piece_id
            )
            choices = [
                [
                    item for item in refs
                    if item[0].piece_id == piece_id
                    and item[2] < long_length - 1e-9
                ]
                for piece_id in other_piece_ids
            ]
            if any(not rows for rows in choices):
                continue
            hypotheses: list[tuple[float, tuple]] = []
            for selected_shorts in product(*choices):
                total_short = float(sum(item[2] for item in selected_shorts))
                residual = float(long_length - total_short)
                for order in permutations(selected_shorts):
                    hypotheses.append((abs(residual), order))
            hypotheses.sort(key=lambda row: (
                row[0] / max(long_length, 1e-12),
                tuple(
                    (item[0].piece_id, item[1]) for item in row[1]
                ),
            ))
            for _absolute_residual, order in hypotheses[
                :triple_hypothesis_limit_per_long_edge
            ]:
                total_short = float(sum(item[2] for item in order))
                residual = float(long_length - total_short)
                composite_count += 1
                composite_id = f"composite_{composite_count:05d}"
                long_segments: list[AtomicSegment] = []
                short_segments: list[AtomicSegment] = []
                cursor = 0.0
                for index, (short_piece, short_edge, short_length) in enumerate(
                    order
                ):
                    start_t = cursor / max(total_short, 1e-12)
                    cursor += short_length
                    end_t = (
                        1.0
                        if index == len(order) - 1
                        else cursor / max(total_short, 1e-12)
                    )
                    long_segments.append(make_atomic_segment(
                        long_piece.piece_id,
                        long_piece.vertices,
                        long_edge,
                        start_t,
                        end_t,
                    ))
                    short_segments.append(make_atomic_segment(
                        short_piece.piece_id,
                        short_piece.vertices,
                        short_edge,
                    ))
                composite_candidates.append(_candidate(
                    composite_id,
                    long_piece.piece_id,
                    tuple(item[0].piece_id for item in order),
                    long_segments,
                    short_segments,
                    "one_to_three",
                    residual,
                    composite_id,
                    uncertainty_profile=profile,
                ))
                for long_segment, short_segment in zip(
                    long_segments, short_segments
                ):
                    add_pairwise(
                        long_segment,
                        short_segment,
                        "one_to_three_part",
                        composite_id,
                        residual,
                    )

    composite_candidates.sort(
        key=lambda item: (
            abs(1.0 - item.coverage_ratio),
            item.score,
            item.interface_id,
        )
    )
    # 一对二是一个原子 T 节点假设。此处保留全部有限组合，真正参与
    # 某一搜索阶段的数量由 ``max_matches_per_edge`` 渐进打开。不能在
    # 候选生成阶段按局部长度排名永久删除真实拓扑，否则 recovery 即使
    # 增加预算也无法找回早期被剪掉的接口。
    composite_groups: dict[tuple[str, int], list[SharedInterfaceCandidate]] = {}
    for item in composite_candidates:
        composite_groups.setdefault(
            (item.piece_a, item.segments_a[0].edge_index), []
        ).append(item)
    composite_budget_per_long_edge = max(
        (len(rows) for rows in composite_groups.values()), default=0
    )
    discarded_composite_count = 0
    candidates.extend(composite_candidates)
    generic_partial_count = 0
    partial_to_partial_count = 0
    sourced_interval_count = 0
    rejected_unsourced_interval_count = 0
    rejected_unrelated_projection_count = 0
    if include_generic_partial:
        event_intervals = _event_intervals(refs, profile)
        sourced_interval_count = len(event_intervals)
        candidate_pairs: list[tuple[AtomicSegment, AtomicSegment]] = []
        # 对每个原子区间保留若干个“不同纸片/边”的最近长度候选。它是
        # 计算预算上的多样性上限，而不是误差半径：任何长度残差都可入选，
        # 因而未知的 1--2 mm 误差不会把正确接口直接剪掉。
        per_interval_limit = 12
        for first in event_intervals:
            alternatives = [
                second
                for second in event_intervals
                if second.piece_id != first.piece_id
            ]
            alternatives.sort(key=lambda second: (
                abs(first.length_mm - second.length_mm)
                / max(first.length_mm, second.length_mm, 1e-9),
                second.piece_id,
                second.edge_index,
                second.t_start,
                second.t_end,
            ))
            for second in alternatives[:per_interval_limit]:
                key = (
                    first.piece_id,
                    first.edge_index,
                    first.t_start,
                    first.t_end,
                )
                other_key = (
                    second.piece_id,
                    second.edge_index,
                    second.t_start,
                    second.t_end,
                )
                if key < other_key:
                    candidate_pairs.append((first, second))
        for first, second in candidate_pairs:
            if first.piece_id == second.piece_id:
                continue
            if (
                first.t_start <= 1e-12
                and first.t_end >= 1.0 - 1e-12
                and second.t_start <= 1e-12
                and second.t_end >= 1.0 - 1e-12
            ):
                continue
            residual = first.length_mm - second.length_mm
            first_partial = (
                first.t_start > 1e-12 or first.t_end < 1.0 - 1e-12
            )
            second_partial = (
                second.t_start > 1e-12 or second.t_end < 1.0 - 1e-12
            )
            if first_partial and second_partial:
                # 两端都是投影出来的任意区间时，没有一个真实顶点锚定
                # 该拼缝的终点。这正是此前大量“边贴边随机碰撞”的来源。
                # 真正的一对二节点已由上面的显式复合拓扑覆盖；其余
                # partial↔partial 不进入随机搜索的候选池。
                rejected_unsourced_interval_count += 1
                continue
            elif first_partial:
                kind = "partial_to_full"
                ambiguity = 0.1
                constraint_rank = 2
                determines_full_pose = True
                remaining_dof = 0
            else:
                kind = "full_to_partial"
                ambiguity = 0.1
                constraint_rank = 2
                determines_full_pose = True
                remaining_dof = 0
            if first_partial != second_partial:
                partial_segment = first if first_partial else second
                full_segment = second if first_partial else first
                required_prefix = (
                    f"{full_segment.piece_id}:edge_"
                    f"{full_segment.edge_index}->projection_on:"
                )
                projection_sources = [
                    source.source_id
                    for source in partial_segment.start_sources
                    + partial_segment.end_sources
                    if source.source_type == "projected_vertex"
                ]
                if not any(
                    source_id.startswith(required_prefix)
                    for source_id in projection_sources
                ):
                    rejected_unrelated_projection_count += 1
                    continue
            before = len(placements)
            junction_ids = tuple(
                sorted(
                    {
                        source.source_id
                        for segment in (first, second)
                        for source in segment.start_sources
                        + segment.end_sources
                        if source.source_type != "physical_vertex"
                    }
                )
            )
            add_pairwise(
                first,
                second,
                kind,
                None,
                residual,
                constraint_rank=constraint_rank,
                determines_full_pose=determines_full_pose,
                remaining_dof=remaining_dof,
                geometric_explanation_score=min(
                    first.confidence, second.confidence
                ),
                ambiguity_penalty=ambiguity,
                junction_ids=junction_ids,
            )
            generic_partial_count += len(placements) - before
    graph_edges = {
        frozenset((placement.first_piece, placement.second_piece))
        for placement in placements
    }
    ids = [piece.piece_id for piece in pieces]
    reached = {ids[0]} if ids else set()
    changed = True
    while changed:
        changed = False
        for edge in graph_edges:
            if edge & reached and not edge <= reached:
                reached.update(edge)
                changed = True
    diagnostics = {
        "legacy_noise_label_present": noise_mm is not None,
        "legacy_noise_label_used_by_solver": False,
        "uncertainty_profile": profile.as_json(),
        "shared_interface_candidate_count": len(candidates),
        "pairwise_interface_placement_count": len(placements),
        "one_to_one_candidate_count": sum(
            item.interface_type == "one_to_one" for item in candidates
        ),
        "one_to_many_candidate_count": len(composite_candidates),
        "virtual_t_junction_count": len(composite_candidates),
        "one_to_two_budget_per_long_edge": composite_budget_per_long_edge,
        "one_to_two_candidates_pruned_by_topology_budget": (
            discarded_composite_count
        ),
        "generic_partial_candidate_count": generic_partial_count,
        "partial_to_partial_candidate_count": partial_to_partial_count,
        "sourced_atomic_interval_count": sourced_interval_count,
        "rejected_unsourced_interval_count": rejected_unsourced_interval_count,
        "rejected_unrelated_projection_count": (
            rejected_unrelated_projection_count
        ),
        "interface_graph_connected": len(reached) == len(ids),
        "interface_graph_reached_piece_ids": sorted(reached),
    }
    if diagnostics_level == "full":
        diagnostic_limit = 250
        diagnostics["interface_candidates"] = [
            item.as_json()
            for item in sorted(
                candidates, key=lambda value: value.score
            )[:diagnostic_limit]
        ]
        diagnostics["interface_candidates_truncated"] = (
            len(candidates) > diagnostic_limit
        )
    elif diagnostics_level == "summary":
        diagnostics["best_interface_candidates"] = [
            item.as_json()
            for item in sorted(candidates, key=lambda value: value.score)[:10]
        ]
    return candidates, placements, diagnostics


def placement_segment_points(
    placement: PairwiseInterfacePlacement,
    piece_id: str,
) -> np.ndarray:
    segment = (
        placement.first_segment
        if placement.first_piece == piece_id
        else placement.second_segment
    )
    return np.asarray([segment.start_point, segment.end_point], dtype=float)
