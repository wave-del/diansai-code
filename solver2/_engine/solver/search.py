"""2～4 片随机拼图的公共界面图搜索与真实几何评分。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import heapq
import math
from time import perf_counter
from typing import Any, Callable

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from ..geometry import pairwise_overlap
from ..runtime_config import RuntimeConfig
from ..transforms import compose_transform, estimate_rigid, transform_points
from .atomic_segments import interval_is_available
from .atomic_segments import make_atomic_segment
from .interface_matching import (
    PairwiseInterfacePlacement,
    build_shared_interface_candidates,
)
from .global_interface_search import (
    canonical_pose_state,
    feasible_width_interval,
    placement_priority,
)
from .preprocess import ProcessedPiece
from .geometry_cache import CachedPoseGeometry
from .random_safety import (
    RandomCandidatePool,
    RandomSafetyAssessment,
    assess_random_assembly,
)
from .observation_uncertainty import (
    DEFAULT_HARD_ERROR_RADIUS_MM,
    ObservationUncertaintyProfile,
)
from .progressive_search import (
    ProgressiveSearchStage,
    interface_ambiguity_metrics,
    stages_from_config,
)
from .placed_interface_discovery import discover_interfaces_in_placed_state
from .partition_search import enumerate_partition_candidates
from .optimizer import optimize_generic_candidate


@dataclass
class RandomPose:
    piece_id: str
    angle: float
    translation: np.ndarray
    target_vertices_local: np.ndarray
    top_edge_index: int = -1
    matched_prev_edge: int | None = None
    matched_next_edge: int | None = None
    geometry: CachedPoseGeometry | None = None


@dataclass
class GeometryCandidate:
    poses: list[RandomPose]
    total: float
    width: float
    height: float
    fill_ratio: float
    overlap: float
    uncovered: float
    edge_rms: float
    outside: float = 0.0
    matches: list[tuple[str, int, str, int]] = field(default_factory=list)
    interfaces: list[PairwiseInterfacePlacement] = field(default_factory=list)
    geometry_valid: bool = False
    geometry_score: float = 0.0
    appearance_score: float = 0.0
    legal_complete_candidate_count: int = 0
    cycle_residual_mm: float = 0.0
    admissibility: RandomSafetyAssessment | None = None
    generation_method: str = "directed_edge_chain_reverse_partition"
    topology_template: str = ""
    partition_evidence: list[dict[str, Any]] = field(default_factory=list)


# Backwards-compatible public name.  Every instance emitted by solve_random is
# now an admitted GeometryCandidate, never merely a fully placed search state.
RandomCandidate = GeometryCandidate


@dataclass(frozen=True)
class SearchRunStats:
    anchor_index: int
    candidate_count: int
    search_nodes: int
    budget_exhausted: bool
    wall_time_limited: bool
    callback_stopped: bool
    elapsed_seconds: float


def _segment_points(
    pose: RandomPose, placement: PairwiseInterfacePlacement
) -> np.ndarray:
    segment = (
        placement.first_segment
        if placement.first_piece == pose.piece_id
        else placement.second_segment
    )
    points = pose.target_vertices_local
    start = points[segment.edge_index]
    end = points[(segment.edge_index + 1) % len(points)]
    return np.asarray(
        [
            start + segment.t_start * (end - start),
            start + segment.t_end * (end - start),
        ],
        dtype=float,
    )


def _interface_rms(
    placement: PairwiseInterfacePlacement, by_id: dict[str, RandomPose]
) -> float:
    first = _segment_points(by_id[placement.first_piece], placement)
    second = _segment_points(by_id[placement.second_piece], placement)
    errors = np.vstack((first[0] - second[1], first[1] - second[0]))
    return float(np.sqrt(np.mean(np.square(errors))))


def _canonical_candidates(
    poses: dict[int, RandomPose],
    interfaces: list[PairwiseInterfacePlacement],
    appearance: dict[tuple[str, int, str, int], float],
) -> list[RandomCandidate]:
    polygons = [Polygon(p.target_vertices_local) for p in poses.values()]
    union = unary_union(polygons)
    if union.is_empty:
        return []
    rectangle = union.minimum_rotated_rectangle
    coords = np.asarray(rectangle.exterior.coords[:-1], dtype=float)
    directions = [coords[(index + 1) % 4] - coords[index] for index in range(4)]
    results: list[RandomCandidate] = []
    for direction in directions:
        angle = -math.degrees(math.atan2(direction[1], direction[0]))
        rotated = {
            index: transform_points(p.target_vertices_local, angle)
            for index, p in poses.items()
        }
        stacked = np.vstack(list(rotated.values()))
        shift = -stacked.min(axis=0)
        canonical: list[RandomPose] = []
        for index, pose in poses.items():
            combined_angle, combined_translation = compose_transform(
                pose.angle, pose.translation, angle, shift
            )
            canonical.append(
                RandomPose(
                    pose.piece_id,
                    combined_angle,
                    combined_translation,
                    rotated[index] + shift,
                )
            )
        scored = _score(canonical, interfaces, appearance)
        if scored is not None:
            results.append(scored)
    return results


def _score(
    poses: list[RandomPose],
    interfaces: list[PairwiseInterfacePlacement],
    appearance: dict[tuple[str, int, str, int], float],
) -> RandomCandidate | None:
    polygons = [Polygon(p.target_vertices_local) for p in poses]
    if any(not polygon.is_valid or polygon.area <= 1.0 for polygon in polygons):
        return None
    union = unary_union(polygons)
    minx, miny, maxx, maxy = union.bounds
    width, height = float(maxx - minx), float(maxy - miny)
    if not (42.0 <= width <= 128.0 and 42.0 <= height <= 128.0):
        return None
    rectangle = box(minx, miny, maxx, maxy)
    overlap = pairwise_overlap(polygons)
    uncovered = float(rectangle.difference(union).area)
    outside = float(union.difference(rectangle).area)
    fill = (
        float(union.intersection(rectangle).area / rectangle.area)
        if rectangle.area
        else 0.0
    )
    by_id = {pose.piece_id: pose for pose in poses}
    interface_errors = [
        _interface_rms(interface, by_id) for interface in interfaces
    ]
    edge_rms = (
        float(np.sqrt(np.mean(np.square(interface_errors))))
        if interface_errors
        else 0.0
    )
    connected = (
        union.geom_type == "Polygon"
        or unary_union(
            [polygon.buffer(1e-6) for polygon in polygons]
        ).geom_type
        == "Polygon"
    )
    legal = (
        90.0 - 1e-5 <= width <= 120.0 + 1e-5
        and 50.0 - 1e-5 <= height <= 90.0 + 1e-5
    )
    strict_valid = (
        connected
        and legal
        and overlap <= 1e-5
        and uncovered <= 1e-4
        and outside <= 1e-4
        and edge_rms <= 1e-5
    )
    boundary_complexity = (
        max(0, len(union.convex_hull.exterior.coords) - 5)
        if connected
        else 20
    )
    dimension_penalty = (
        max(0.0, 90.0 - width) ** 2
        + max(0.0, width - 120.0) ** 2
        + max(0.0, 50.0 - height) ** 2
        + max(0.0, height - 90.0) ** 2
    )
    interface_model_penalty = sum(
        (1.0 - float(item.quality_score))
        + 0.2 * min(float(item.score), 5.0)
        + 0.25 * float(item.ambiguity_penalty)
        + 0.10 * float(item.outer_edge_risk)
        for item in interfaces
    )
    geometry_score = (
        10000.0 * (0 if strict_valid else 1)
        + 8.0 * overlap
        + uncovered
        + 10.0 * outside
        + 3.0 * edge_rms
        + 5.0 * dimension_penalty
        + boundary_complexity
        + 2.0 * interface_model_penalty
    )
    appearance_scores = [
        appearance.get(
            (
                item.first_piece,
                item.first_segment.edge_index,
                item.second_piece,
                item.second_segment.edge_index,
            ),
            appearance.get(
                (
                    item.second_piece,
                    item.second_segment.edge_index,
                    item.first_piece,
                    item.first_segment.edge_index,
                ),
                0.0,
            ),
        )
        for item in interfaces
    ]
    appearance_score = (
        float(np.mean(appearance_scores)) if appearance_scores else 0.0
    )
    matches = [
        (
            item.first_piece,
            item.first_segment.edge_index,
            item.second_piece,
            item.second_segment.edge_index,
        )
        for item in interfaces
    ]
    return RandomCandidate(
        poses=poses,
        total=float(geometry_score - 5.0 * appearance_score),
        width=width,
        height=height,
        fill_ratio=fill,
        overlap=float(overlap),
        uncovered=uncovered,
        edge_rms=edge_rms,
        outside=outside,
        matches=matches,
        interfaces=list(interfaces),
        geometry_valid=strict_valid,
        geometry_score=float(geometry_score),
        appearance_score=appearance_score,
        cycle_residual_mm=max(interface_errors, default=0.0),
    )


def _occupied_from_interfaces(
    interfaces: list[PairwiseInterfacePlacement],
) -> dict[tuple[str, int], list[tuple[float, float]]]:
    occupied: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for item in interfaces:
        for piece_id, segment in (
            (item.first_piece, item.first_segment),
            (item.second_piece, item.second_segment),
        ):
            occupied.setdefault((piece_id, segment.edge_index), []).append(
                (segment.t_start, segment.t_end)
            )
    return occupied


def _augment_closed_interfaces(
    poses: dict[int, RandomPose],
    selected: list[PairwiseInterfacePlacement],
    all_placements: list[PairwiseInterfacePlacement],
    uncertainty_profile: ObservationUncertaintyProfile,
) -> list[PairwiseInterfacePlacement]:
    by_id = {pose.piece_id: pose for pose in poses.values()}
    result = list(selected)
    selected_ids = {item.interface_id for item in selected}
    occupied = _occupied_from_interfaces(result)
    # “闭环”只能补充当前刚体落位中已经数值重合的接口。若用观测误差
    # 半径把相距近 1 mm 的边也声明为闭合，粗布局中的间隙/重叠会被
    # 误写成确定拓扑，随后又无法被实际接触发现证实。视觉误差由后续
    # soft_l1 处理，不在此处把近邻升级为拼缝。
    del uncertainty_profile
    tolerance = 1e-5
    for item in all_placements:
        if item.interface_id in selected_ids:
            continue
        if not _composite_is_compatible(result, item):
            continue
        if item.first_piece not in by_id or item.second_piece not in by_id:
            continue
        if _interface_rms(item, by_id) > tolerance:
            continue
        segments = (
            (item.first_piece, item.first_segment),
            (item.second_piece, item.second_segment),
        )
        if not all(
            interval_is_available(
                occupied.get((piece_id, segment.edge_index), []),
                (segment.t_start, segment.t_end),
                tolerance=1e-7,
            )
            for piece_id, segment in segments
        ):
            continue
        result.append(item)
        selected_ids.add(item.interface_id)
        for piece_id, segment in segments:
            occupied.setdefault((piece_id, segment.edge_index), []).append(
                (segment.t_start, segment.t_end)
            )
    return result


def _placements_for_stage(
    placements: list[PairwiseInterfacePlacement],
    stage: ProgressiveSearchStage,
) -> list[PairwiseInterfacePlacement]:
    """按物理边渐进开放接口，同时保持一对二拓扑的原子性。

    一对二候选由两个 ``one_to_two_part`` 组成。旧逻辑把两半分别排名，
    会让同一长边的 Top-K 被不同复合组的半边占满，最终搜索出并不存在
    的混搭 T 节点。这里以 ``composite_id`` 为排名单元；一个复合组入选
    时同时开放它的两半，并且只在其长边上占用一个 Top-K 名额。
    """

    atomic_groups: dict[str, list[PairwiseInterfacePlacement]] = {}
    for item in placements:
        if not item.determines_full_pose:
            continue
        group_id = item.composite_id or item.interface_id
        atomic_groups.setdefault(group_id, []).append(item)

    edge_groups: dict[tuple[str, int], set[str]] = {}
    for group_id, rows in atomic_groups.items():
        if rows[0].composite_id is not None:
            long_edge = _composite_long_edge(rows[0])
            if long_edge is not None:
                edge_groups.setdefault(long_edge, set()).add(group_id)
            continue
        item = rows[0]
        edge_groups.setdefault(
            (item.first_piece, item.first_segment.edge_index), set()
        ).add(group_id)
        edge_groups.setdefault(
            (item.second_piece, item.second_segment.edge_index), set()
        ).add(group_id)

    def group_priority(group_id: str) -> tuple:
        # 最差的一半决定复合组的质量；避免一半极好、另一半很差的组
        # 仅凭局部最优残差挤掉两半都一致的真实 T 节点。
        rows = atomic_groups[group_id]
        if rows[0].composite_id is not None:
            # 虚拟分割端点没有完整物理边的双端角色，不能让该局部角色
            # 的 valid/ambiguous 标志支配整个 T 节点的开放顺序。
            return (
                0.0,
                0.0,
                max(float(item.score) for item in rows),
                max(float(item.directed_endpoint_cost) for item in rows),
                -min(float(item.quality_score) for item in rows),
                max(float(item.outer_edge_risk) for item in rows),
                max(float(
                    item.remaining_dof + item.ambiguity_penalty
                ) for item in rows),
                -min(float(item.geometric_explanation_score) for item in rows),
                group_id,
            )
        return placement_priority(rows[0])

    opened: dict[str, PairwiseInterfacePlacement] = {}
    for group_ids in edge_groups.values():
        # 不同阶数是不同切割类型，不能让大量低阶局部匹配占满同一个
        # Top-K 后把合法的一对二/一对三永久挤掉。每种阶数独立获得
        # 渐进配额，随后仍在同一个全局布局评分中公平竞争。
        by_arity: dict[int, list[str]] = {}
        for group_id in group_ids:
            by_arity.setdefault(
                len(atomic_groups[group_id]), []
            ).append(group_id)
        for arity_group_ids in by_arity.values():
            for group_id in sorted(
                arity_group_ids, key=group_priority
            )[: stage.max_matches_per_edge]:
                for item in atomic_groups[group_id]:
                    opened[item.interface_id] = item
    return sorted(opened.values(), key=placement_priority)


def _composite_long_edge(
    placement: PairwiseInterfacePlacement,
) -> tuple[str, int] | None:
    """返回一对二候选中被分段的长边。"""

    for piece_id, segment in (
        (placement.first_piece, placement.first_segment),
        (placement.second_piece, placement.second_segment),
    ):
        if segment.t_start > 1e-12 or segment.t_end < 1.0 - 1e-12:
            return piece_id, segment.edge_index
    return None


def _composite_is_compatible(
    selected: list[PairwiseInterfacePlacement],
    candidate: PairwiseInterfacePlacement,
) -> bool:
    """禁止同一条长边混用两个不同的一对二分割假设。"""

    if candidate.composite_id is None:
        return True
    long_edge = _composite_long_edge(candidate)
    if long_edge is None:
        return False
    return all(
        item.composite_id is None
        or _composite_long_edge(item) != long_edge
        or item.composite_id == candidate.composite_id
        for item in selected
    )


def _composite_groups_complete(
    selected: list[PairwiseInterfacePlacement],
    available: list[PairwiseInterfacePlacement],
) -> bool:
    """完整布局中每个已使用的一对二假设必须包含它的全部半边。"""

    expected: Counter[str] = Counter(
        item.composite_id
        for item in available
        if item.composite_id is not None
    )
    actual: Counter[str] = Counter(
        item.composite_id
        for item in selected
        if item.composite_id is not None
    )
    return all(actual[group_id] == expected[group_id] for group_id in actual)


def _partial_layout_cost(current_union: Any, placed_area_mm2: float) -> float:
    """评估部分布局的紧凑程度，仅用于搜索队列的软排序。

    此函数不产生拒绝结论。轻微观测误差会使正确拼法在闭环前出现缝隙、
    小面积重叠或暂时超出外框；将这些量作为连续代价，可优先探索紧凑
    拓扑，同时保留所有有限的接口组合。
    """

    if current_union.is_empty or placed_area_mm2 <= 1e-12:
        return 0.0
    union_area = float(current_union.area)
    overlap_ratio = max(0.0, placed_area_mm2 - union_area) / placed_area_mm2
    hull_sprawl_ratio = max(
        0.0,
        float(current_union.convex_hull.area) - union_area,
    ) / placed_area_mm2
    rectangle = current_union.minimum_rotated_rectangle
    rectangle_area = float(rectangle.area)
    rectangularity_gap = max(0.0, rectangle_area - union_area) / max(
        placed_area_mm2, 1e-12
    )
    coordinates = np.asarray(rectangle.exterior.coords[:-1], dtype=float)
    side_lengths = [
        float(np.linalg.norm(
            coordinates[(index + 1) % len(coordinates)] - point
        ))
        for index, point in enumerate(coordinates)
    ]
    long_side = max(side_lengths, default=0.0)
    short_side = min(side_lengths, default=0.0)
    oversize_penalty = (
        max(0.0, long_side - 120.0) ** 2
        + max(0.0, short_side - 90.0) ** 2
    ) / 100.0
    return float(
        10.0 * overlap_ratio
        + 8.0 * rectangularity_gap
        + 4.0 * hull_sprawl_ratio
        + oversize_penalty
    )


def _enforce_depth_beam(
    queue: list[tuple[tuple, int, tuple]],
    *,
    depth: int,
    beam_width: int,
) -> int:
    """只保留同一装配深度的最佳束，返回本次移除的状态数。

    ``ProgressiveSearchStage.beam_width`` 过去只存在于配置中而没有参与
    搜索。随机拼图的分支因而会被局部长边匹配淹没。按深度束宽淘汰的是
    已由外框/紧凑度软评分排在末尾的状态，不使用未知毫米误差作拒绝条件。
    """

    if beam_width <= 0:
        return 0
    matching = [
        index for index, (_priority, _order, state) in enumerate(queue)
        if len(state[0]) == depth
    ]
    if len(matching) <= beam_width:
        return 0
    retained = set(sorted(
        matching,
        key=lambda index: (queue[index][0], queue[index][1]),
    )[:beam_width])
    removed = len(matching) - len(retained)
    queue[:] = [
        row for index, row in enumerate(queue)
        if index not in matching or index in retained
    ]
    heapq.heapify(queue)
    return removed


def _solve_random_legacy_unused(
    pieces: list[ProcessedPiece],
    noise_mm: float = 0.0,
    top_n: int = 100,
    appearance_edge_matches: list[dict] | None = None,
    allowed_adjacency: set[frozenset[str]] | None = None,
    diagnostics: dict[str, Any] | None = None,
    runtime_config: RuntimeConfig | None = None,
    stage_timer: Any | None = None,
    anchor_index: int = 0,
    deadline: float | None = None,
    on_candidate: Callable[[RandomCandidate], bool] | None = None,
    uncertainty_profile: ObservationUncertaintyProfile | None = None,
    progressive_stage: ProgressiveSearchStage | None = None,
) -> list[RandomCandidate]:
    """渐进枚举拼缝拓扑，并在完整装配后执行全局几何准入。

    搜索阶段只使用接口残差和部分布局紧凑度进行软排序，不以未知的
    观测毫米误差删除拓扑。完整边、一对多区间和带物理来源的部分区间
    依次随搜索阶段开放；只有全部碎片落位后才验证矩形外框、拼缝覆盖、
    节点闭合和面积关系。该函数是随机模式唯一正式搜索入口。
    """
    config = runtime_config or RuntimeConfig()
    if config.unlimited_search:
        deadline = None
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
    stage = progressive_stage or stages_from_config(config)[0]
    n = len(pieces)
    anchor = int(anchor_index)
    if not 0 <= anchor < n:
        raise ValueError("anchor_index 越界")
    anchor_geometry = CachedPoseGeometry.from_vertices(
        pieces[anchor].vertices.copy()
    )
    initial = {
        anchor: RandomPose(
            pieces[anchor].piece_id,
            0.0,
            np.zeros(2),
            pieces[anchor].vertices.copy(),
            geometry=anchor_geometry,
        )
    }
    interface_started = perf_counter()
    _interface_candidates, placements, interface_diagnostics = (
        build_shared_interface_candidates(
            pieces,
            noise_mm,
            include_generic_partial=stage.allow_generic_partial,
            diagnostics_level=config.diagnostics_level,
            uncertainty_profile=profile,
        )
    )
    if stage_timer is not None:
        stage_timer.add(
            "interface_generation", perf_counter() - interface_started
        )
    if allowed_adjacency is not None:
        placements = [
            item
            for item in placements
            if frozenset((item.first_piece, item.second_piece))
            in allowed_adjacency
        ]
    hard_pool_count = len(placements)
    hard_topology_placements = list(placements)
    ambiguity_metrics = interface_ambiguity_metrics(placements)
    ranking_started = perf_counter()
    placements = _placements_for_stage(placements, stage)
    if stage_timer is not None:
        stage_timer.add(
            "interface_ranking", perf_counter() - ranking_started
        )
    opened_interface_count = len(placements)
    # 闭环只能使用当前阶段原子开放的接口。若回到完整硬池补边，不但绕过
    # 渐进 Top-K 预算，还可能给已选拓扑偷偷混入一个未开放的 T 分割。
    # 当前组内的两半会一起开放，因此合法环和 T 节点仍可完整补全。
    closure_placements = list(placements)
    search_placements = [
        item for item in placements if item.determines_full_pose
    ]
    by_pair: dict[frozenset[str], list[PairwiseInterfacePlacement]] = {}
    for item in search_placements:
        by_pair.setdefault(
            frozenset((item.first_piece, item.second_piece)), []
        ).append(item)
    for pair_candidates in by_pair.values():
        pair_candidates.sort(key=placement_priority)

    candidate_pool = RandomCandidatePool(
        capacity=max(
            1,
            min(
                top_n,
                stage.candidate_limit,
                config.maximum_complete_candidates,
            ),
        ),
        max_variants_per_discrete_signature=(
            config.max_variants_per_discrete_signature
        ),
    )
    fully_placed_state_count = 0
    fully_placed_rejected_count = 0
    geometry_candidate_admitted_count = 0
    admission_rejections: Counter[str] = Counter()
    admission_rejection_samples: list[dict[str, Any]] = []
    pool_first_full_node_count: int | None = None
    visited_states: set[tuple] = set()
    node_count = 0
    generated_states = 0
    duplicate_pruned = 0
    rectangle_pruned = 0
    overlap_pruned = 0
    cycle_pruned = 0
    interface_conflict_pruned = 0
    beam_pruned = 0
    max_nodes = min(config.max_search_nodes, stage.node_limit)
    current_node_limit = max_nodes
    # Pool capacity controls memory and downstream validation only.  It must
    # never be used as a search termination condition.
    completion_limit = candidate_pool.capacity
    appearance = {
        (
            str(item["piece_a"]),
            int(item["edge_a"]),
            str(item["piece_b"]),
            int(item["edge_b"]),
        ): float(item.get("match_score", 0.0))
        for item in (appearance_edge_matches or [])
    }
    total_piece_area = float(sum(piece.area for piece in pieces))
    feasible_width_low, feasible_width_high = feasible_width_interval(
        total_piece_area
    )
    search_wall_started = perf_counter()
    wall_time_limited = False
    callback_stopped = False

    auditing_complete_hyperedge = False
    queue: list[tuple[tuple, int, tuple]] = []
    scheduled_counter = 0

    def recurse(
        poses: dict[int, RandomPose],
        pose_geometries: dict[int, CachedPoseGeometry],
        current_union: Any,
        occupied: dict[tuple[str, int], list[tuple[float, float]]],
        selected: list[PairwiseInterfacePlacement],
    ) -> None:
        nonlocal node_count
        nonlocal generated_states, duplicate_pruned, rectangle_pruned
        nonlocal overlap_pruned, cycle_pruned, interface_conflict_pruned
        nonlocal wall_time_limited, callback_stopped
        nonlocal fully_placed_state_count, fully_placed_rejected_count
        nonlocal geometry_candidate_admitted_count
        nonlocal pool_first_full_node_count
        if callback_stopped:
            return
        if (
            not auditing_complete_hyperedge
            and deadline is not None
            and perf_counter() >= deadline
        ):
            wall_time_limited = True
            return
        if (
            perf_counter() - search_wall_started
            > config.wall_time_guard_seconds
        ):
            wall_time_limited = True
            return
        node_count += 1
        if node_count > current_node_limit:
            return
        if len(poses) == n:
            fully_placed_state_count += 1
            augmented = _augment_closed_interfaces(
                poses, selected, closure_placements, profile
            )
            if not _composite_groups_complete(
                augmented, hard_topology_placements
            ):
                interface_conflict_pruned += 1
                fully_placed_rejected_count += 1
                return
            canonical_started = perf_counter()
            new_candidates = _canonical_candidates(
                poses, augmented, appearance
            )
            if stage_timer is not None:
                stage_timer.add(
                    "canonicalization",
                    perf_counter() - canonical_started,
                )
            accepted_from_state = 0
            for new_candidate in new_candidates:
                admission_started = perf_counter()
                admission = assess_random_assembly(
                    {
                        pose.piece_id: pose.target_vertices_local
                        for pose in new_candidate.poses
                    },
                    profile,
                    config,
                    expected_piece_count=n,
                    declared_interface_count=len(augmented),
                    stage_name=stage.name,
                )
                if stage_timer is not None:
                    stage_timer.add(
                        "fully_placed_admissibility",
                        perf_counter() - admission_started,
                    )
                new_candidate.admissibility = admission
                if not admission.accepted:
                    admission_rejections.update(admission.rejection_reasons)
                    if (
                        len(admission_rejection_samples)
                        < config.diagnostic_rejection_sample_limit
                    ):
                        admission_rejection_samples.append({
                            "rejection_reasons": admission.rejection_reasons,
                            # 保留少量完整落位，供离线诊断直接渲染“已拼完
                            # 但未准入”的反例；不参与搜索排序或执行。
                            "poses": [
                                {
                                    "piece_id": pose.piece_id,
                                    "vertices": np.asarray(
                                        pose.target_vertices_local,
                                        dtype=float,
                                    ).tolist(),
                                }
                                for pose in new_candidate.poses
                            ],
                            "width_mm": admission.target_rectangle_width_mm,
                            "height_mm": admission.target_rectangle_height_mm,
                            "fill_ratio": admission.rectangle_fill_ratio,
                            "rectangle_iou": (
                                admission.union_rectangle_iou_estimate
                            ),
                            "overlap_area_mm2": admission.overlap_area_mm2,
                            "uncovered_area_mm2": (
                                admission.uncovered_area_mm2
                            ),
                            "declared_interface_count": (
                                admission.declared_interface_count
                            ),
                            "discovered_interface_count": (
                                admission.discovered_interface_count
                            ),
                            "interface_explanation": (
                                admission.interface_explanation.as_json()
                            ),
                            "directed_endpoint_context_valid": (
                                admission.directed_endpoint_context_valid
                            ),
                            "opposite_side_valid": (
                                admission.opposite_side_valid
                            ),
                            "rectangle_frame_fit": (
                                admission.rectangle_frame_fit.as_json()
                                if admission.rectangle_frame_fit is not None
                                else None
                            ),
                            "outer_roles": {
                                piece_id: role.as_json()
                                for piece_id, role
                                in admission.outer_roles.items()
                            },
                            "rectangle_role_consistency": (
                                admission.rectangle_role_consistency.as_json()
                                if admission.rectangle_role_consistency
                                is not None else None
                            ),
                            "junction_closure": [
                                row.as_json()
                                for row in admission.junction_closure
                            ],
                        })
                    continue
                accepted_from_state += 1
                geometry_candidate_admitted_count += 1
                pool_started = perf_counter()
                changed = candidate_pool.consider(new_candidate)
                if (
                    candidate_pool.ever_full
                    and pool_first_full_node_count is None
                ):
                    pool_first_full_node_count = node_count
                if stage_timer is not None:
                    stage_timer.add(
                        "candidate_pool_management",
                        perf_counter() - pool_started,
                    )
                if changed and on_candidate is not None:
                    if not on_candidate(new_candidate):
                        callback_stopped = True
                        break
            if accepted_from_state == 0:
                fully_placed_rejected_count += 1
            return

        placed = sorted(poses)
        unplaced = [index for index in range(n) if index not in poses]
        for new_index in unplaced:
            new_piece = pieces[new_index]
            for parent_index in placed:
                parent_piece = pieces[parent_index]
                parent_pose = poses[parent_index]
                pair_candidates = by_pair.get(
                    frozenset((parent_piece.piece_id, new_piece.piece_id)), []
                )
                for raw in pair_candidates:
                    item = (
                        raw
                        if raw.first_piece == parent_piece.piece_id
                        else raw.reversed()
                    )
                    if not _composite_is_compatible(selected, item):
                        interface_conflict_pruned += 1
                        continue
                    parent_segment = item.first_segment
                    new_segment = item.second_segment
                    parent_key = (
                        parent_piece.piece_id,
                        parent_segment.edge_index,
                    )
                    new_key = (
                        new_piece.piece_id,
                        new_segment.edge_index,
                    )
                    if not interval_is_available(
                        occupied.get(parent_key, []),
                        (parent_segment.t_start, parent_segment.t_end),
                    ):
                        interface_conflict_pruned += 1
                        continue
                    if not interval_is_available(
                        occupied.get(new_key, []),
                        (new_segment.t_start, new_segment.t_end),
                    ):
                        interface_conflict_pruned += 1
                        continue
                    if (
                        config.precompute_relative_transforms
                        and item.relative_angle is not None
                        and item.relative_translation is not None
                    ):
                        angle, translation = compose_transform(
                            item.relative_angle,
                            item.relative_translation,
                            parent_pose.angle,
                            parent_pose.translation,
                        )
                    else:
                        parent_points = _segment_points(parent_pose, item)
                        target_segment = np.asarray(
                            [parent_points[1], parent_points[0]], dtype=float
                        )
                        source_segment = np.asarray(
                            [
                                new_segment.start_point,
                                new_segment.end_point,
                            ],
                            dtype=float,
                        )
                        angle, translation, _endpoint_rms = estimate_rigid(
                            source_segment, target_segment
                        )
                    local = transform_points(
                        new_piece.vertices, angle, translation
                    )
                    new_geometry = CachedPoseGeometry.from_vertices(local)
                    new_polygon = new_geometry.polygon
                    new_union = (
                        current_union.union(new_polygon)
                        if config.incremental_union
                        else unary_union([
                            Polygon(p.target_vertices_local)
                            for p in poses.values()
                        ] + [new_polygon])
                    )
                    # 不在部分布局阶段把“看起来重叠/超出矩形”的状态当作
                    # 不可逆错误。带有轻微顶点误差的正确拼法常在第三、四片
                    # 闭环前暂时呈现这两种现象；完整装配后由统一的全局几何
                    # 评分和准入审核决定，而不是由未知毫米误差半径决定。
                    pose_key = canonical_pose_state(
                        [
                            (
                                pose.piece_id,
                                pose.angle,
                                float(pose.translation[0]),
                                float(pose.translation[1]),
                            )
                            for pose in poses.values()
                        ]
                        + [
                            (
                                new_piece.piece_id,
                                angle,
                                float(translation[0]),
                                float(translation[1]),
                            )
                        ],
                    )
                    occupied_key = tuple(
                        sorted(
                            (
                                piece_id,
                                edge_index,
                                tuple(
                                    sorted(
                                        (
                                            round(start, 7),
                                            round(end, 7),
                                        )
                                        for start, end in intervals
                                    )
                                ),
                            )
                            for (piece_id, edge_index), intervals in occupied.items()
                        )
                    )
                    key = (
                        pose_key,
                        occupied_key,
                        (
                            parent_key,
                            round(parent_segment.t_start, 7),
                            round(parent_segment.t_end, 7),
                        ),
                        (
                            new_key,
                            round(new_segment.t_start, 7),
                            round(new_segment.t_end, 7),
                        ),
                    )
                    if key in visited_states:
                        duplicate_pruned += 1
                        continue
                    visited_states.add(key)
                    generated_states += 1
                    updated_poses = dict(poses)
                    updated_poses[new_index] = RandomPose(
                        new_piece.piece_id, angle, translation, local,
                        geometry=new_geometry,
                    )
                    updated_geometries = dict(pose_geometries)
                    updated_geometries[new_index] = new_geometry
                    updated_occupied = {
                        edge: list(intervals)
                        for edge, intervals in occupied.items()
                    }
                    updated_occupied.setdefault(parent_key, []).append(
                        (parent_segment.t_start, parent_segment.t_end)
                    )
                    updated_occupied.setdefault(new_key, []).append(
                        (new_segment.t_start, new_segment.t_end)
                    )
                    schedule(
                        updated_poses,
                        updated_geometries,
                        new_union,
                        updated_occupied,
                        selected + [item],
                    )

    def schedule(
        poses: dict[int, RandomPose],
        pose_geometries: dict[int, CachedPoseGeometry],
        current_union: Any,
        occupied: dict[tuple[str, int], list[tuple[float, float]]],
        selected: list[PairwiseInterfacePlacement],
        *,
        enforce_beam: bool = True,
    ) -> None:
        nonlocal scheduled_counter, beam_pruned
        if config.search_strategy == "legacy_deterministic":
            recurse(
                poses, pose_geometries, current_union, occupied, selected
            )
            return
        constraint_rank = sum(item.constraint_rank for item in selected)
        confirmed_length = sum(
            min(
                item.first_segment.length_mm,
                item.second_segment.length_mm,
            )
            for item in selected
        )
        explanation = sum(
            item.geometric_explanation_score for item in selected
        )
        ambiguity = sum(
            item.ambiguity_penalty + item.outer_edge_risk
            for item in selected
        )
        residual = sum(item.score for item in selected)
        placed_area = sum(pieces[index].area for index in poses)
        soft_layout_cost = _partial_layout_cost(current_union, placed_area)
        stable_interfaces = tuple(
            item.interface_id for item in selected
        )
        priority = (
            -len(poses),
            -constraint_rank,
            round(soft_layout_cost, 9),
            round(ambiguity, 9),
            round(residual, 9),
            -round(confirmed_length, 9),
            -round(explanation, 9),
            stable_interfaces,
        )
        scheduled_counter += 1
        heapq.heappush(
            queue,
            (
                priority,
                scheduled_counter,
                (
                    poses, pose_geometries, current_union,
                    occupied, selected,
                ),
            ),
        )
        if enforce_beam:
            beam_pruned += _enforce_depth_beam(
                queue,
                depth=len(poses),
                beam_width=stage.beam_width,
            )

    search_started = perf_counter()
    schedule(
        initial, {anchor: anchor_geometry},
        anchor_geometry.polygon, {}, [],
    )

    # 能覆盖全部碎片的一对多接口本身就是完整拓扑超边。若仍把它拆成
    # n-1 次普通树扩展，正确组会在每一层重复与无关链式假设竞争，并在
    # 四片时轻易耗尽束宽。这里整组生成一个完整落位种子，但仍进入同一
    # 全局准入、优化、间隙与执行安全链路，不构造任何特殊成功捷径。
    piece_index_by_id = {
        piece.piece_id: index for index, piece in enumerate(pieces)
    }
    composite_rows: dict[str, list[PairwiseInterfacePlacement]] = {}
    for item in hard_topology_placements:
        if item.composite_id is not None:
            composite_rows.setdefault(item.composite_id, []).append(item)
    hyperedge_seeds: list[tuple[float, float, tuple]] = []
    for rows in composite_rows.values():
        involved = {
            piece_id
            for item in rows
            for piece_id in (item.first_piece, item.second_piece)
        }
        if len(rows) != n - 1 or len(involved) != n:
            continue
        long_edge = _composite_long_edge(rows[0])
        if long_edge is None:
            continue
        long_piece_id = long_edge[0]
        long_index = piece_index_by_id[long_piece_id]
        seeded_poses: dict[int, RandomPose] = {
            long_index: RandomPose(
                long_piece_id,
                0.0,
                np.zeros(2),
                pieces[long_index].vertices.copy(),
                geometry=CachedPoseGeometry.from_vertices(
                    pieces[long_index].vertices.copy()
                ),
            )
        }
        seeded_geometries = {
            long_index: seeded_poses[long_index].geometry
        }
        seeded_occupied: dict[
            tuple[str, int], list[tuple[float, float]]
        ] = {}
        valid_seed = True
        for raw in rows:
            item = raw if raw.first_piece == long_piece_id else raw.reversed()
            short_index = piece_index_by_id[item.second_piece]
            if (
                item.relative_angle is None
                or item.relative_translation is None
                or short_index in seeded_poses
            ):
                valid_seed = False
                break
            angle = float(item.relative_angle)
            translation = np.asarray(item.relative_translation, dtype=float)
            local = transform_points(
                pieces[short_index].vertices, angle, translation
            )
            geometry = CachedPoseGeometry.from_vertices(local)
            seeded_poses[short_index] = RandomPose(
                item.second_piece,
                angle,
                translation,
                local,
                geometry=geometry,
            )
            seeded_geometries[short_index] = geometry
            for piece_id, segment in (
                (item.first_piece, item.first_segment),
                (item.second_piece, item.second_segment),
            ):
                seeded_occupied.setdefault(
                    (piece_id, segment.edge_index), []
                ).append((segment.t_start, segment.t_end))
        if valid_seed and len(seeded_poses) == n:
            seeded_union = unary_union([
                geometry.polygon
                for geometry in seeded_geometries.values()
            ])
            hyperedge_seeds.append((
                _partial_layout_cost(seeded_union, total_piece_area),
                sum(float(item.score) for item in rows),
                (
                    seeded_poses,
                    seeded_geometries,
                    seeded_union,
                    seeded_occupied,
                    list(rows),
                ),
            ))
    # 完整超边已经是可直接审核的完整布局，无需再伪装成普通树节点。
    # 先按旋转不变的全局紧凑度审核少量最佳组；若都不安全，再继续
    # 常规逐片搜索。这样完整组不会被大量链式完整状态拖到截止时间后。
    hyperedge_seeds.sort(key=lambda row: (row[0], row[1]))
    hyperedge_audit_limit = min(2, stage.beam_width)
    auditing_complete_hyperedge = True
    for _layout_cost, _interface_cost, state in hyperedge_seeds[
        :hyperedge_audit_limit
    ]:
        recurse(*state)
        if callback_stopped or wall_time_limited:
            break
    auditing_complete_hyperedge = False
    while (
        queue
        and not callback_stopped
        and (deadline is None or perf_counter() < deadline)
    ):
        _priority, _order, state = heapq.heappop(queue)
        recurse(*state)
    if stage_timer is not None:
        stage_timer.add("search_expansion", perf_counter() - search_started)
    if len(candidate_pool) == 0 and not stage.allow_generic_partial:
        # 普通整边/一对二搜索无解时，才枚举任意长边端部的部分区间，
        # 用于多内部节点、长对长部分覆盖；避免拖慢常规案例。
        fallback_build_started = perf_counter()
        (
            _interface_candidates,
            placements,
            fallback_diagnostics,
        ) = build_shared_interface_candidates(
            pieces,
            noise_mm,
            include_generic_partial=True,
            diagnostics_level=(
                "full"
                if config.diagnostics_level != "off"
                else "off"
            ),
            uncertainty_profile=profile,
        )
        if stage_timer is not None:
            fallback_build_elapsed = perf_counter() - fallback_build_started
            stage_timer.add(
                "atomic_interval_build", fallback_build_elapsed
            )
        if allowed_adjacency is not None:
            placements = [
                item
                for item in placements
                if frozenset((item.first_piece, item.second_piece))
                in allowed_adjacency
            ]
        closure_placements = list(placements)
        placements = _placements_for_stage(placements, stage)
        by_pair = {}
        search_placements = [
            item for item in placements if item.determines_full_pose
        ]
        for item in search_placements:
            by_pair.setdefault(
                frozenset((item.first_piece, item.second_piece)), []
            ).append(item)
        for pair_candidates in by_pair.values():
            pair_candidates.sort(key=placement_priority)
        interface_diagnostics = fallback_diagnostics
        interface_diagnostics["generic_partial_fallback_used"] = True
        visited_states.clear()
        # Compatibility fallback; normal progressive stages already open
        # sourced partial interfaces in their hard pool.
        current_node_limit = node_count + config.fallback_search_nodes
        fallback_started = perf_counter()
        queue.clear()
        schedule(
            initial, {anchor: anchor_geometry},
            anchor_geometry.polygon, {}, [],
        )
        while (
            queue
            and not callback_stopped
            and (deadline is None or perf_counter() < deadline)
        ):
            _priority, _order, state = heapq.heappop(queue)
            recurse(*state)
        if stage_timer is not None:
            stage_timer.add(
                "generic_partial_fallback", perf_counter() - fallback_started
            )
    else:
        interface_diagnostics["generic_partial_fallback_used"] = False
    if deadline is not None and perf_counter() >= deadline:
        wall_time_limited = True
    dedup_started = perf_counter()
    completed = candidate_pool.items()
    completed.sort(key=lambda candidate: (
        candidate.admissibility.rank_key()
        if candidate.admissibility is not None else (1.0,),
        candidate.total,
    ))
    unique: list[RandomCandidate] = []
    signatures: set[tuple] = set()
    for candidate in completed:
        signature = (
            round(candidate.width, 4),
            round(candidate.height, 4),
            tuple(
                sorted(
                    (
                        pose.piece_id,
                        round(
                            float(
                                np.mean(
                                    pose.target_vertices_local[:, 0]
                                )
                            ),
                            3,
                        ),
                        round(
                            float(
                                np.mean(
                                    pose.target_vertices_local[:, 1]
                                )
                            ),
                            3,
                        ),
                        round(pose.angle, 3),
                    )
                    for pose in candidate.poses
                )
            ),
        )
        if signature not in signatures:
            signatures.add(signature)
            if len(unique) < top_n:
                unique.append(candidate)
    for candidate in unique:
        candidate.legal_complete_candidate_count = len(completed)
    if stage_timer is not None:
        stage_timer.add(
            "candidate_deduplication", perf_counter() - dedup_started
        )

    if diagnostics is not None:
        diagnostics.update(interface_diagnostics)
        diagnostics.update(
            {
                "search_nodes": node_count,
                "max_search_nodes": max_nodes,
                # Compatibility alias: now counts admitted geometry candidates,
                # never all fully placed states.
                "legal_complete_candidate_count": len(completed),
                "completion_limit": completion_limit,
                "candidate_count": len(unique),
                "search_state_count": node_count,
                "fully_placed_state_count": fully_placed_state_count,
                "fully_placed_rejected_count": fully_placed_rejected_count,
                "geometry_candidate_count": (
                    geometry_candidate_admitted_count
                ),
                "geometry_candidate_pool_size": len(candidate_pool),
                "geometry_candidate_rejection_reasons": dict(
                    sorted(admission_rejections.items())
                ),
                "geometry_candidate_rejection_samples": (
                    admission_rejection_samples
                ),
                "candidate_pool_replacements": candidate_pool.replacements,
                "candidate_pool_duplicates": candidate_pool.duplicates,
                "candidate_pool_discarded": candidate_pool.discarded,
                "candidate_pool_ever_full": candidate_pool.ever_full,
                "search_continued_after_pool_full": (
                    pool_first_full_node_count is not None
                    and node_count > pool_first_full_node_count
                ),
                "progressive_stage": stage.as_json(),
                "hard_interface_pool_count": hard_pool_count,
                "opened_interface_count": opened_interface_count,
                "search_ambiguity": ambiguity_metrics.as_json(),
                "legacy_noise_label_present": noise_mm is not None,
                "legacy_noise_label_used_by_solver": False,
                "search_exhausted": (
                    node_count > current_node_limit
                    or node_count > max_nodes
                    or wall_time_limited
                ),
                "generated_states": generated_states,
                "expanded_states": node_count,
                "duplicate_pruned": duplicate_pruned,
                "rectangle_pruned": rectangle_pruned,
                "overlap_pruned": overlap_pruned,
                "cycle_pruned": cycle_pruned,
                "interface_conflict_pruned": interface_conflict_pruned,
                "beam_pruned": beam_pruned,
                "beam_width": stage.beam_width,
                "budget_exhausted": (
                    node_count > current_node_limit
                    or node_count > max_nodes
                    or wall_time_limited
                ),
                "search_budget_limited": (
                    node_count > current_node_limit
                    or node_count > max_nodes
                    or wall_time_limited
                ),
                "wall_time_limited": wall_time_limited,
                "callback_stopped": callback_stopped,
                "search_level_reached": {
                    "fast": 0,
                    "expanded": 1,
                    "recovery": 2,
                }.get(stage.name, 0),
                "search_strategy": config.search_strategy,
                "random_candidate_policy": (
                    "soft_topology_then_global_geometry"
                ),
                "uses_default_hard_error_radius_for_early_pruning": False,
                "feasible_width_interval_mm": [
                    feasible_width_low,
                    feasible_width_high,
                ],
            }
        )
        if not unique:
            diagnostics["failure_type"] = (
                "disconnected_positive_length_adjacency"
                if not interface_diagnostics["interface_graph_connected"]
                else "no_geometry_candidate"
            )
    return unique


def _declared_interfaces_from_geometry(
    poses: list[RandomPose],
    pieces: list[ProcessedPiece],
    tolerance_mm: float,
) -> list[PairwiseInterfacePlacement]:
    """把最终几何中的边—边区间还原为现有候选接口对象。"""

    placed = {
        pose.piece_id: np.asarray(pose.target_vertices_local, dtype=float)
        for pose in poses
    }
    by_id = {piece.piece_id: piece for piece in pieces}
    discovered = discover_interfaces_in_placed_state(
        placed, tolerance_mm=tolerance_mm, length_epsilon_mm=1.0
    )
    result: list[PairwiseInterfacePlacement] = []
    for index, row in enumerate(discovered["interfaces"]):
        first_id = str(row["piece_a"])
        second_id = str(row["piece_b"])
        first_interval = tuple(sorted(
            float(np.clip(value, 0.0, 1.0)) for value in row["interval_a"]
        ))
        second_interval = tuple(sorted(
            float(np.clip(value, 0.0, 1.0)) for value in row["interval_b"]
        ))
        if (
            first_interval[1] - first_interval[0] <= 1e-8
            or second_interval[1] - second_interval[0] <= 1e-8
        ):
            continue
        first = make_atomic_segment(
            first_id,
            by_id[first_id].vertices,
            int(row["edge_a"]),
            *first_interval,
        )
        second = make_atomic_segment(
            second_id,
            by_id[second_id].vertices,
            int(row["edge_b"]),
            *second_interval,
        )
        residual = max(
            float(row["support_line_distance_mm"]),
            float(row["parallel_error"]),
        )
        result.append(PairwiseInterfacePlacement(
            interface_id=f"partition_contact_{index:03d}",
            first_piece=first_id,
            first_segment=first,
            second_piece=second_id,
            second_segment=second,
            interface_type="reverse_partition_contact",
            composite_id=None,
            score=residual,
            geometric_explanation_score=1.0,
            ambiguity_penalty=0.0,
            transform_residual=residual,
            quality_score=float(1.0 / (1.0 + residual)),
        ))
    return result


def _attach_partition_evidence(
    candidate: RandomCandidate,
    layout: Any,
    admission: RandomSafetyAssessment,
) -> None:
    """把解析树和安全门重建的原始边图写回同一候选。"""

    candidate.topology_template = str(layout.topology_template)
    candidate.partition_evidence = [{
        "evidence_type": "topology_template",
        "template": candidate.topology_template,
    }]
    candidate.partition_evidence.extend(
        item.as_json() for item in layout.evidence
    )
    candidate.partition_evidence.append({
        "evidence_type": "final_original_edge_correspondence_graph",
        "interfaces": [
            interface.as_json()
            for interface in admission.topology.interfaces
        ],
        "junctions": [
            junction.as_json()
            for junction in admission.topology.junctions
        ],
        "adjacency_pairs": [
            list(pair) for pair in admission.topology.adjacency_pairs
        ],
    })


def solve_random(
    pieces: list[ProcessedPiece],
    noise_mm: float = 0.0,
    top_n: int = 100,
    appearance_edge_matches: list[dict] | None = None,
    allowed_adjacency: set[frozenset[str]] | None = None,
    diagnostics: dict[str, Any] | None = None,
    runtime_config: RuntimeConfig | None = None,
    stage_timer: Any | None = None,
    anchor_index: int = 0,
    deadline: float | None = None,
    on_candidate: Callable[[RandomCandidate], bool] | None = None,
    uncertainty_profile: ObservationUncertaintyProfile | None = None,
    progressive_stage: ProgressiveSearchStage | None = None,
    rigid_backend=None,
) -> list[RandomCandidate]:
    """由有向边链对应关系反向解析矩形分割并发布已评估候选。

    ``noise_mm``、锚点和渐进阶段参数为兼容既有封装而保留；它们不再
    切换旧的自由位姿搜索或通用部分边回退。
    """

    del noise_mm, anchor_index
    config = runtime_config or RuntimeConfig()
    if config.unlimited_search:
        deadline = None
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
    stage = progressive_stage or stages_from_config(config)[0]
    sigma = float(profile.typical_sigma_mm or 0.0)
    tolerance_mm = float(np.clip(
        max(1.8, profile.hard_error_radius_mm + sigma), 1.8, 4.0
    ))
    result_limit = max(
        240,
        top_n * 6,
        stage.candidate_limit * 4,
        config.maximum_complete_candidates * 4,
    )
    generation_deadline = deadline
    if deadline is not None:
        generation_deadline = deadline - 2.0
    partition_diagnostics = {}

    def iter_layouts():
        # Validate measured edge-chain proposals before spending the budget on
        # supplementary corner assignments. Four rectangular
        # pieces have many equivalent outer-corner assignments; eagerly building
        # that supplementary pool can delay even an exact grid by minutes.
        partition_diagnostics['outer_corner_hypothesis_count'] = 0
        partition_diagnostics['outer_corner_hypotheses_deferred'] = True
        phases = [True, False] if len(pieces) == 4 else [False]
        expanded = generated = 0
        elapsed = 0.0
        for strict in phases:
            if deadline is not None and perf_counter() >= generation_deadline:
                break
            layouts, details = enumerate_partition_candidates(
                pieces, tolerance_mm=tolerance_mm, deadline=generation_deadline,
                maximum_results=result_limit,
                maximum_merge_options=max(64, min(96, stage.beam_width * 3)),
                maximum_states_per_depth=(32 if len(pieces) == 4 else 160),
                rigid_backend=rigid_backend,
                **({'strict_only': strict, '_run_strict_probe': strict} if len(pieces) == 4 else {}),
            )
            expanded += details['partition_search_expanded_states']
            generated += details['partition_search_generated_states']
            elapsed += details['partition_search_elapsed_seconds']
            partition_diagnostics.update(details)
            partition_diagnostics.update(partition_search_expanded_states=expanded,
                partition_search_generated_states=generated,
                partition_search_elapsed_seconds=elapsed)
            yield from layouts
        # Only build this supplementary pool if validated whole/partial chains
        # have not satisfied the caller. A rejected strict proposal may still
        # have a cheap T-junction solution, so do not insert corners between the
        # two ordinary phases.
        if (len(pieces) == 4 and config.rectangle_corner_hypotheses
                and config.random_competition_rules
                and (generation_deadline is None or perf_counter() < generation_deadline)):
            from .corner_hypotheses import rectangle_corner_candidates
            corners = rectangle_corner_candidates(pieces, deadline=generation_deadline)
            partition_diagnostics['outer_corner_hypothesis_count'] = len(corners)
            yield from corners

    partition_diagnostics['partition_search_expanded_states'] = 0
    appearance = {
        (
            str(item["piece_a"]),
            int(item["edge_a"]),
            str(item["piece_b"]),
            int(item["edge_b"]),
        ): float(item.get("match_score", 0.0))
        for item in (appearance_edge_matches or [])
    }
    pool = RandomCandidatePool(
        capacity=max(1, min(
            top_n, stage.candidate_limit, config.maximum_complete_candidates
        )),
        max_variants_per_discrete_signature=(
            config.max_variants_per_discrete_signature
        ),
    )
    rejection_reasons: Counter[str] = Counter()
    rejection_samples: list[dict[str, Any]] = []
    fully_placed = 0
    topology_fit_count = 0
    callback_stopped = False
    for layout_index, layout in enumerate(iter_layouts()):
        if (
            deadline is not None
            and perf_counter() >= deadline
            and layout_index >= 8
        ):
            break
        indexed_poses = {
            index: RandomPose(
                piece.piece_id,
                float(layout.poses[piece.piece_id][0]),
                np.asarray(layout.poses[piece.piece_id][1], dtype=float),
                np.asarray(layout.poses[piece.piece_id][2], dtype=float),
            )
            for index, piece in enumerate(pieces)
        }
        canonical_options = _canonical_candidates(
            indexed_poses, [], appearance
        )
        # Float32 accelerator round-off must not choose the opposite 180-degree
        # representative of the same rectangle. Tie only below scoring resolution.
        canonical_options.sort(key=lambda item: (
            round(item.total, 3),
            tuple(tuple(np.round(pose.target_vertices_local, 2).ravel())
                  for pose in sorted(item.poses, key=lambda p: p.piece_id)),
        ))
        # 四个外接矩形方向中 180° 旋转不产生新的拓扑解释；正式生成只
        # 联合拟合赛题宽高方向最合理的一种，避免同一错误布局耗尽预算。
        for rough_candidate in canonical_options[:1]:
            fully_placed += 1
            raw_interfaces = _declared_interfaces_from_geometry(
                rough_candidate.poses, pieces, tolerance_mm
            )
            raw_candidate = _score(
                rough_candidate.poses, raw_interfaces, appearance
            )
            if raw_candidate is not None:
                raw_admission = assess_random_assembly(
                    {
                        pose.piece_id: pose.target_vertices_local
                        for pose in raw_candidate.poses
                    },
                    profile,
                    config,
                    expected_piece_count=len(pieces),
                    declared_interface_count=len(raw_interfaces),
                    stage_name=stage.name,
                )
                raw_candidate.admissibility = raw_admission
                _attach_partition_evidence(
                    raw_candidate, layout, raw_admission
                )
                raw_candidate.total += float(layout.cost)
                if raw_admission.accepted:
                    changed = pool.consider(raw_candidate)
                    if (
                        changed
                        and on_candidate is not None
                        and not on_candidate(raw_candidate)
                    ):
                        callback_stopped = True
                    continue
            interfaces = _declared_interfaces_from_geometry(
                rough_candidate.poses, pieces, max(tolerance_mm, 5.0)
            )
            if allowed_adjacency is not None and any(
                frozenset((item.first_piece, item.second_piece))
                not in allowed_adjacency
                for item in interfaces
            ):
                continue
            fitted_poses = rough_candidate.poses
            # 顺序合并只负责确定离散拓扑和初值。一个 T/Y 结点会同时约束
            # 三片，必须联合微调全部刚体位姿，不能让最后合并的一条边独占
            # 位姿决定权。优化仍只使用刚体、已发现边区间和矩形外框。
            if (
                interfaces
                and topology_fit_count < 120
                and (deadline is None or perf_counter() < deadline)
            ):
                records = [
                    {
                        "piece_id": pose.piece_id,
                        "observed": next(
                            piece.vertices
                            for piece in pieces
                            if piece.piece_id == pose.piece_id
                        ),
                        "target": pose.target_vertices_local,
                        "angle": pose.angle,
                        "translation": pose.translation,
                    }
                    for pose in rough_candidate.poses
                ]
                optimized, _width, _height, _details = optimize_generic_candidate(
                    records,
                    rough_candidate.width,
                    rough_candidate.height,
                    interfaces=interfaces,
                    dimension_bounds=((90.0, 50.0), (120.0, 90.0))
                        if config.random_competition_rules else None,
                    max_nfev_override=120,
                )
                topology_fit_count += 1
                fitted_poses = [
                    RandomPose(
                        str(record["piece_id"]),
                        float(record["angle"]),
                        np.asarray(record["translation"], dtype=float),
                        np.asarray(record["target"], dtype=float),
                    )
                    for record in optimized
                ]
                interfaces = _declared_interfaces_from_geometry(
                    fitted_poses, pieces, tolerance_mm
                )
            candidate = _score(fitted_poses, interfaces, appearance)
            if candidate is None:
                continue
            candidate.total += float(layout.cost)
            admission = assess_random_assembly(
                {
                    pose.piece_id: pose.target_vertices_local
                    for pose in candidate.poses
                },
                profile,
                config,
                expected_piece_count=len(pieces),
                declared_interface_count=len(interfaces),
                stage_name=stage.name,
            )
            candidate.admissibility = admission
            _attach_partition_evidence(candidate, layout, admission)
            if not admission.accepted:
                rejection_reasons.update(admission.rejection_reasons)
                if len(rejection_samples) < config.diagnostic_rejection_sample_limit:
                    rejection_samples.append({
                        "rejection_reasons": admission.rejection_reasons,
                        "rectangle_iou": admission.union_rectangle_iou_estimate,
                        "overlap_area_mm2": admission.overlap_area_mm2,
                        "uncovered_area_mm2": admission.uncovered_area_mm2,
                        "partition_evidence": candidate.partition_evidence,
                        "poses": [
                            {
                                "piece_id": pose.piece_id,
                                "vertices": pose.target_vertices_local.tolist(),
                            }
                            for pose in candidate.poses
                        ],
                    })
            changed = pool.consider(candidate)
            if changed and on_candidate is not None and not on_candidate(candidate):
                callback_stopped = True
                break
        if callback_stopped:
            break
    candidates = pool.items()
    admitted_count = sum(
        bool(
            candidate.admissibility is not None
            and candidate.admissibility.accepted
        )
        for candidate in candidates
    )
    for candidate in candidates:
        candidate.legal_complete_candidate_count = admitted_count
    if diagnostics is not None:
        diagnostics.update(partition_diagnostics)
        diagnostics.update({
            "search_nodes": partition_diagnostics[
                "partition_search_expanded_states"
            ],
            "candidate_count": len(candidates),
            "legal_complete_candidate_count": admitted_count,
            "assessed_complete_candidate_count": len(candidates),
            "best_candidate_accepted": bool(
                candidates
                and candidates[0].admissibility is not None
                and candidates[0].admissibility.accepted
            ),
            "fully_placed_state_count": fully_placed,
            "fully_placed_rejected_count": sum(rejection_reasons.values()),
            "geometry_candidate_count": len(candidates),
            "partition_topology_joint_fit_count": topology_fit_count,
            "geometry_candidate_rejection_reasons": dict(rejection_reasons),
            "geometry_candidate_rejection_samples": rejection_samples,
            "generic_partial_fallback_used": False,
            "old_candidate_generator_used": False,
            "random_candidate_policy": (
                "directed_edges_vertices_angles_reverse_partition"
            ),
            "search_strategy": "reverse_partition_topology",
            "budget_exhausted": bool(
                deadline is not None and perf_counter() >= deadline
            ),
            "wall_time_limited": bool(
                deadline is not None and perf_counter() >= deadline
            ),
            "callback_stopped": callback_stopped,
            "progressive_stage": stage.as_json(),
        })
        if not candidates:
            diagnostics["failure_type"] = "no_reverse_partition_candidate"
    return candidates


def stream_random_candidates(
    pieces: list[ProcessedPiece],
    noise_mm: float,
    top_n: int,
    appearance_edge_matches: list[dict] | None,
    *,
    anchor_index: int,
    deadline: float,
    runtime_config: RuntimeConfig,
    on_candidate: Callable[[RandomCandidate], bool],
    diagnostics: dict[str, Any] | None = None,
    stage_timer: Any | None = None,
    uncertainty_profile: ObservationUncertaintyProfile | None = None,
    progressive_stage: ProgressiveSearchStage | None = None,
    rigid_backend=None,
) -> SearchRunStats:
    """Run deterministic search while publishing admitted candidates eagerly."""
    started = perf_counter()
    local_diagnostics: dict[str, Any] = {}
    candidates = solve_random(
        pieces,
        noise_mm,
        top_n,
        appearance_edge_matches,
        diagnostics=local_diagnostics,
        runtime_config=runtime_config,
        stage_timer=stage_timer,
        anchor_index=anchor_index,
        deadline=deadline,
        on_candidate=on_candidate,
        uncertainty_profile=uncertainty_profile,
        progressive_stage=progressive_stage,
        rigid_backend=rigid_backend,
    )
    if diagnostics is not None:
        diagnostics.update(local_diagnostics)
    return SearchRunStats(
        anchor_index=int(anchor_index),
        candidate_count=len(candidates),
        search_nodes=int(local_diagnostics.get("search_nodes", 0)),
        budget_exhausted=bool(
            local_diagnostics.get("budget_exhausted", False)
        ),
        wall_time_limited=bool(
            local_diagnostics.get("wall_time_limited", False)
        ),
        callback_stopped=bool(
            local_diagnostics.get("callback_stopped", False)
        ),
        elapsed_seconds=float(perf_counter() - started),
    )
