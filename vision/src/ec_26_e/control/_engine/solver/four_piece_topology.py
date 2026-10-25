"""四片随机拼图的通用边证据拓扑解析器。

该模块从边长、反向边链、端点、转角和多路结点证据生成二叉区域合并
树，并按成员分区保持搜索多样性；不按碎片边数或已知答案模板分类。
"""

from __future__ import annotations

from itertools import combinations, permutations, product
from math import atan2, degrees
from time import perf_counter
from typing import Any, Callable

import numpy as np
from ec_26_e.control.acceleration import RigidFitRequest, fit_search_batch
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..geometry import pairwise_overlap
from ..transforms import transform_points
from .partition_search import (
    EdgeChainEvidence,
    PartitionComponent,
    PartitionSearchResult,
    _enumerate_recursive_partition_candidates,
    _merge_options,
    _rectangle_rank,
    _state_rank,
)
from .preprocess import ProcessedPiece


def _leaf(piece: ProcessedPiece) -> PartitionComponent:
    """把一片观测多边形转换为零代价叶组件。"""

    return PartitionComponent(
        {
            piece.piece_id: (
                0.0,
                np.zeros(2, dtype=float),
                piece.vertices.copy(),
            )
        },
        Polygon(piece.vertices),
        (),
        0.0,
    )


def _deadline_reached(deadline: float | None) -> bool:
    """返回当前模板是否已经用完候选生成预算。"""

    return deadline is not None and perf_counter() >= deadline


def _rotation_transform(
    source_center: np.ndarray,
    source_ray_end: np.ndarray,
    target_center: np.ndarray,
    target_ray_end: np.ndarray,
) -> tuple[float, np.ndarray]:
    """令中心严格重合，并把一条径向边旋转到目标径向边。"""

    source = source_ray_end - source_center
    target = target_ray_end - target_center
    angle = degrees(atan2(target[1], target[0]) - atan2(source[1], source[0]))
    rotated_center = transform_points(
        np.asarray([source_center], dtype=float), angle
    )[0]
    return float(angle), np.asarray(target_center - rotated_center, dtype=float)


def _fan_incident_vertices(
    piece: ProcessedPiece,
    center_index: int,
    flipped: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """返回中心、循环左/右径向端点及其长度。"""

    count = len(piece.vertices)
    previous = piece.vertices[(center_index - 1) % count]
    following = piece.vertices[(center_index + 1) % count]
    center = piece.vertices[center_index]
    left, right = (
        (following, previous) if flipped else (previous, following)
    )
    return (
        center,
        left,
        right,
        float(np.linalg.norm(left - center)),
        float(np.linalg.norm(right - center)),
    )


def _fan_hypotheses(
    pieces: list[ProcessedPiece],
    tolerance_mm: float,
    maximum_hypotheses: int = 96,
) -> list[tuple[float, tuple[int, ...], tuple[int, ...], tuple[bool, ...]]]:
    """按中心补角和循环径向边长度生成有限四向中心假设。"""

    hypotheses: list[
        tuple[float, tuple[int, ...], tuple[int, ...], tuple[bool, ...]]
    ] = []
    # Endpoint geometry depends only on the piece and corner, not on the 6*5^4*8
    # possible arrangements. Avoid thousands of repeated NumPy norm allocations.
    incident_cache = {
        (index, center, flipped): _fan_incident_vertices(piece, center, flipped)
        for index, piece in enumerate(pieces)
        for center in range(len(piece.vertices))
        for flipped in (False, True)
    }
    # 固定第一片消除循环旋转；反向循环仍保留，用于吸收轮廓方向差异。
    # Corner products and radial compatibility are vectorized; the evidence and
    # ranking are unchanged. Only 48 order/handedness combinations run in Python.
    for tail in permutations(range(1, 4)):
        order = (0, *tail)
        centers = np.asarray(list(product(*(range(len(pieces[i].vertices)) for i in order))), dtype=int)
        residuals = np.abs(sum(pieces[i].internal_angles[centers[:, j]]
                              for j, i in enumerate(order)) - 360.0)
        keep = residuals <= 55.0
        centers, residuals = centers[keep], residuals[keep]
        if not len(centers):
            continue
        for tail_flips in product((False, True), repeat=3):
            flips = (False, *tail_flips)
            left = np.column_stack([
                np.asarray([incident_cache[(i, c, flipped)][3] for c in range(len(pieces[i].vertices))])[centers[:, j]]
                for j, (i, flipped) in enumerate(zip(order, flips))])
            right = np.column_stack([
                np.asarray([incident_cache[(i, c, flipped)][4] for c in range(len(pieces[i].vertices))])[centers[:, j]]
                for j, (i, flipped) in enumerate(zip(order, flips))])
            next_left = np.roll(left, -1, axis=1)
            difference = np.abs(right - next_left)
            limit = np.maximum(2.2 * tolerance_mm, 0.14 * np.maximum(right, next_left))
            valid = np.all(difference <= limit, axis=1)
            costs = 0.04 * residuals + np.sum(difference / np.maximum(limit, 1e-9), axis=1)
            for row in np.flatnonzero(valid):
                hypotheses.append((float(costs[row]), tuple(order),
                    tuple(int(v) for v in centers[row]), tuple(flips)))
    hypotheses.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    return hypotheses[:maximum_hypotheses]


def _fan_candidates(
    pieces: list[ProcessedPiece],
    tolerance_mm: float,
    deadline: float | None,
    maximum_results: int,
) -> tuple[list[PartitionSearchResult], int]:
    """一次性闭合四片中心、四条径向接口和矩形外框。"""

    results: list[PartitionSearchResult] = []
    evaluated = 0
    for prior_cost, order, centers, flips in _fan_hypotheses(
        pieces, tolerance_mm
    ):
        if _deadline_reached(deadline):
            break
        evaluated += 1
        ordered = [pieces[index] for index in order]
        incident = [
            _fan_incident_vertices(piece, center, flipped)
            for piece, center, flipped in zip(ordered, centers, flips)
        ]
        anchor_center = incident[0][0]
        poses: dict[str, tuple[float, np.ndarray, np.ndarray]] = {
            ordered[0].piece_id: (
                0.0,
                np.zeros(2, dtype=float),
                ordered[0].vertices.copy(),
            )
        }
        transformed_incident: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = [
            (incident[0][0], incident[0][1], incident[0][2])
        ]
        evidence: list[EdgeChainEvidence] = []
        cost = float(prior_cost)
        for index in range(1, 4):
            previous_right = transformed_incident[index - 1][2]
            center, left, right, left_length, _right_length = incident[index]
            angle, translation = _rotation_transform(
                center, left, anchor_center, previous_right
            )
            vertices = transform_points(
                ordered[index].vertices, angle, translation
            )
            transformed = transform_points(
                np.asarray((center, left, right)), angle, translation
            )
            poses[ordered[index].piece_id] = (
                angle,
                translation,
                vertices,
            )
            transformed_incident.append((
                transformed[0], transformed[1], transformed[2]
            ))
            previous_length = float(
                np.linalg.norm(previous_right - anchor_center)
            )
            residual = abs(previous_length - left_length)
            cost += residual
            evidence.append(EdgeChainEvidence(
                (ordered[index - 1].piece_id,),
                (ordered[index].piece_id,),
                1,
                1,
                previous_length,
                left_length,
                residual / 2.0,
                residual / 2.0,
                0.0,
                (
                    tuple(anchor_center.tolist()),
                    tuple(previous_right.tolist()),
                ),
                (
                    tuple(anchor_center.tolist()),
                    tuple(transformed[1].tolist()),
                ),
            ))
        closure_right = transformed_incident[-1][2]
        closure_left = transformed_incident[0][1]
        closure_residual = float(np.linalg.norm(closure_right - closure_left))
        closure_limit = max(
            2.5 * tolerance_mm,
            0.14 * max(
                np.linalg.norm(closure_right - anchor_center),
                np.linalg.norm(closure_left - anchor_center),
            ),
        )
        if closure_residual > closure_limit:
            continue
        evidence.append(EdgeChainEvidence(
            (ordered[-1].piece_id,),
            (ordered[0].piece_id,),
            1,
            1,
            float(np.linalg.norm(closure_right - anchor_center)),
            float(np.linalg.norm(closure_left - anchor_center)),
            closure_residual / 2.0,
            closure_residual / 2.0,
            0.0,
            (
                tuple(anchor_center.tolist()),
                tuple(closure_right.tolist()),
            ),
            (
                tuple(anchor_center.tolist()),
                tuple(closure_left.tolist()),
            ),
        ))
        polygons = [Polygon(pose[2]) for pose in poses.values()]
        if any(not polygon.is_valid for polygon in polygons):
            continue
        overlap = float(pairwise_overlap(polygons))
        total_area = sum(float(polygon.area) for polygon in polygons)
        if overlap > max(12.0, 0.035 * total_area):
            continue
        union = unary_union(polygons)
        if union.geom_type != "Polygon":
            radius = max(0.2, 0.55 * tolerance_mm)
            union = unary_union([
                polygon.buffer(radius) for polygon in polygons
            ]).buffer(-radius)
        if union.geom_type != "Polygon" or union.is_empty:
            continue
        component = PartitionComponent(
            poses,
            union,
            tuple(evidence),
            float(cost + 0.2 * overlap),
        )
        results.append(PartitionSearchResult(
            component.poses,
            component.evidence,
            component.cost,
            "four_way_fan",
        ))
    results.sort(key=lambda item: _rectangle_rank(PartitionComponent(
        item.poses,
        unary_union([Polygon(pose[2]) for pose in item.poses.values()]),
        item.evidence,
        item.cost,
    )))
    return results[:maximum_results], evaluated


def _component_result(
    component: PartitionComponent,
    template: str,
) -> PartitionSearchResult:
    """把完整组件转换成带模板名的公开搜索结果。"""

    return PartitionSearchResult(
        component.poses,
        component.evidence,
        component.cost,
        template,
    )


def _subassembly_rank(component: PartitionComponent) -> tuple[float, ...]:
    """评价未完成子装配，只看闭合紧凑度和接口代价，不套最终尺寸。"""

    convex_gap, rectangle_gap = component.partial_rank
    return _stable_partial_rank((convex_gap, rectangle_gap, component.cost))


def _stable_partial_rank(rank: tuple[float, ...]) -> tuple[float, ...]:
    """Ignore sub-micron floating point tie noise only when ordering search work.

    CPU/Float32 fits of an exactly convex subassembly otherwise differ around
    zero and can reverse every later ranking criterion. Geometry admission and
    final rectangle checks continue to use their unrounded measurements.
    """
    return round(rank[0], 6), round(rank[1], 6), round(rank[2], 3)


def _disjoint_pair_tree_candidates(
    leaves: list[PartitionComponent],
    tolerance_mm: float,
    deadline: float | None,
    maximum_results: int,
) -> tuple[list[PartitionSearchResult], int]:
    """枚举四个节点的三种不相交配对，再合并两个复合区域。

    这只是通用二叉合并树的一种计算顺序，不约束碎片边数、切割角度、
    T 结点数量或目标答案类别。
    """

    pairings = (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    )
    results: list[PartitionSearchResult] = []
    expanded = 0
    pair_options: dict[tuple[int, int], list[PartitionComponent]] = {}
    for first_index, second_index in combinations(range(4), 2):
        if _deadline_reached(deadline):
            break
        pair_options[(first_index, second_index)] = _merge_options(
            leaves[first_index],
            leaves[second_index],
            tolerance_mm,
            20,
            allow_endpoint_partials=False,
            preserve_leaf_edges=False,
            deadline=deadline,
        )
    combinations_to_try: list[
        tuple[PartitionComponent, PartitionComponent]
    ] = []
    for first_pair, second_pair in pairings:
        combinations_to_try.extend(
            (first, second)
            for first in pair_options.get(first_pair, [])
            for second in pair_options.get(second_pair, [])
        )
    combinations_to_try.sort(key=lambda row: _stable_partial_rank(_state_rank(list(row))))
    for first, second in combinations_to_try[:144]:
        if _deadline_reached(deadline):
            break
        expanded += 1
        merged = _merge_options(
            first,
            second,
            tolerance_mm,
            80,
            allow_endpoint_partials=True,
            preserve_leaf_edges=True,
            deadline=deadline,
            complete_assembly=True,
        )
        generated = [
            _component_result(component, "edge_evidence_graph")
            for component in merged
        ]
        results.extend(generated)
        strong = []
        for result in generated:
            component = PartitionComponent(
                result.poses,
                unary_union([
                    Polygon(pose[2])
                    for pose in result.poses.values()
                ]),
                result.evidence,
                result.cost,
            )
            rank = _rectangle_rank(component)
            # 这只是停止继续扩张同类合并树的充分证据，不是最终安全门。
            # 真实实拍在联合拟合前约有 0.92～0.99 IoU，全部候选仍会在
            # solve_random 中用原始边图、安全质量和机器约束统一排序。
            if rank[0] <= 1.0 and rank[1] <= 0.08:
                strong.append(result)
        if strong:
            strong.sort(key=lambda item: _rectangle_rank(
                PartitionComponent(
                    item.poses,
                    unary_union([
                        Polygon(pose[2])
                        for pose in item.poses.values()
                    ]),
                    item.evidence,
                    item.cost,
                )
            ))
            return strong[:maximum_results], expanded
    results.sort(key=lambda item: _rectangle_rank(PartitionComponent(
        item.poses,
        unary_union([Polygon(pose[2]) for pose in item.poses.values()]),
        item.evidence,
        item.cost,
    )))
    return results[:maximum_results], expanded


def _rooted_growth_tree_candidates(
    leaves: list[PartitionComponent],
    tolerance_mm: float,
    deadline: float | None,
    maximum_results: int,
) -> tuple[list[PartitionSearchResult], int]:
    """Round-robin all four last-piece branches before revisiting a branch."""
    results: list[PartitionSearchResult] = []
    expanded = 0

    def branch(singleton):
        nonlocal expanded
        triple = [index for index in range(4) if index != singleton]
        triple_components = []
        for first_index, second_index in combinations(triple, 2):
            if _deadline_reached(deadline):
                return
            third_index = next(i for i in triple if i not in (first_index, second_index))
            pairs = _merge_options(
                leaves[first_index], leaves[second_index], tolerance_mm, 20,
                allow_endpoint_partials=False, preserve_leaf_edges=False, deadline=deadline,
            )
            yield []
            for pair in pairs[:16]:
                if _deadline_reached(deadline):
                    return
                expanded += 1
                triple_components.extend(_merge_options(
                    pair, leaves[third_index], tolerance_mm, 24,
                    allow_endpoint_partials=True, preserve_leaf_edges=True, deadline=deadline,
                ))
                yield []
        triple_components.sort(key=_subassembly_rank)
        for component in triple_components[:48]:
            if _deadline_reached(deadline):
                return
            expanded += 1
            merged = _merge_options(
                component, leaves[singleton], tolerance_mm, 80,
                allow_endpoint_partials=True, preserve_leaf_edges=True, deadline=deadline,
                complete_assembly=True,
            )
            yield [_component_result(c, "edge_evidence_graph") for c in merged]

    active = [iter(branch(i)) for i in range(4)]
    while active and not _deadline_reached(deadline):
        for iterator in active[:]:
            if _deadline_reached(deadline):
                break
            try:
                generated = next(iterator)
            except StopIteration:
                active.remove(iterator)
                continue
            results.extend(generated)
            strong = [r for r in generated if (
                lambda rank: rank[0] <= 1.0 and rank[1] <= 0.025
            )(_rectangle_rank(PartitionComponent(
                r.poses, unary_union([Polygon(p[2]) for p in r.poses.values()]), r.evidence, r.cost,
            ))) ]
            if strong:
                strong.sort(key=lambda r: _rectangle_rank(PartitionComponent(
                    r.poses, unary_union([Polygon(p[2]) for p in r.poses.values()]), r.evidence, r.cost,
                )))
                return strong[:maximum_results], expanded
    results.sort(key=lambda r: _rectangle_rank(PartitionComponent(
        r.poses, unary_union([Polygon(p[2]) for p in r.poses.values()]), r.evidence, r.cost,
    )))
    return results[:maximum_results], expanded


def _result_signature(result: PartitionSearchResult) -> tuple[Any, ...]:
    """按相对质心距离和模板构造四片候选去重签名。"""

    centers = {
        piece_id: np.mean(pose[2], axis=0)
        for piece_id, pose in result.poses.items()
    }
    distances = tuple(sorted(
        (
            first,
            second,
            round(float(np.linalg.norm(centers[first] - centers[second])), 1),
        )
        for first, second in combinations(sorted(centers), 2)
    ))
    return result.topology_template, distances


def enumerate_four_piece_candidates(
    pieces: list[ProcessedPiece],
    *,
    tolerance_mm: float,
    deadline: float | None,
    maximum_results: int,
    maximum_merge_options: int = 64,
    maximum_states_per_depth: int = 32,
    on_state: Callable[[], bool] | None = None,
    strict_only: bool = False,
    _run_strict_probe: bool = True,
    **_unused: Any,
) -> tuple[list[PartitionSearchResult], dict[str, Any]]:
    """用边链证据枚举四片的全部二叉分割解释。

    四向中心、双 T、条带和不平衡切割都能表示为同一棵二叉区域合并树。
    这里不先判断片形属于哪个答案模板，而是在每一层按成员分区保留搜索
    多样性，最后统一以矩形闭合、原始边对应和观测残差排序。
    """

    started = perf_counter()
    # Whole-chain matches have a much smaller search space than endpoint-sourced
    # T-junction variants. Probe them first; retain the full search on failure.
    # Whole-chain matching also applies to triangular, pentagonal and concave
    # pieces. Restricting this probe to quadrilaterals sent real mixed cards
    # through the much larger endpoint-partial search before validating a plan.
    if _run_strict_probe and all(3 <= len(piece.vertices) <= 5 for piece in pieces):
        strict_deadline = min(deadline, started+3.0) if deadline is not None else None
        strict_results, strict_diagnostics = _enumerate_recursive_partition_candidates(
            pieces, tolerance_mm=tolerance_mm, deadline=strict_deadline,
            maximum_results=maximum_results, maximum_merge_options=24,
            maximum_states_per_depth=32, allow_endpoint_partials=False,
            preserve_leaf_edges=False, on_state=on_state, _run_strict_probe=False)
        if strict_only:
            # These are proposals, not admitted solutions. Publish the best few
            # immediately to the full geometry/execution gates. A noisy IoU just
            # below the fast-stop heuristic must not discard a valid topology.
            strict_diagnostics.update(piece_count_branch='four_piece_edge_evidence_graph',
                partition_search_phase='strict_edge_chains',
                partition_search_strict_probe_succeeded=False)
            return [PartitionSearchResult(r.poses, r.evidence, r.cost, 'edge_evidence_graph')
                    for r in strict_results[:8]], strict_diagnostics
        exact = [result for result in strict_results if (
            lambda rank: rank[0] <= 1.0 and rank[1] <= .025)(_rectangle_rank(
                PartitionComponent(result.poses, unary_union([Polygon(p[2]) for p in result.poses.values()]),
                                   result.evidence, result.cost)))]
        if exact:
            strict_diagnostics.update(piece_count_branch='four_piece_edge_evidence_graph',
                partition_search_phase='strict_edge_chains', partition_search_strict_probe_succeeded=True)
            return [PartitionSearchResult(r.poses,r.evidence,r.cost,'edge_evidence_graph') for r in exact], strict_diagnostics
    junction_deadline = (
        None
        if deadline is None
        else min(deadline, perf_counter() + 2.0)
    )
    junction_raw, junction_expanded = _fan_candidates(
        pieces,
        tolerance_mm,
        junction_deadline,
        maximum_results,
    )
    junction_results = [
        PartitionSearchResult(
            result.poses,
            result.evidence,
            result.cost,
            "edge_evidence_graph",
        )
        for result in junction_raw
    ]
    # A closed four-way junction already supplies the complete edge-evidence graph.
    # White/shape-only solving does not need to spend another 7 s enumerating
    # disjoint trees for the same assembly. Refit actual piece poses in one batch;
    # admission, clearance and the final execution gate still run in search/solve.
    closed_junctions = []
    for result in junction_results:
        component = PartitionComponent(result.poses,
            unary_union([Polygon(pose[2]) for pose in result.poses.values()]),
            result.evidence, result.cost)
        rank = _rectangle_rank(component)
        if rank[0] <= 1.0 and rank[1] <= 0.025:
            closed_junctions.append(result)
    if closed_junctions:
        by_id = {piece.piece_id: piece for piece in pieces}
        for result in closed_junctions:
            ids = list(result.poses)
            fits = fit_search_batch([RigidFitRequest(by_id[key].vertices, result.poses[key][2]) for key in ids])
            for key, fit in zip(ids, fits):
                result.poses[key] = (fit.angle, fit.translation,
                    transform_points(by_id[key].vertices, fit.angle, fit.translation))
        return closed_junctions[:maximum_results], {
            'piece_count_branch': 'four_piece_edge_evidence_graph',
            'partition_search_policy': 'topology_agnostic_directed_edge_chain_graph',
            'partition_search_phase': 'closed_multiway_junction',
            'partition_search_tolerance_mm': tolerance_mm,
            'partition_search_elapsed_seconds': perf_counter()-started,
            'partition_search_generated_states': len(closed_junctions),
            'partition_search_expanded_states': junction_expanded,
            'partition_search_depth_counts': [len(closed_junctions)],
            'partition_search_template_counts': {}, 'partition_search_template_order': [],
            'partition_search_strict_probe_succeeded': False,
        }
    if deadline is None:
        disjoint_deadline = None
    else:
        remaining = max(0.0, deadline - started)
        disjoint_deadline = min(
            deadline,
            started + min(7.0, max(2.0, 0.25 * remaining)),
        )
    disjoint_results, disjoint_expanded = _disjoint_pair_tree_candidates(
        [_leaf(piece) for piece in pieces],
        tolerance_mm,
        disjoint_deadline,
        maximum_results,
    )
    exact_disjoint = []
    for result in junction_results + disjoint_results:
        component = PartitionComponent(
            result.poses,
            unary_union([
                Polygon(pose[2]) for pose in result.poses.values()
            ]),
            result.evidence,
            result.cost,
        )
        rank = _rectangle_rank(component)
        if rank[0] <= 1.0 and rank[1] <= 0.025:
            exact_disjoint.append(result)
    if exact_disjoint:
        exact_disjoint.sort(key=lambda item: _rectangle_rank(
            PartitionComponent(
                item.poses,
                unary_union([
                    Polygon(pose[2])
                    for pose in item.poses.values()
                ]),
                item.evidence,
                item.cost,
            )
        ))
        return exact_disjoint[:maximum_results], {
            "piece_count_branch": "four_piece_edge_evidence_graph",
            "partition_search_policy": (
                "topology_agnostic_directed_edge_chain_graph"
            ),
            "partition_search_phase": (
                "multiway_junction_and_disjoint_pair_binary_trees"
            ),
            "partition_search_tolerance_mm": tolerance_mm,
            "partition_search_elapsed_seconds": perf_counter() - started,
            "partition_search_generated_states": (
                len(junction_results) + len(disjoint_results)
            ),
            "partition_search_expanded_states": (
                junction_expanded + disjoint_expanded
            ),
            "partition_search_depth_counts": [
                len(junction_results), len(disjoint_results)
            ],
            "partition_search_template_counts": {},
            "partition_search_template_order": [],
            "partition_search_strict_probe_succeeded": False,
        }

    if deadline is None:
        rooted_deadline = None
    else:
        remaining = max(0.0, deadline - perf_counter())
        rooted_deadline = min(
            deadline,
            perf_counter() + min(12.0, max(3.0, 0.55 * remaining)),
        )
    rooted_results, rooted_expanded = _rooted_growth_tree_candidates(
        [_leaf(piece) for piece in pieces],
        tolerance_mm,
        rooted_deadline,
        maximum_results,
    )
    exact_rooted = []
    for result in junction_results + disjoint_results + rooted_results:
        component = PartitionComponent(
            result.poses,
            unary_union([
                Polygon(pose[2]) for pose in result.poses.values()
            ]),
            result.evidence,
            result.cost,
        )
        rank = _rectangle_rank(component)
        if rank[0] <= 1.0 and rank[1] <= 0.025:
            exact_rooted.append(result)
    if exact_rooted:
        exact_rooted.sort(key=lambda item: _rectangle_rank(
            PartitionComponent(
                item.poses,
                unary_union([
                    Polygon(pose[2])
                    for pose in item.poses.values()
                ]),
                item.evidence,
                item.cost,
            )
        ))
        return exact_rooted[:maximum_results], {
            "piece_count_branch": "four_piece_edge_evidence_graph",
            "partition_search_policy": (
                "topology_agnostic_directed_edge_chain_graph"
            ),
            "partition_search_phase": (
                "multiway_disjoint_and_rooted_binary_trees"
            ),
            "partition_search_tolerance_mm": tolerance_mm,
            "partition_search_elapsed_seconds": perf_counter() - started,
            "partition_search_generated_states": (
                len(junction_results)
                + len(disjoint_results)
                + len(rooted_results)
            ),
            "partition_search_expanded_states": (
                junction_expanded + disjoint_expanded + rooted_expanded
            ),
            "partition_search_depth_counts": [
                len(junction_results),
                len(disjoint_results),
                len(rooted_results),
            ],
            "partition_search_template_counts": {},
            "partition_search_template_order": [],
            "partition_search_strict_probe_succeeded": False,
        }

    recursive_results, diagnostics = _enumerate_recursive_partition_candidates(
        pieces,
        tolerance_mm=tolerance_mm,
        deadline=deadline,
        maximum_results=maximum_results,
        maximum_merge_options=max(32, min(maximum_merge_options, 48)),
        maximum_states_per_depth=max(
            32, min(maximum_states_per_depth, 64)
        ),
        on_state=on_state,
        allow_endpoint_partials=True,
        preserve_leaf_edges=True,
        preserve_partition_diversity=True,
        _run_strict_probe=False,
    )
    generalized_recursive = [
        PartitionSearchResult(
            result.poses,
            result.evidence,
            result.cost,
            "edge_evidence_graph",
        )
        for result in recursive_results
    ]
    combined = (
        junction_results
        + disjoint_results
        + rooted_results
        + generalized_recursive
    )
    combined.sort(key=lambda item: _rectangle_rank(PartitionComponent(
        item.poses,
        unary_union([
            Polygon(pose[2]) for pose in item.poses.values()
        ]),
        item.evidence,
        item.cost,
    )))
    generalized: list[PartitionSearchResult] = []
    signatures: set[tuple[Any, ...]] = set()
    for result in combined:
        signature = _result_signature(result)
        if signature in signatures:
            continue
        signatures.add(signature)
        generalized.append(result)
        if len(generalized) >= maximum_results:
            break
    diagnostics.update({
        "piece_count_branch": "four_piece_edge_evidence_graph",
        "partition_search_policy": (
            "topology_agnostic_directed_edge_chain_graph"
        ),
        "partition_search_phase": (
            "disjoint_pair_then_partition_diverse_binary_trees"
        ),
        "partition_search_disjoint_pair_states": disjoint_expanded,
        "partition_search_multiway_junction_states": junction_expanded,
        "partition_search_rooted_growth_states": rooted_expanded,
        "partition_search_template_counts": {},
        "partition_search_template_order": [],
    })
    return generalized, diagnostics
