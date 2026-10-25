from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from threading import Event
from typing import Callable

import numpy as np
from shapely import union_all
from shapely.geometry import Point, Polygon

from .geometry import Edge, identity, polygon_edges, rigid_from_segments, transform_point, transform_polygon
from .models import Candidate, PuzzlePiece, SolveResult


LogFn = Callable[[str], None]


@dataclass(slots=True)
class _State:
    poses: dict[int, np.ndarray]
    shapes: dict[int, Polygon]
    seams: list[dict]
    seam_length: float


class RectanglePuzzleSolver:
    """Small-N polygon assembly solver.

    It deliberately keeps edge-interface candidates until a whole-rectangle score is
    available.  Full-edge, long/short endpoint and shared-long-edge contacts are
    supported; no fixed cut topology is assumed.
    """

    def __init__(
        self,
        beam_width: int = 60,
        top_k: int = 40,
        *,
        allow_interior_interfaces: bool = False,
    ) -> None:
        self.beam_width = beam_width
        self.top_k = top_k
        # 默认保持既有的端点/整边接口枚举。只有四片近似四边形的专项
        # 加宽恢复会打开此项，以覆盖“两个相近长边只在中段相接”的 T 型
        # 或折线切缝，避免为普通随机输入扩大状态空间。
        self.allow_interior_interfaces = bool(allow_interior_interfaces)

        # Keep the old near-exact rectangle diagnostics.  They remain useful in
        # reports, but are deliberately no longer used to suppress a contest
        # result: real camera contours have more error than synthetic cuts.
        self.early_coverage = 0.99
        self.early_perimeter_ratio = 0.99
        # Final union is snapped to a 0.01-mm grid.  On a 6000-mm² board that
        # can create roughly 0.6 mm² of purely numerical overlap at an otherwise
        # exact seam, so retain a sub-1-mm² tolerance.
        self.early_max_overlap_mm2 = 0.80

    def solve(
        self,
        pieces: list[PuzzlePiece],
        timeout_s: float = 70.0,
        logger: LogFn | None = None,
        cancel: Event | None = None,
        allow_best_effort: bool = False,
        rescue_max_seconds: float | None = None,
    ) -> SolveResult:
        """Run a fast heuristic pass, then a wider recovery pass if needed."""
        started = time.monotonic()
        logger = logger or (lambda _: None)
        cancel = cancel or Event()
        ambiguous_four_quads = self._is_ambiguous_four_quadrilateral_input(pieces)
        if timeout_s <= 12.0:
            primary = self._solve_once(pieces, timeout_s, logger, cancel, allow_best_effort)
            if cancel.is_set():
                return primary
            return self._force_output(
                pieces,
                primary.candidates,
                primary.elapsed_s,
                primary.expanded_states,
                logger,
            )

        # Reserve enough time to recover from a misleading Beam ranking.  Most
        # ordinary inputs converge in the first phase and never pay this cost.
        # Keep the recovery pass to at most 25 seconds.  Four-piece layouts
        # still receive a useful 45-second primary pass at the normal 70-second
        # limit, while shorter manual limits retain at least 30 seconds for the
        # main search before any remaining time is handed to recovery.
        rescue_reserve = min(25.0, max(0.0, timeout_s - 30.0))
        if rescue_max_seconds is not None:
            if not np.isfinite(rescue_max_seconds) or rescue_max_seconds < 0.0:
                raise ValueError("rescue_max_seconds 必须为非负有限数")
            rescue_reserve = min(float(rescue_max_seconds), max(0.0, timeout_s - 12.0))
        fast_budget = timeout_s - rescue_reserve
        primary = self._solve_once(pieces, fast_budget, logger, cancel, allow_best_effort)
        if cancel.is_set():
            return primary

        remaining = max(0.0, timeout_s - primary.elapsed_s)
        # 第 3 问还需要用余下时间做花纹消歧，因此可显式限制救援搜索的
        # 最大时长；默认行为保持原先的“把所有剩余时间交给救援”。
        if rescue_max_seconds is not None:
            remaining = min(remaining, rescue_reserve)
        # 四片近似四边形的关键恢复是沿公共边的连续滑移，而不是继续扩大
        # 原有端点接口 Beam。为它留出一个小而确定的窗口，普通形状不变。
        sliding_reserve = 0.0
        if ambiguous_four_quads and primary.candidates:
            sliding_reserve = min(8.0, max(0.0, remaining - 1.0))
            remaining = max(0.0, remaining - sliding_reserve)
            if sliding_reserve >= 1.0:
                logger(
                    "四片滑移恢复预留 "
                    f"{sliding_reserve:.1f} 秒；旧 Beam 救援使用剩余 {remaining:.1f} 秒。"
                )
        if remaining < 1.0:
            return self._force_output(
                pieces,
                primary.candidates,
                primary.elapsed_s,
                primary.expanded_states,
                logger,
            )
        rescue_width = max(180, self.beam_width * 3)
        logger(
            f"主搜索阶段结束（{primary.status}）；启动救援搜索："
            f"Beam {self.beam_width} → {rescue_width}，剩余 {remaining:.1f} 秒。"
        )
        original_width = self.beam_width
        self.beam_width = rescue_width
        try:
            rescue = self._solve_once(pieces, remaining, logger, cancel, allow_best_effort)
        finally:
            self.beam_width = original_width

        candidates = self._deduplicate(primary.candidates + rescue.candidates)
        self._sort_candidates(candidates)
        elapsed = primary.elapsed_s + rescue.elapsed_s
        if cancel.is_set():
            return SolveResult(
                candidates[: self.top_k],
                elapsed,
                "已取消",
                primary.expanded_states + rescue.expanded_states,
            )
        # 对四片近似四边形的特殊歧义，正确拼缝有时不是“短边贴长边
        # 的任一端点”，而是两个已形成的子拼图沿该公共边相对滑动。
        # 主搜索保持原有有限接口枚举；这里只在明显非矩形时，对已有完整
        # 候选做有约束的一维恢复，不扩大普通随机拼图的搜索空间。
        candidates = self._refine_ambiguous_four_quad_slides(
            pieces,
            candidates,
            deadline=min(
                started + max(0.0, timeout_s),
                time.monotonic() + sliding_reserve,
            ),
            logger=logger,
        )
        self._sort_candidates(candidates)
        elapsed = time.monotonic() - started
        return self._force_output(
            pieces,
            candidates,
            elapsed,
            primary.expanded_states + rescue.expanded_states,
            logger,
        )

    @staticmethod
    def _is_ambiguous_four_quadrilateral_input(pieces: list[PuzzlePiece]) -> bool:
        """识别仅需做“子拼图沿缝滑移”恢复的窄输入集合。"""

        if len(pieces) != 4:
            return False
        areas: list[float] = []
        for piece in pieces:
            points = list(piece.polygon.exterior.coords)[:-1]
            if len(points) != 4:
                return False
            areas.append(float(piece.polygon.area))
        return bool(min(areas, default=0.0) > 1e-6 and max(areas) / min(areas) <= 1.45)

    @staticmethod
    def _needs_sliding_refinement(candidate: Candidate) -> bool:
        """避免对已可信的普通矩形多做任何枚举。"""

        metrics = candidate.metrics
        return bool(
            float(metrics.get("coverage", 0.0)) < 0.93
            or float(metrics.get("frame_corners", 0.0)) < 3.0
            or float(metrics.get("perimeter_ratio", 0.0)) < 0.90
        )

    @staticmethod
    def _split_seam_components(
        piece_ids: set[int], seams: list[dict], cut_index: int,
    ) -> tuple[set[int], set[int]] | None:
        """移除一条树拼缝后，返回其两侧保持刚体的子拼图集合。"""

        cut = seams[cut_index]
        first = int(cut["a_piece"])
        second = int(cut["b_piece"])
        if first not in piece_ids or second not in piece_ids:
            return None
        graph = {piece_id: set() for piece_id in piece_ids}
        for index, seam in enumerate(seams):
            if index == cut_index:
                continue
            a_piece = int(seam["a_piece"])
            b_piece = int(seam["b_piece"])
            if a_piece in graph and b_piece in graph:
                graph[a_piece].add(b_piece)
                graph[b_piece].add(a_piece)
        left: set[int] = set()
        stack = [first]
        while stack:
            current = stack.pop()
            if current in left:
                continue
            left.add(current)
            stack.extend(graph[current] - left)
        if second in left:
            # 这条 seam 不是树边；移动一侧会破坏其它闭环接触，因此跳过。
            return None
        right = piece_ids - left
        if not left or not right:
            return None
        return left, right

    def _refine_ambiguous_four_quad_slides(
        self,
        pieces: list[PuzzlePiece],
        candidates: list[Candidate],
        *,
        deadline: float,
        logger: LogFn,
    ) -> list[Candidate]:
        """在真实公共边上细化两个刚性子拼图的相对滑移。

        只扫描移除某一棵树拼缝后得到的两侧子图，故不会把单片任意漂移。
        对每个位置要求原公共边投影仍至少重合一小段，并用完整的矩形评分
        和面积重叠门复验；没有改善时原 candidates 原封不动地返回。
        """

        if (
            not candidates
            or not self._is_ambiguous_four_quadrilateral_input(pieces)
            or not self._needs_sliding_refinement(candidates[0])
            or time.monotonic() >= deadline
        ):
            return candidates

        by_id = {piece.id: piece for piece in pieces}
        total_area = float(sum(piece.polygon.area for piece in pieces))
        refined: list[Candidate] = []
        # 只检查质量最高的一小组完整候选，时间有界；每个候选最多三条树边。
        source_candidates = candidates[: min(16, len(candidates))]
        for source_index, source in enumerate(source_candidates):
            if time.monotonic() >= deadline:
                break
            piece_ids = set(source.poses)
            if piece_ids != set(by_id) or len(source.seams) < len(pieces) - 1:
                continue
            for seam_index, seam in enumerate(source.seams):
                if time.monotonic() >= deadline:
                    break
                try:
                    first = int(seam["a_piece"])
                    second = int(seam["b_piece"])
                    first_edge = int(seam["a_edge"])
                    second_edge = int(seam["b_edge"])
                    local_a = polygon_edges(by_id[first].polygon, first)[first_edge]
                    local_b = polygon_edges(by_id[second].polygon, second)[second_edge]
                except (KeyError, IndexError, ValueError):
                    continue
                world_a0 = transform_point(source.poses[first], local_a.p0)
                world_a1 = transform_point(source.poses[first], local_a.p1)
                world_b0 = transform_point(source.poses[second], local_b.p0)
                world_b1 = transform_point(source.poses[second], local_b.p1)
                direction = world_a1 - world_a0
                length_a = float(np.linalg.norm(direction))
                if length_a < 1e-6:
                    continue
                tangent = direction / length_a
                b_start = float(np.dot(world_b0 - world_a0, tangent))
                b_end = float(np.dot(world_b1 - world_a0, tangent))
                b_min, b_max = min(b_start, b_end), max(b_start, b_end)
                # 至少保留 5 mm 的真实公共边，远高于点接触，又兼容分段拼缝。
                minimum_contact = min(8.0, max(5.0, min(length_a, abs(b_end - b_start)) * 0.12))
                lower = minimum_contact - b_max
                upper = length_a - minimum_contact - b_min
                if upper - lower < 0.25:
                    continue
                components = self._split_seam_components(piece_ids, source.seams, seam_index)
                if components is None:
                    continue
                _left, moving = components
                # 先粗扫，再围绕最优位置 0.1 mm 细扫。0 必须保留，确保不会
                # 因数值采样而丢失原候选。
                coarse = sorted({float(value) for value in np.linspace(lower, upper, 31)} | {0.0})
                best_local: tuple[float, float] | None = None

                def make_candidate(delta: float) -> Candidate | None:
                    translation = np.array([
                        float(delta * tangent[0]), float(delta * tangent[1]),
                    ])
                    poses: dict[int, np.ndarray] = {}
                    shapes: dict[int, Polygon] = {}
                    for piece_id, pose in source.poses.items():
                        updated = np.array(pose, dtype=float, copy=True)
                        if piece_id in moving:
                            updated[:2, 2] += translation
                        poses[piece_id] = updated
                        shapes[piece_id] = transform_polygon(by_id[piece_id].polygon, updated)
                    # 公共边必须真的保持一段接触；仅仅外框变紧凑但拼缝已脱开
                    # 的候选不能进入结果池。
                    moved_b0 = world_b0 + translation
                    moved_b1 = world_b1 + translation
                    projected_min = min(float(np.dot(moved_b0 - world_a0, tangent)), float(np.dot(moved_b1 - world_a0, tangent)))
                    projected_max = max(float(np.dot(moved_b0 - world_a0, tangent)), float(np.dot(moved_b1 - world_a0, tangent)))
                    contact = max(0.0, min(length_a, projected_max) - max(0.0, projected_min))
                    if contact + 1e-7 < minimum_contact:
                        return None
                    adjusted_seams = [dict(item) for item in source.seams]
                    adjusted_seams[seam_index]["kind"] = "partial-interior-refined"
                    adjusted_seams[seam_index]["length"] = float(contact)
                    seam_length = float(sum(float(item.get("length", 0.0)) for item in adjusted_seams))
                    candidate = self._score_complete(
                        _State(poses, shapes, adjusted_seams, seam_length),
                        pieces,
                        total_area,
                    )
                    if candidate is None:
                        return None
                    # 保持原有数值容忍度；不让“外框更紧”以实质面积重叠为代价。
                    if float(candidate.metrics.get("overlap_mm2", float("inf"))) > self.early_max_overlap_mm2:
                        return None
                    candidate.metrics["sliding_refinement"] = 1.0
                    candidate.metrics["sliding_delta_mm"] = float(delta)
                    candidate.metrics["sliding_contact_mm"] = float(contact)
                    candidate.metrics["sliding_source_rank"] = float(source_index)
                    return candidate

                for delta in coarse:
                    if time.monotonic() >= deadline:
                        break
                    candidate = make_candidate(delta)
                    if candidate is None:
                        continue
                    refined.append(candidate)
                    rank = (
                        float(candidate.metrics.get("coverage", 0.0)),
                        float(candidate.score),
                    )
                    if best_local is None or rank > best_local:
                        best_local = rank
                        best_delta = delta
                if best_local is None or time.monotonic() >= deadline:
                    continue
                step = max((upper - lower) / 30.0, 0.1)
                for delta in np.arange(max(lower, best_delta - step), min(upper, best_delta + step) + 1e-9, 0.1):
                    if time.monotonic() >= deadline:
                        break
                    candidate = make_candidate(float(delta))
                    if candidate is not None:
                        refined.append(candidate)
        if not refined:
            logger("四片滑移恢复：未找到满足公共边接触和无重叠约束的改进候选。")
            return candidates
        merged = self._deduplicate(candidates + refined)
        self._sort_candidates(merged)
        best = merged[0]
        logger(
            "四片滑移恢复：追加 "
            f"{len(refined)} 个受约束候选，当前最佳覆盖率 "
            f"{best.metrics.get('coverage', 0.0):.2%}。"
        )
        return merged

    def _solve_once(
        self,
        pieces: list[PuzzlePiece],
        timeout_s: float = 70.0,
        logger: LogFn | None = None,
        cancel: Event | None = None,
        allow_best_effort: bool = False,
    ) -> SolveResult:
        started = time.monotonic()
        logger = logger or (lambda _: None)
        cancel = cancel or Event()
        if not 2 <= len(pieces) <= 4:
            return SolveResult([], 0.0, "需要 2 至 4 块碎片", 0)

        by_id = {piece.id: piece for piece in pieces}
        edge_map = {piece.id: polygon_edges(piece.polygon, piece.id) for piece in pieces}
        total_edges = sum(len(edges) for edges in edge_map.values())
        total_area = sum(piece.polygon.area for piece in pieces)
        logger(f"检测到 {len(pieces)} 块碎片、{total_edges} 条有效边；开始生成边接口。")
        logger(f"碎片总面积 {total_area:.2f} mm²；完整矩形的外框面积必须与其一致。")

        all_candidates: list[Candidate] = []
        expanded = 0
        # A connected assembly can be grown from any chosen piece.  Fixing the
        # smallest ID as the root removes N equivalent global-coordinate searches.
        root = min(pieces, key=lambda piece: piece.id)
        logger(f"以碎片 {root.id + 1} 为固定根搜索布局。")
        initial = _State({root.id: identity()}, {root.id: root.polygon}, [], 0.0)
        frontier = [initial]
        for depth in range(1, len(pieces)):
            # Four-piece cases can spend the whole budget enumerating the
            # three-piece frontier, leaving no time to actually close the last
            # residual gap.  Preserve a small, bounded finalization reserve;
            # it is only used while growing the directly preceding frontier.
            growth_timeout_s = timeout_s
            prune_timeout_s = timeout_s
            if depth == len(pieces) - 2:
                finalization_reserve_s = min(
                    8.0, max(2.0, timeout_s * 0.20)
                )
                pruning_reserve_s = min(2.0, finalization_reserve_s * 0.25)
                growth_timeout_s = max(
                    0.1, timeout_s - finalization_reserve_s - pruning_reserve_s
                )
                prune_timeout_s = max(0.1, timeout_s - finalization_reserve_s)
            next_states: list[_State] = []
            for state in frontier:
                if self._stop(started, growth_timeout_s, cancel):
                    break
                allow_interior_interval = self.allow_interior_interfaces
                if depth == len(pieces) - 1:
                    _, oversize, frame_corners, _, _ = self._partial_frame_features(state.shapes.values())
                    # Interior sliding is a recovery tool for an otherwise
                    # plausible nearly-complete frame, not a broad source of
                    # random placements.
                    allow_interior_interval = frame_corners >= 2 and oversize <= 1.5
                # Each placed edge is reused for every unplaced piece in this
                # state.  Transform it once here rather than repeating the
                # same rigid transform inside every next_id branch.
                world_edges = {
                    placed_id: [
                        Edge(
                            placed_id,
                            placed_edge.index,
                            transform_point(state.poses[placed_id], placed_edge.p0),
                            transform_point(state.poses[placed_id], placed_edge.p1),
                            placed_edge.length,
                        )
                        for placed_edge in edge_map[placed_id]
                    ]
                    for placed_id in state.poses
                }
                for next_id in by_id:
                    if self._stop(started, growth_timeout_s, cancel):
                        break
                    if next_id in state.poses:
                        continue
                    if depth == len(pieces) - 1:
                        # When three pieces already define the final frame, the
                        # remaining polygon can be fitted directly into its gap.
                        # This uses several boundary constraints at once instead
                        # of hoping a single-edge sliding sample lands at the
                        # correct partial-to-partial seam position.
                        for pose in self._gap_completion_poses(state, by_id[next_id], total_area):
                            expanded += 1
                            if self._stop(started, growth_timeout_s, cancel):
                                break
                            new_poly = transform_polygon(by_id[next_id].polygon, pose)
                            if not self._non_overlapping(new_poly, state.shapes.values()):
                                continue
                            gap_state = _State(
                                {**state.poses, next_id: pose},
                                {**state.shapes, next_id: new_poly},
                                state.seams,
                                state.seam_length,
                            )
                            candidate = self._score_complete(
                                gap_state, pieces, total_area
                            )
                            if candidate is None:
                                continue
                            all_candidates.append(candidate)
                        if self._stop(started, growth_timeout_s, cancel):
                            break
                    for placed_id in tuple(state.poses):
                        if self._stop(started, growth_timeout_s, cancel):
                            break
                        for placed in world_edges[placed_id]:
                            if self._stop(started, growth_timeout_s, cancel):
                                break
                            for raw_edge in edge_map[next_id]:
                                if self._stop(started, growth_timeout_s, cancel):
                                    break
                                for pose, kind in self._interface_poses(
                                    placed,
                                    raw_edge,
                                    allow_interior_interval=allow_interior_interval,
                                ):
                                    expanded += 1
                                    if self._stop(started, growth_timeout_s, cancel):
                                        break
                                    new_poly = transform_polygon(by_id[next_id].polygon, pose)
                                    if not self._non_overlapping(new_poly, state.shapes.values()):
                                        continue
                                    next_state = _State(
                                        {**state.poses, next_id: pose},
                                        {**state.shapes, next_id: new_poly},
                                        state.seams
                                        + [
                                            {
                                                "a_piece": placed_id,
                                                "a_edge": placed.index,
                                                "b_piece": next_id,
                                                "b_edge": raw_edge.index,
                                                "kind": kind,
                                                "length": min(placed.length, raw_edge.length),
                                            }
                                        ],
                                        state.seam_length + min(placed.length, raw_edge.length),
                                    )
                                    if depth == len(pieces) - 1:
                                        # A complete layout carries enough
                                        # information for exact global scoring.
                                        # Do it now instead of waiting for every
                                        # remaining branch to be enumerated.
                                        candidate = self._score_complete(
                                            next_state, pieces, total_area
                                        )
                                        if candidate is None:
                                            continue
                                        all_candidates.append(candidate)
                                    else:
                                        next_states.append(next_state)
                                if self._stop(started, growth_timeout_s, cancel):
                                    break
            if self._stop(started, timeout_s, cancel):
                break
            if depth == len(pieces) - 1:
                logger(f"深度 {depth}：已即时评分 {len(all_candidates)} 个完整候选。")
                break
            frontier = self._prune(
                next_states, total_area, started, prune_timeout_s, cancel
            )
            logger(f"深度 {depth}：生成 {len(next_states)} 个布局，保留 {len(frontier)} 个候选。")
            if not frontier:
                break
        all_candidates = self._deduplicate(all_candidates)
        self._sort_candidates(all_candidates)
        elapsed = time.monotonic() - started
        if cancel.is_set():
            status = "已取消"
        elif elapsed >= timeout_s:
            status = "达到时间上限"
        elif all_candidates:
            status = "完成"
        else:
            status = "没有找到完整布局候选"
        logger(f"{status}：完整候选 {len(all_candidates)} 个，展开状态 {expanded} 个，用时 {elapsed:.2f} 秒。")
        return SolveResult(all_candidates[: self.top_k], elapsed, status, expanded)

    @staticmethod
    def _candidate_sort_key(candidate: Candidate) -> tuple[float, float, float, float]:
        """Rank output candidates by coverage first, then existing geometry score."""
        metrics = candidate.metrics
        return (
            metrics.get("coverage", 0.0),
            candidate.score,
            metrics.get("perimeter_ratio", 0.0),
            -metrics.get("overlap_mm2", float("inf")),
        )

    def _sort_candidates(self, candidates: list[Candidate]) -> None:
        candidates.sort(key=self._candidate_sort_key, reverse=True)

    def _force_output(
        self,
        pieces: list[PuzzlePiece],
        candidates: list[Candidate],
        elapsed_s: float,
        expanded_states: int,
        logger: LogFn,
    ) -> SolveResult:
        """Return a usable candidate at deadline; never leave the UI empty.

        Complete states are built with pairwise non-overlap checks.  At this
        final selection point we intentionally no longer reject them for an
        imperfect outline, corner count, dimensions, or numerical overlap
        estimate.  If the search did not reach any complete state at all, keep
        every piece at its observed pose as a clearly labelled last fallback.
        """
        candidates = self._deduplicate(candidates)
        if not candidates:
            fallback = self._input_layout_candidate(pieces)
            if fallback is not None:
                candidates = [fallback]
        self._sort_candidates(candidates)
        if candidates:
            best = candidates[0]
            best.metrics["forced_output"] = 1.0
            status = (
                "时间到：输出最佳候选"
                f"（覆盖率 {best.metrics.get('coverage', 0.0):.2%}）"
            )
        else:
            # Only malformed inputs (for example non-polygon geometry) can
            # arrive here.  Supported 2-4 piece puzzle inputs always get the
            # identity-pose fallback above.
            status = "无法构造保底候选"
        logger(
            f"{status}：候选 {len(candidates)} 个，展开状态 {expanded_states} 个，"
            f"用时 {elapsed_s:.2f} 秒。"
        )
        return SolveResult(candidates[: self.top_k], elapsed_s, status, expanded_states)

    def _input_layout_candidate(self, pieces: list[PuzzlePiece]) -> Candidate | None:
        """Build an all-piece fallback pose for a supported puzzle input."""
        if not 2 <= len(pieces) <= 4:
            return None
        state = _State(
            poses={piece.id: identity() for piece in pieces},
            shapes={piece.id: piece.polygon for piece in pieces},
            seams=[],
            seam_length=0.0,
        )
        candidate = self._score_complete(
            state,
            pieces,
            sum(piece.polygon.area for piece in pieces),
        )
        if candidate is not None:
            candidate.metrics["input_pose_fallback"] = 1.0
            return candidate
        # This should be unreachable for valid Polygon inputs, but it keeps the
        # output contract intact even when a vision contour is numerically odd.
        return Candidate(
            poses=state.poses,
            score=-1_000_000.0,
            metrics={
                "coverage": 0.0,
                "overlap_mm2": 0.0,
                "short_mm": 0.0,
                "long_mm": 0.0,
                "perimeter_ratio": 0.0,
                "seam_mm": 0.0,
                "frame_corners": 0.0,
                "outer_piece_count": 0.0,
                "neighbor_pairs": 0.0,
                "contact_mm": 0.0,
                "closed_cycles": 0.0,
                "topology_grid_mm": 0.0,
                "trusted_rectangle": 0.0,
                "input_pose_fallback": 1.0,
            },
        )

    def _interface_poses(self, a: Edge, b: Edge, allow_interior_interval: bool = False) -> list[tuple[np.ndarray, str]]:
        """Return finite placements for an opposite-facing shared interface.

        Matching a shorter edge at both endpoints of a long edge lets two pieces
        share one long edge (a T junction) without forbidding edge reuse.
        """
        if min(a.length, b.length) < 18.0:  # Spec is >= 20 mm; retain measurement slack.
            return []
        result: list[tuple[np.ndarray, str]] = []
        length_delta = abs(a.length - b.length)
        near_equal = length_delta <= max(1.2, 0.045 * max(a.length, b.length))
        if near_equal:
            pose = rigid_from_segments(b.p0, b.p1, a.p1, a.p0)
            if pose is not None:
                result.append((pose, "full"))
            # 对常规路径，原有“近等长即整边拼合”的快速假设保持不变。
            # 专项恢复则还要允许二者仅在中段形成一小段真实拼缝。
            if not allow_interior_interval:
                return result

        av = a.p1 - a.p0
        bv = b.p1 - b.p0
        au = av / a.length
        if not near_equal and a.length > b.length:
            for offset, label in ((0.0, "partial-start"), (a.length - b.length, "partial-end")):
                start = a.p0 + au * offset
                end = start + au * b.length
                pose = rigid_from_segments(b.p0, b.p1, end, start)
                if pose is not None:
                    result.append((pose, label))
        elif not near_equal:
            # B is the long edge; place A at either B endpoint.
            bu = bv / b.length
            for offset, label in ((0.0, "partial-start"), (b.length - a.length, "partial-end")):
                b_start = a.p1 + au * offset
                b_end = b_start - au * b.length
                pose = rigid_from_segments(b.p0, b.p1, b_start, b_end)
                if pose is not None:
                    result.append((pose, label))
        ratio = min(a.length, b.length) / max(a.length, b.length)
        # Long/short contacts are already represented by endpoint placement.
        # Interior-to-interior fallback is for two comparably long parent edges.
        if allow_interior_interval and ratio >= 0.65:
            result.extend(self._interior_interval_poses(a, b))
        return result

    @staticmethod
    def _gap_completion_poses(state: _State, piece: PuzzlePiece, total_area: float) -> list[np.ndarray]:
        """Fit the final piece into a nearly-defined rectangular residual gap.

        A three-piece L/田字 state can already expose the target rectangle.  Its
        residual gap is then a much stronger constraint than one individual
        seam, including for arbitrary interior partial-edge contacts.
        """
        partial = union_all(list(state.shapes.values()), grid_size=0.01)
        if partial.geom_type != "Polygon" or partial.area <= 1e-6:
            return []
        rectangle = partial.minimum_rotated_rectangle
        if rectangle.area > total_area * 1.015 or rectangle.area < total_area * 0.94:
            return []
        gap = rectangle.difference(partial)
        if gap.geom_type != "Polygon" or gap.area <= 1e-6:
            return []
        if abs(gap.area - piece.polygon.area) > max(2.0, piece.polygon.area * 0.025):
            return []

        result: list[np.ndarray] = []
        seen: set[tuple[float, float, float]] = set()
        for raw_edge in polygon_edges(piece.polygon, piece.id):
            for gap_edge in polygon_edges(gap, -1):
                if abs(raw_edge.length - gap_edge.length) > max(1.5, 0.07 * max(raw_edge.length, gap_edge.length)):
                    continue
                for target_start, target_end in ((gap_edge.p0, gap_edge.p1), (gap_edge.p1, gap_edge.p0)):
                    pose = rigid_from_segments(raw_edge.p0, raw_edge.p1, target_start, target_end)
                    if pose is None:
                        continue
                    placed = transform_polygon(piece.polygon, pose)
                    if placed.symmetric_difference(gap).area > max(2.0, piece.polygon.area * 0.015):
                        continue
                    key = (
                        round(float(pose[0, 2]), 2),
                        round(float(pose[1, 2]), 2),
                        round(float(np.arctan2(pose[1, 0], pose[0, 0])), 3),
                    )
                    if key not in seen:
                        seen.add(key)
                        result.append(pose)
        return result

    @staticmethod
    def _interior_interval_poses(a: Edge, b: Edge) -> list[tuple[np.ndarray, str]]:
        """Generate a small, finite set of partial-to-partial edge contacts.

        A legal cut can leave a short shared segment in the interior of *both*
        original edges.  The statement constrains original edge lengths, not the
        resulting sub-seam length.  Endpoints plus a few interior landmarks are
        sufficient for the final-piece completion search, where all other frame
        and seam constraints have already narrowed the layout.
        """
        if min(a.length, b.length) < 18.0:
            return []
        min_contact = max(4.0, min(a.length, b.length) * 0.12)
        lower = -b.length + min_contact
        upper = a.length - min_contact
        if upper <= lower:
            return []
        # t is B's directed-edge start projected on A.  B occupies [t,t+Lb]
        # along A's supporting line; only the interval intersection is a seam.
        landmarks = {0.0, a.length - b.length, lower, upper, (a.length - b.length) * 0.5}
        for value in np.linspace(lower, upper, 7):
            landmarks.add(float(value))
        direction = (a.p1 - a.p0) / a.length
        result: list[tuple[np.ndarray, str]] = []
        for offset in sorted(landmarks):
            overlap_start = max(0.0, offset)
            overlap_end = min(a.length, offset + b.length)
            if overlap_end - overlap_start < min_contact:
                continue
            start = a.p0 + direction * offset
            end = start + direction * b.length
            pose = rigid_from_segments(b.p0, b.p1, end, start)
            if pose is not None:
                result.append((pose, "partial-interior"))
        return result

    @staticmethod
    def _non_overlapping(new_poly: Polygon, existing: object) -> bool:
        new_min_x, new_min_y, new_max_x, new_max_y = new_poly.bounds
        for old_poly in existing:
            old_min_x, old_min_y, old_max_x, old_max_y = old_poly.bounds
            if (
                new_max_x <= old_min_x
                or old_max_x <= new_min_x
                or new_max_y <= old_min_y
                or old_max_y <= new_min_y
            ):
                continue
            if new_poly.intersection(old_poly).area > 0.08:
                return False
        return new_poly.is_valid

    def _prune(
        self,
        states: list[_State],
        total_area: float,
        started: float | None = None,
        timeout_s: float | None = None,
        cancel: Event | None = None,
    ) -> list[_State]:
        if not states:
            return []
        ranked: list[tuple[float, _State]] = []
        for index, state in enumerate(states):
            if (
                started is not None
                and timeout_s is not None
                and cancel is not None
                and index % 32 == 0
                and self._stop(started, timeout_s, cancel)
            ):
                break
            if len(state.shapes) >= 2:
                # A valid T/segmented seam can have *less* matched length than
                # a false compact layout, so never use seam length as the
                # primary ranking signal.  The finished object must be a
                # 90-120 by 50-90 mm rectangle: retain partial layouts that
                # already cover a plausible rectangle without exceeding those
                # outer dimensions.  Applying this at two pieces is important:
                # a true sparse pair can otherwise be removed before the
                # decisive three-piece frontier is reached.
                coverage, oversize, frame_corners, mrr_area, union_area = self._partial_frame_features(state.shapes.values())
                # Any final rectangle must contain the current partial union.
                # Since the total material area is known, an enclosing rectangle
                # already larger than it cannot ever become a no-gap solution.
                # This removes the attractive-but-impossible long-strip states
                # that otherwise dominate a near-rectangular 田字 layout.
                if mrr_area > total_area * 1.015:
                    continue
                remaining_area = max(0.0, total_area - union_area)
                inner_gap_area = max(0.0, mrr_area - union_area)
                # When a partial layout already spans both target dimensions,
                # its MRR gap should be exactly the material left to place.  An
                # L-shaped three-piece 田字 state scores highly here; a strip
                # whose frame has grown too large scores poorly (or is rejected
                # above).  Small two-piece layouts remain possible because they
                # may still grow outward with later pieces.
                completion_fit = max(
                    0.0,
                    1.0 - abs(remaining_area - inner_gap_area) / max(total_area * 0.06, 1e-6),
                )
                # The global rectangle's corners must be piece vertices with a
                # convex near-right interior angle.  This is a strong contest
                # prior, not a hard rule: a missing piece can still own one or
                # more future corners, but layouts that already expose valid
                # frame corners deserve to survive the Beam.
                score = coverage * 100.0 - oversize * 8.0 + frame_corners * 12.0
                if len(state.shapes) >= 3:
                    score += completion_fit * 90.0
            ranked.append((score, state))
        ranked.sort(key=lambda item: item[0], reverse=True)
        seen: set[tuple] = set()
        result: list[_State] = []
        # Do not let one attractive early pair consume the whole Beam.  States
        # with different placed-piece sets represent exactly the later "swap the
        # remaining two pieces" alternatives observed in manual testing.
        subsets = {tuple(sorted(state.poses)) for _, state in ranked}
        per_subset_limit = max(8, self.beam_width // max(1, len(subsets)))
        kept_per_subset: dict[tuple[int, ...], int] = {subset: 0 for subset in subsets}

        def try_add(state: _State, enforce_subset_limit: bool) -> bool:
            subset = tuple(sorted(state.poses))
            if enforce_subset_limit and kept_per_subset[subset] >= per_subset_limit:
                return False
            key = tuple(
                (pid, round(pose[0, 2] / 2.0), round(pose[1, 2] / 2.0), round(np.arctan2(pose[1, 0], pose[0, 0]), 2))
                for pid, pose in sorted(state.poses.items())
            )
            if key in seen:
                return False
            seen.add(key)
            kept_per_subset[subset] += 1
            result.append(state)
            return True

        for _, state in ranked:
            try_add(state, enforce_subset_limit=True)
        # Fill unused capacity by score after every subset has representation.
        for _, state in ranked:
            if len(result) >= self.beam_width:
                break
            try_add(state, enforce_subset_limit=False)
        return result[: self.beam_width]

    @staticmethod
    def _partial_frame_features(shapes: object) -> tuple[float, float, int, float, float]:
        polygons = list(shapes)
        partial_union = union_all(polygons, grid_size=0.01)
        rectangle = partial_union.minimum_rotated_rectangle
        corners = list(rectangle.exterior.coords)[:-1]
        sides = [float(np.linalg.norm(np.subtract(corners[(i + 1) % 4], corners[i]))) for i in range(4)]
        short_side, long_side = min(sides), max(sides)
        coverage = partial_union.area / max(rectangle.area, 1e-8)
        # A partial assembly can be smaller than the final rectangle because the
        # missing piece may contribute an outer side.  It can never already be
        # larger than the permitted maximum.
        oversize = max(0.0, short_side - 92.0) + max(0.0, long_side - 122.0)

        frame_corners = RectanglePuzzleSolver._frame_corner_count(polygons, corners)
        return coverage, oversize, frame_corners, rectangle.area, partial_union.area

    @staticmethod
    def _frame_corner_count(polygons: list[Polygon], corners: list[tuple[float, float]]) -> int:
        """Count MRR corners supported by convex near-right piece vertices."""
        right_vertices: list[np.ndarray] = []
        for polygon in polygons:
            coords = [np.asarray(point, dtype=float) for point in list(polygon.exterior.coords)[:-1]]
            winding = 1.0 if polygon.exterior.is_ccw else -1.0
            for index, current in enumerate(coords):
                previous = coords[index - 1]
                following = coords[(index + 1) % len(coords)]
                incoming = current - previous
                outgoing = following - current
                in_length = float(np.linalg.norm(incoming))
                out_length = float(np.linalg.norm(outgoing))
                if in_length < 1e-8 or out_length < 1e-8:
                    continue
                cosine = float(np.clip(np.dot(-incoming, outgoing) / (in_length * out_length), -1.0, 1.0))
                angle_deg = float(np.degrees(np.arccos(cosine)))
                # Concave vertices can have a 90-degree *smaller* angle but can
                # never own a convex outer rectangle corner.
                cross = float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0])
                if 78.0 <= angle_deg <= 102.0 and cross * winding > 0.0:
                    right_vertices.append(current)
        # A 1.5-mm tolerance accommodates detected vertices and grid snapping.
        return sum(any(float(np.linalg.norm(np.subtract(corner, vertex))) <= 1.5 for vertex in right_vertices) for corner in corners)

    def _score_complete(
        self,
        state: _State,
        pieces: list[PuzzlePiece],
        total_area: float,
    ) -> Candidate | None:
        shapes = list(state.shapes.values())
        # Pairwise rigid transforms can leave sub-micron gaps at a multi-edge
        # or segmented seam.  A fixed snap grid is unsafe here: on oblique
        # edges, 0.01 mm can split a seam that 0.001 mm preserves.  Try the
        # least invasive precision first and retain the first connected union;
        # all normal rectangle/overlap gates below remain in force.
        union = None
        topology_grid_mm = 0.0
        union_rank: tuple[float, float, float] | None = None
        for grid_size in (None, 0.001, 0.002, 0.005, 0.01, 0.02):
            candidate_union = (
                union_all(shapes)
                if grid_size is None
                else union_all(shapes, grid_size=grid_size)
            )
            # A slightly noisy real-camera contour can leave a microscopic gap
            # and therefore form a MultiPolygon.  It is still a meaningful
            # complete placement to rank at the deadline; its low coverage will
            # naturally keep it behind a true rectangular union.
            if candidate_union.is_empty or candidate_union.area <= 1e-6:
                continue
            candidate_mrr = candidate_union.minimum_rotated_rectangle
            if candidate_mrr.area <= 1e-6:
                continue
            candidate_coverage = candidate_union.area / candidate_mrr.area
            candidate_perimeter = min(
                1.0, candidate_mrr.length / max(candidate_union.length, 1e-6)
            )
            # A raw floating-point union can be a formally valid Polygon while
            # retaining microscopic doubled boundary segments.  Prefer the
            # union closest to a rectangle, and use the finer grid as a tie
            # breaker.  This keeps normal exact geometry untouched, while a
            # 0.001-mm repair wins over a coarser oblique-edge snap.
            rank = (
                2.0 * candidate_coverage + candidate_perimeter,
                candidate_coverage,
                -float(grid_size or 0.0),
            )
            if union_rank is None or rank > union_rank:
                union = candidate_union
                topology_grid_mm = float(grid_size or 0.0)
                union_rank = rank
        if union is None:
            return None
        overlap = max(0.0, total_area - union.area)
        mrr = union.minimum_rotated_rectangle
        mrr_area = mrr.area
        if mrr_area <= 1e-6:
            return None
        coords = list(mrr.exterior.coords)
        mrr_corners = coords[:-1]
        sides = [float(np.linalg.norm(np.subtract(coords[i + 1], coords[i]))) for i in range(4)]
        short_side, long_side = min(sides), max(sides)
        coverage = union.area / mrr_area
        perimeter_ratio = min(1.0, mrr.length / max(union.length, 1e-6))
        frame_corners = self._frame_corner_count(shapes, mrr_corners)
        outer_piece_count = 0
        for shape in shapes:
            # The statement guarantees at least one entire original edge of
            # every piece lies on the rectangle exterior.  Midpoint distance is
            # robust to the tiny snapping used for union validation.
            has_outer_edge = any(
                union.boundary.distance(Point((edge.p0 + edge.p1) * 0.5)) <= 0.8
                for edge in polygon_edges(shape, -1)
            )
            outer_piece_count += int(has_outer_edge)
        size_penalty = 0.0
        if short_side < 48.0 or short_side > 92.0:
            size_penalty += min(abs(short_side - 48.0), abs(short_side - 92.0)) / 10.0
        if long_side < 88.0 or long_side > 122.0:
            size_penalty += min(abs(long_side - 88.0), abs(long_side - 122.0)) / 10.0
        # Internal seam length is not a correctness signal: a wrong layout can
        # create a longer seam than the true segmented-edge topology.  A closed
        # rectangular frame with genuine right-angle piece vertices is stronger.
        neighbor_pairs, contact_length = self._contact_topology(shapes)
        closed_cycles = max(0, neighbor_pairs - (len(shapes) - 1))
        # Do not require a cycle: some legal random-cut instances are a tree.
        # But a 田字-style final placement normally closes an additional seam,
        # which is a reliable preference when geometrically valid answers tie.
        score = (
            100.0 * coverage
            + 18.0 * perimeter_ratio
            + 12.0 * frame_corners
            + 0.03 * state.seam_length
            + 4.0 * closed_cycles
            + 0.01 * contact_length
            - 80.0 * overlap / max(total_area, 1.0)
            - 12.0 * size_penalty
            - 25.0 * (len(shapes) - outer_piece_count)
        )
        trusted_rectangle = bool(
            coverage >= self.early_coverage
            and perimeter_ratio >= self.early_perimeter_ratio
            and frame_corners >= 4.0
            and outer_piece_count == len(shapes)
        )
        return Candidate(
            poses=state.poses,
            score=score,
            metrics={
                "coverage": coverage,
                "overlap_mm2": overlap,
                "short_mm": short_side,
                "long_mm": long_side,
                "perimeter_ratio": perimeter_ratio,
                "seam_mm": state.seam_length,
                "frame_corners": float(frame_corners),
                "outer_piece_count": float(outer_piece_count),
                "neighbor_pairs": float(neighbor_pairs),
                "contact_mm": contact_length,
                "closed_cycles": float(closed_cycles),
                "topology_grid_mm": topology_grid_mm,
                "trusted_rectangle": float(trusted_rectangle),
            },
            seams=state.seams,
        )

    @staticmethod
    def _contact_topology(shapes: list[Polygon]) -> tuple[int, float]:
        """Measure true piece-to-piece boundary contacts in a complete layout."""
        pairs = 0
        total_length = 0.0
        for first, second in itertools.combinations(shapes, 2):
            length = first.boundary.intersection(second.boundary).length
            if length < 1.0:
                # A pose reached through a different chain of rigid transforms
                # can miss an otherwise common seam by a few hundredths of a
                # millimetre.  Measure the portion of the second boundary lying
                # within a narrow tolerance tube around the first one.  A point
                # contact contributes far below the 1-mm seam threshold below.
                length = first.boundary.buffer(0.15).intersection(second.boundary).length
            # Ignore isolated shared vertices: only a tangible common edge is a
            # puzzle seam.  The threshold is far below the 20-mm original-edge
            # rule because a legal segmented seam can be short.
            if length >= 1.0:
                pairs += 1
                total_length += length
        return pairs, total_length

    @staticmethod
    def _deduplicate(candidates: list[Candidate]) -> list[Candidate]:
        result: list[Candidate] = []
        seen: set[tuple] = set()
        for candidate in candidates:
            key = tuple(
                (pid, round(pose[0, 2]), round(pose[1, 2]), round(np.arctan2(pose[1, 0], pose[0, 0]), 3))
                for pid, pose in sorted(candidate.poses.items())
            )
            if key not in seen:
                seen.add(key)
                result.append(candidate)
        return result

    @staticmethod
    def _stop(started: float, timeout_s: float, cancel: Event) -> bool:
        return cancel.is_set() or time.monotonic() - started >= timeout_s
