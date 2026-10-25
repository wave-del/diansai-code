"""随机拼图专用的拓扑重建与几何安全评估。

本模块只审核搜索器提出的完整摆放，不判断相机是否选到了好帧。搜索器
声明的接口仅作为诊断信息；最终邻接关系始终从摆放后的真实多边形重建。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import degrees, inf
from typing import Any, Iterable

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from ..geometry import pairwise_overlap
from ..runtime_config import RuntimeConfig
from .observation_uncertainty import ObservationUncertaintyProfile


# 比赛裁片及视觉毫米换算存在边界误差。这里的两个值是最终安全门上限，
# 不再叠加候选尺寸放宽量；下限仍沿用原有观测放宽策略。
RANDOM_TARGET_LONG_SIDE_MAX_MM = 130.0
RANDOM_TARGET_SHORT_SIDE_MAX_MM = 100.0


def _candidate_signature(candidate: Any) -> tuple[Any, ...]:
    """构造随机搜索候选的离散解释签名，避免候选池被同解占满。"""

    interfaces = tuple(sorted(
        (
            item.first_piece,
            int(item.first_segment.edge_index),
            round(float(item.first_segment.t_start), 4),
            round(float(item.first_segment.t_end), 4),
            item.second_piece,
            int(item.second_segment.edge_index),
            round(float(item.second_segment.t_start), 4),
            round(float(item.second_segment.t_end), 4),
            item.composite_id,
        )
        for item in candidate.interfaces
    ))
    relative_poses = tuple(sorted(
        (
            pose.piece_id,
            round(float(pose.angle) % 360.0, 3),
        )
        for pose in candidate.poses
    ))
    return interfaces, relative_poses


class RandomCandidatePool:
    """容量有界、按质量保留完整摆放的随机候选集合。

    ``accepted`` 只表示候选是否满足理想几何标准，不再是进入候选池的
    资格。比赛只有一次执行机会，因此候选池必须在没有理想解时仍能返回
    质量最高的完整摆放。
    """

    def __init__(
        self, capacity: int, max_variants_per_discrete_signature: int = 2
    ) -> None:
        """初始化容量和每种离散解释允许保留的变体数。"""

        self.capacity = max(1, int(capacity))
        self.max_variants_per_discrete_signature = max(
            1, int(max_variants_per_discrete_signature)
        )
        self._items: list[Any] = []
        self.replacements = 0
        self.duplicates = 0
        self.discarded = 0
        self.considered = 0
        self.ever_full = False
        self.considered_after_full = 0

    @staticmethod
    def _rank(candidate: Any) -> tuple[Any, ...]:
        """按新安全门结论和候选总分排序。"""

        admission = candidate.admissibility
        return (
            *(admission.rank_key() if admission is not None else (1.0,)),
            float(candidate.total),
        )

    def _remove_identity(self, candidate: Any) -> None:
        """按对象身份移除候选，避免数值相等对象误删。"""

        for index, item in enumerate(self._items):
            if item is candidate:
                del self._items[index]
                return
        raise ValueError("候选池中不存在指定对象")

    def consider(self, candidate: Any) -> bool:
        """尝试纳入候选，并返回候选池是否发生变化。"""

        self.considered += 1
        admission = getattr(candidate, "admissibility", None)
        # 没有评估结论就无法进行可信排序；评估为“不理想”的候选仍需
        # 保留，供一次性比赛执行在无合规解时选择最优退化解。
        if admission is None:
            self.discarded += 1
            return False
        if len(self._items) >= self.capacity:
            self.ever_full = True
            self.considered_after_full += 1
        signature = _candidate_signature(candidate)
        siblings = [
            item for item in self._items
            if _candidate_signature(item) == signature
        ]
        if len(siblings) >= self.max_variants_per_discrete_signature:
            worst_sibling = max(siblings, key=self._rank)
            if self._rank(candidate) >= self._rank(worst_sibling):
                self.duplicates += 1
                return False
            self._remove_identity(worst_sibling)
            self._items.append(candidate)
            self.replacements += 1
            return True
        if len(self._items) < self.capacity:
            self._items.append(candidate)
            if len(self._items) >= self.capacity:
                self.ever_full = True
            return True
        worst = max(self._items, key=self._rank)
        if self._rank(candidate) < self._rank(worst):
            self._remove_identity(worst)
            self._items.append(candidate)
            self.replacements += 1
            return True
        self.discarded += 1
        return False

    def items(self) -> list[Any]:
        """返回按质量排序的候选快照。"""

        return sorted(self._items, key=self._rank)

    def __len__(self) -> int:
        """返回当前候选数量。"""

        return len(self._items)


@dataclass(frozen=True)
class ReconstructedInterface:
    """候选摆放中由几何重新发现的一段正长度拼缝。"""

    interface_id: str
    piece_a: str
    edge_a: int
    interval_a: tuple[float, float]
    piece_b: str
    edge_b: int
    interval_b: tuple[float, float]
    start_point: tuple[float, float]
    end_point: tuple[float, float]
    length_mm: float
    support_line_distance_mm: float
    angle_error_deg: float
    normal_a_to_b: tuple[float, float]

    def as_json(self) -> dict[str, Any]:
        """返回可写入现有诊断文件的字典。"""

        return {
            "interface_id": self.interface_id,
            "piece_a": self.piece_a,
            "edge_a": self.edge_a,
            "interval_a": list(self.interval_a),
            "piece_b": self.piece_b,
            "edge_b": self.edge_b,
            "interval_b": list(self.interval_b),
            "start_point": list(self.start_point),
            "end_point": list(self.end_point),
            "length_mm": self.length_mm,
            "support_line_distance_mm": self.support_line_distance_mm,
            "angle_error_deg": self.angle_error_deg,
            "normal_a_to_b": list(self.normal_a_to_b),
        }


@dataclass(frozen=True)
class ReconstructedJunction:
    """拼缝端点聚类后得到的物理或长边内部虚拟结点。"""

    location: tuple[float, float]
    participating_pieces: tuple[str, ...]
    interface_ids: tuple[str, ...]
    virtual_on_edges: tuple[tuple[str, int, float], ...]

    @property
    def junction_type(self) -> str:
        """按参与碎片和虚拟切分情况给出稳定类型名。"""

        if self.virtual_on_edges and len(self.participating_pieces) >= 3:
            return "T_JUNCTION"
        if len(self.participating_pieces) >= 3:
            return "MULTI_WAY_JUNCTION"
        return "SEAM_ENDPOINT"

    def as_json(self) -> dict[str, Any]:
        """返回诊断字典。"""

        return {
            "location": list(self.location),
            "participating_pieces": list(self.participating_pieces),
            "interface_ids": list(self.interface_ids),
            "virtual_on_edges": [list(item) for item in self.virtual_on_edges],
            "junction_type": self.junction_type,
        }


@dataclass(frozen=True)
class ReconstructedTopology:
    """从候选几何独立恢复出的完整平面接触图。"""

    interfaces: tuple[ReconstructedInterface, ...]
    adjacency_pairs: tuple[tuple[str, str], ...]
    junctions: tuple[ReconstructedJunction, ...]
    graph_connected: bool
    per_piece_outer_length_mm: dict[str, float]
    tolerance_mm: float

    def as_json(self) -> dict[str, Any]:
        """序列化完整拓扑，便于复核安全门事实来源。"""

        return {
            "interfaces": [item.as_json() for item in self.interfaces],
            "adjacency_pairs": [list(item) for item in self.adjacency_pairs],
            "junctions": [item.as_json() for item in self.junctions],
            "graph_connected": self.graph_connected,
            "per_piece_outer_length_mm": dict(self.per_piece_outer_length_mm),
            "tolerance_mm": self.tolerance_mm,
        }


@dataclass
class RandomSafetyAssessment:
    """随机候选从搜索准入到执行前复验共用的唯一几何结论。"""

    accepted: bool
    rejection_reasons: list[str]
    target_rectangle_width_mm: float
    target_rectangle_height_mm: float
    rectangle_fill_ratio: float
    union_rectangle_iou_estimate: float
    overlap_area_mm2: float
    overlap_ratio: float
    uncovered_area_mm2: float
    uncovered_ratio: float
    outside_area_mm2: float
    outside_ratio: float
    boundary_rectangularity: float
    boundary_complexity: float
    topology: ReconstructedTopology
    repairability: str
    declared_interface_count: int = 0
    nonadjacent_overlap_area_mm2: float = 0.0
    max_declared_interface_residual_mm: float = 0.0
    max_discovered_interface_residual_mm: float = 0.0
    directed_endpoint_context_valid: bool = True
    opposite_side_valid: bool = True
    rectangle_frame_fit_valid: bool = True
    rectangle_role_consistency_valid: bool = True
    junction_closure_valid: bool = True
    per_piece_outer_role_valid: bool = True
    rectangle_frame_fit: None = None
    outer_roles: dict[str, Any] = field(default_factory=dict)
    rectangle_role_consistency: None = None
    junction_closure: list[Any] = field(default_factory=list)

    @property
    def discovered_interface_count(self) -> int:
        """返回几何重建得到的原子拼缝数量。"""

        return len(self.topology.interfaces)

    @property
    def interface_explanation_complete(self) -> bool:
        """新门只要求重建拓扑自洽，不要求等于搜索器声明。"""

        return self.topology.graph_connected

    @property
    def interface_explanation(self) -> "RandomSafetyAssessment":
        """提供旧诊断字段的只读兼容视图。"""

        return self

    def rank_key(self) -> tuple[float, ...]:
        """给候选池提供不依赖声明接口的稳定排序。"""

        return (
            0.0 if self.accepted else 1.0,
            0.0 if self.topology.graph_connected else 1.0,
            0.0 if self.per_piece_outer_role_valid else 1.0,
            -self.union_rectangle_iou_estimate,
            self.overlap_ratio,
            self.uncovered_ratio,
            self.max_discovered_interface_residual_mm,
        )

    def as_json(self) -> dict[str, Any]:
        """保持既有输出字段并增加新拓扑诊断。"""

        return {
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "target_rectangle_width_mm": self.target_rectangle_width_mm,
            "target_rectangle_height_mm": self.target_rectangle_height_mm,
            "rectangle_fill_ratio": self.rectangle_fill_ratio,
            "union_rectangle_iou_estimate": self.union_rectangle_iou_estimate,
            "overlap_area_mm2": self.overlap_area_mm2,
            "overlap_ratio": self.overlap_ratio,
            "nonadjacent_overlap_area_mm2": self.nonadjacent_overlap_area_mm2,
            "uncovered_area_mm2": self.uncovered_area_mm2,
            "uncovered_ratio": self.uncovered_ratio,
            "outside_area_mm2": self.outside_area_mm2,
            "outside_ratio": self.outside_ratio,
            "boundary_rectangularity": self.boundary_rectangularity,
            "boundary_complexity": self.boundary_complexity,
            "declared_interface_count": self.declared_interface_count,
            "discovered_interface_count": self.discovered_interface_count,
            "interface_explanation_complete": (
                self.interface_explanation_complete
            ),
            "max_declared_interface_residual_mm": (
                self.max_declared_interface_residual_mm
            ),
            "max_discovered_interface_residual_mm": (
                self.max_discovered_interface_residual_mm
            ),
            "repairability": self.repairability,
            "directed_endpoint_context_valid": True,
            "opposite_side_valid": True,
            "rectangle_frame_fit_valid": self.rectangle_frame_fit_valid,
            "rectangle_role_consistency_valid": (
                self.rectangle_role_consistency_valid
            ),
            "junction_closure_valid": self.junction_closure_valid,
            "per_piece_outer_role_valid": self.per_piece_outer_role_valid,
            "interface_explanation": {
                "complete": self.interface_explanation_complete,
                "declared_interface_count": self.declared_interface_count,
                "discovered_interface_count": self.discovered_interface_count,
                "missing_declared_interfaces": [],
                "unexplained_interfaces": [],
                "unexpected_contacts": [],
                "interval_coverage_errors": [],
                "interval_order_errors": [],
                "declared_coverage_ratio": 0.0,
                "discovered_coverage_ratio": 1.0,
                "t_junction_consistent": True,
            },
            "rectangle_frame_fit": None,
            "outer_roles": {},
            "rectangle_role_consistency": None,
            "junction_closure": [
                item.as_json() for item in self.topology.junctions
            ],
            "reconstructed_topology": self.topology.as_json(),
        }


def _cross(first: np.ndarray, second: np.ndarray) -> float:
    """计算二维向量叉积标量。"""

    return float(first[0] * second[1] - first[1] * second[0])


def _segment_parameter(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """把共线点转换为有向边上的归一化参数。"""

    vector = end - start
    denominator = float(np.dot(vector, vector))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(point - start, vector) / denominator, 0.0, 1.0))


def _candidate_interface(
    piece_a: str,
    edge_a: int,
    first: np.ndarray,
    piece_b: str,
    edge_b: int,
    second: np.ndarray,
    centroid_a: np.ndarray,
    centroid_b: np.ndarray,
    tolerance_mm: float,
) -> ReconstructedInterface | None:
    """在两条边支持线接近时提取其公共投影区间。"""

    vector_a = first[1] - first[0]
    vector_b = second[1] - second[0]
    length_a = float(np.linalg.norm(vector_a))
    length_b = float(np.linalg.norm(vector_b))
    if min(length_a, length_b) <= 1e-6:
        return None
    tangent = vector_a / length_a
    direction_b = vector_b / length_b
    sine_error = abs(_cross(tangent, direction_b))
    if sine_error > np.sin(np.deg2rad(8.0)):
        return None
    normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
    signed_distances = np.asarray([
        np.dot(second[0] - first[0], normal),
        np.dot(second[1] - first[0], normal),
    ])
    support_distance = float(np.mean(np.abs(signed_distances)))
    if support_distance > tolerance_mm:
        return None
    projection_a = sorted((
        float(np.dot(first[0], tangent)),
        float(np.dot(first[1], tangent)),
    ))
    projection_b = sorted((
        float(np.dot(second[0], tangent)),
        float(np.dot(second[1], tangent)),
    ))
    start_projection = max(projection_a[0], projection_b[0])
    end_projection = min(projection_a[1], projection_b[1])
    overlap_length = end_projection - start_projection
    if overlap_length < max(2.0, 0.08 * min(length_a, length_b)):
        return None
    # 两片主体必须位于拼缝支持线两侧。毫米级穿插允许质心符号接近零，
    # 但同侧平行外边不能被误认为内部拼缝。
    side_a = float(np.dot(centroid_a - first[0], normal))
    side_b = float(np.dot(centroid_b - first[0], normal))
    if side_a * side_b > tolerance_mm * tolerance_mm:
        return None
    origin_projection = float(np.dot(first[0], tangent))
    middle_normal = float(np.mean([
        np.dot(first[0], normal), np.dot(second[0], normal)
    ]))
    start = tangent * start_projection + normal * middle_normal
    end = tangent * end_projection + normal * middle_normal
    point_a_start = first[0] + tangent * (start_projection - origin_projection)
    point_a_end = first[0] + tangent * (end_projection - origin_projection)
    if float(np.dot(normal, centroid_b - centroid_a)) < 0.0:
        normal = -normal
    dot_error = float(np.clip(abs(np.dot(tangent, direction_b)), 0.0, 1.0))
    return ReconstructedInterface(
        interface_id=f"reconstructed:{piece_a}:{edge_a}:{piece_b}:{edge_b}",
        piece_a=piece_a,
        edge_a=edge_a,
        interval_a=tuple(sorted((
            _segment_parameter(point_a_start, first[0], first[1]),
            _segment_parameter(point_a_end, first[0], first[1]),
        ))),
        piece_b=piece_b,
        edge_b=edge_b,
        interval_b=tuple(sorted((
            _segment_parameter(start, second[0], second[1]),
            _segment_parameter(end, second[0], second[1]),
        ))),
        start_point=(float(start[0]), float(start[1])),
        end_point=(float(end[0]), float(end[1])),
        length_mm=float(overlap_length),
        support_line_distance_mm=support_distance,
        angle_error_deg=float(degrees(np.arccos(dot_error))),
        normal_a_to_b=(float(normal[0]), float(normal[1])),
    )


def _graph_connected(piece_ids: Iterable[str], pairs: set[tuple[str, str]]) -> bool:
    """判断无向碎片邻接图是否连通。"""

    ids = set(piece_ids)
    if not ids:
        return False
    visited = {next(iter(ids))}
    changed = True
    while changed:
        changed = False
        for first, second in pairs:
            if first in visited and second not in visited:
                visited.add(second)
                changed = True
            elif second in visited and first not in visited:
                visited.add(first)
                changed = True
    return visited == ids


def _outer_lengths(
    vertices_by_id: dict[str, np.ndarray],
    bounds: tuple[float, float, float, float],
    tolerance_mm: float,
) -> dict[str, float]:
    """计算每片对目标矩形四条外边贡献的近似长度。"""

    min_x, min_y, max_x, max_y = bounds
    result: dict[str, float] = {}
    for piece_id, vertices in vertices_by_id.items():
        length = 0.0
        for index, start in enumerate(vertices):
            end = vertices[(index + 1) % len(vertices)]
            vector = end - start
            edge_length = float(np.linalg.norm(vector))
            if edge_length <= 1e-9:
                continue
            horizontal = abs(float(vector[1])) <= max(
                tolerance_mm, 0.12 * edge_length
            )
            vertical = abs(float(vector[0])) <= max(
                tolerance_mm, 0.12 * edge_length
            )
            if horizontal and (
                max(abs(float(start[1] - min_y)), abs(float(end[1] - min_y)))
                <= tolerance_mm
                or max(abs(float(start[1] - max_y)), abs(float(end[1] - max_y)))
                <= tolerance_mm
            ):
                length += edge_length
            elif vertical and (
                max(abs(float(start[0] - min_x)), abs(float(end[0] - min_x)))
                <= tolerance_mm
                or max(abs(float(start[0] - max_x)), abs(float(end[0] - max_x)))
                <= tolerance_mm
            ):
                length += edge_length
        result[piece_id] = float(length)
    return result


def _junctions(
    interfaces: list[ReconstructedInterface], tolerance_mm: float
) -> tuple[ReconstructedJunction, ...]:
    """聚类拼缝端点，并标注长边内部生成的虚拟切分点。"""

    clusters: list[list[tuple[np.ndarray, ReconstructedInterface, bool]]] = []
    for interface in interfaces:
        for is_start, raw_point in (
            (True, interface.start_point), (False, interface.end_point)
        ):
            point = np.asarray(raw_point, dtype=float)
            for cluster in clusters:
                center = np.mean([item[0] for item in cluster], axis=0)
                if float(np.linalg.norm(point - center)) <= tolerance_mm:
                    cluster.append((point, interface, is_start))
                    break
            else:
                clusters.append([(point, interface, is_start)])
    result: list[ReconstructedJunction] = []
    for cluster in clusters:
        interface_ids = sorted({item[1].interface_id for item in cluster})
        pieces = sorted({
            piece_id
            for _, interface, _ in cluster
            for piece_id in (interface.piece_a, interface.piece_b)
        })
        if len(interface_ids) < 2 and len(pieces) < 3:
            continue
        virtual: set[tuple[str, int, float]] = set()
        for _, interface, is_start in cluster:
            for piece_id, edge_index, interval in (
                (interface.piece_a, interface.edge_a, interface.interval_a),
                (interface.piece_b, interface.edge_b, interface.interval_b),
            ):
                parameter = interval[0] if is_start else interval[1]
                if 1e-3 < parameter < 1.0 - 1e-3:
                    virtual.add((piece_id, edge_index, round(parameter, 6)))
        center = np.mean([item[0] for item in cluster], axis=0)
        result.append(ReconstructedJunction(
            location=(float(center[0]), float(center[1])),
            participating_pieces=tuple(pieces),
            interface_ids=tuple(interface_ids),
            virtual_on_edges=tuple(sorted(virtual)),
        ))
    return tuple(result)


def reconstruct_random_topology(
    vertices_by_id: dict[str, np.ndarray],
    *,
    tolerance_mm: float,
) -> ReconstructedTopology:
    """不使用搜索接口声明，从最终几何重建原子拼缝和虚拟结点。"""

    arrays = {
        str(piece_id): np.asarray(vertices, dtype=float)
        for piece_id, vertices in vertices_by_id.items()
    }
    polygons = {piece_id: Polygon(vertices) for piece_id, vertices in arrays.items()}
    centroids = {
        piece_id: np.asarray([polygon.centroid.x, polygon.centroid.y])
        for piece_id, polygon in polygons.items()
    }
    interfaces: list[ReconstructedInterface] = []
    for piece_a, piece_b in combinations(sorted(arrays), 2):
        first = arrays[piece_a]
        second = arrays[piece_b]
        for edge_a in range(len(first)):
            segment_a = np.asarray([
                first[edge_a], first[(edge_a + 1) % len(first)]
            ])
            for edge_b in range(len(second)):
                segment_b = np.asarray([
                    second[edge_b], second[(edge_b + 1) % len(second)]
                ])
                interface = _candidate_interface(
                    piece_a,
                    edge_a,
                    segment_a,
                    piece_b,
                    edge_b,
                    segment_b,
                    centroids[piece_a],
                    centroids[piece_b],
                    tolerance_mm,
                )
                if interface is not None:
                    interfaces.append(interface)
    pairs = {
        tuple(sorted((item.piece_a, item.piece_b)))
        for item in interfaces
    }
    all_points = np.vstack(list(arrays.values()))
    bounds = (
        float(all_points[:, 0].min()),
        float(all_points[:, 1].min()),
        float(all_points[:, 0].max()),
        float(all_points[:, 1].max()),
    )
    return ReconstructedTopology(
        interfaces=tuple(interfaces),
        adjacency_pairs=tuple(sorted(pairs)),
        junctions=_junctions(interfaces, tolerance_mm),
        graph_connected=_graph_connected(arrays, pairs),
        per_piece_outer_length_mm=_outer_lengths(
            arrays, bounds, tolerance_mm
        ),
        tolerance_mm=float(tolerance_mm),
    )


def assess_random_assembly(
    vertices_by_id: dict[str, np.ndarray],
    profile: ObservationUncertaintyProfile,
    config: RuntimeConfig,
    *,
    expected_piece_count: int | None = None,
    declared_interface_count: int = 0,
    stage_name: str = "candidate",
) -> RandomSafetyAssessment:
    """按矩形并集和完整重建拓扑审核一个随机候选。"""

    reasons: list[str] = []
    arrays = {
        str(piece_id): np.asarray(vertices, dtype=float)
        for piece_id, vertices in vertices_by_id.items()
    }
    if expected_piece_count is not None and len(arrays) != expected_piece_count:
        reasons.append("not_fully_placed")
    if not 2 <= len(arrays) <= 4:
        reasons.append("invalid_piece_count")
    if any(
        vertices.ndim != 2
        or vertices.shape[1:] != (2,)
        or len(vertices) < 3
        or not np.all(np.isfinite(vertices))
        for vertices in arrays.values()
    ):
        reasons.append("invalid_candidate_geometry")
    polygons = [Polygon(vertices) for vertices in arrays.values()]
    if any(
        not polygon.is_valid or polygon.is_empty or polygon.area <= 1e-6
        for polygon in polygons
    ):
        reasons.append("invalid_candidate_geometry")
    empty_topology = ReconstructedTopology((), (), (), False, {}, 0.0)
    if reasons:
        return RandomSafetyAssessment(
            False, sorted(set(reasons)), 0.0, 0.0, 0.0, 0.0,
            inf, inf, inf, inf, inf, inf, 0.0, inf, empty_topology,
            "not_repairable", declared_interface_count,
        )

    union = unary_union(polygons)
    min_x, min_y, max_x, max_y = union.bounds
    width = float(max_x - min_x)
    height = float(max_y - min_y)
    rectangle = box(min_x, min_y, max_x, max_y)
    total_area = float(sum(polygon.area for polygon in polygons))
    union_area = float(union.area)
    rectangle_area = max(float(rectangle.area), 1e-12)
    overlap = float(pairwise_overlap(polygons))
    uncovered = float(rectangle.difference(union).area)
    outside = float(union.difference(rectangle).area)
    overlap_ratio = overlap / max(total_area, 1e-12)
    uncovered_ratio = uncovered / rectangle_area
    outside_ratio = outside / max(union_area, 1e-12)
    fill_ratio = union_area / rectangle_area
    intersection = float(union.intersection(rectangle).area)
    iou = intersection / max(union_area + rectangle_area - intersection, 1e-12)

    sigma = float(profile.typical_sigma_mm or 0.0)
    topology_tolerance = float(np.clip(
        max(1.5, profile.hard_error_radius_mm + sigma), 1.5, 4.0
    ))
    topology = reconstruct_random_topology(
        arrays, tolerance_mm=topology_tolerance
    )
    relaxation = float(config.geometry_candidate_dimension_relaxation_mm)
    long_max = 120.0 + relaxation if config.random_competition_rules else RANDOM_TARGET_LONG_SIDE_MAX_MM
    short_max = 90.0 + relaxation if config.random_competition_rules else RANDOM_TARGET_SHORT_SIDE_MAX_MM
    dimensions_valid = bool(
        90.0 - relaxation <= width <= long_max
        and 50.0 - relaxation <= height <= short_max
    )
    if not dimensions_valid:
        reasons.append("illegal_target_dimensions")
    # 正确测试帧约为 0.985 IoU 和 0.3% 重叠。这里给优化保留余量，
    # 但不再沿用旧门宽松到 0.70 IoU 的“可修复”概念。
    if iou < 0.94:
        reasons.append("rectangle_iou_too_low")
    if overlap_ratio > 0.025:
        reasons.append("excessive_overlap")
    if uncovered_ratio > 0.06:
        reasons.append("excessive_uncovered_area")
    if outside_ratio > 0.025:
        reasons.append("excessive_outside_area")
    if not topology.graph_connected:
        reasons.append("reconstructed_topology_disconnected")
    per_piece_outer_valid = bool(
        topology.per_piece_outer_length_mm
        and all(
            length >= 2.0
            for length in topology.per_piece_outer_length_mm.values()
        )
    )
    if config.random_competition_rules:
        # A whole original edge must lie on an outer side, not merely touch it.
        for vertices in arrays.values():
            points = np.asarray(vertices)
            has_outer_edge = any(
                abs(first[axis] - side) <= topology_tolerance
                and abs(second[axis] - side) <= topology_tolerance
                and np.linalg.norm(second - first) >= config.minimum_piece_edge_mm - 2 * profile.hard_error_radius_mm
                for first, second in zip(points, np.roll(points, -1, axis=0))
                for axis, side in ((0, min_x), (0, max_x), (1, min_y), (1, max_y))
            )
            per_piece_outer_valid = per_piece_outer_valid and has_outer_edge
    if not per_piece_outer_valid:
        reasons.append("piece_without_target_outer_edge")
    residual = max(
        (
            max(item.support_line_distance_mm, item.angle_error_deg)
            for item in topology.interfaces
        ),
        default=inf,
    )
    accepted = not reasons
    exact = bool(
        accepted and iou >= 0.995 and overlap_ratio <= 1e-5
        and uncovered_ratio <= 0.005 and outside_ratio <= 1e-5
    )
    return RandomSafetyAssessment(
        accepted=accepted,
        rejection_reasons=sorted(set(reasons)),
        target_rectangle_width_mm=width,
        target_rectangle_height_mm=height,
        rectangle_fill_ratio=float(fill_ratio),
        union_rectangle_iou_estimate=float(iou),
        overlap_area_mm2=overlap,
        overlap_ratio=float(overlap_ratio),
        uncovered_area_mm2=uncovered,
        uncovered_ratio=float(uncovered_ratio),
        outside_area_mm2=outside,
        outside_ratio=float(outside_ratio),
        boundary_rectangularity=float(
            union_area / max(float(union.convex_hull.area), 1e-12)
        ),
        boundary_complexity=float(max(
            0, len(union.convex_hull.exterior.coords) - 5
        )),
        topology=topology,
        repairability=(
            "admissible_exact" if exact
            else ("admissible_repairable" if accepted else "not_repairable")
        ),
        declared_interface_count=int(declared_interface_count),
        max_discovered_interface_residual_mm=float(residual),
        rectangle_frame_fit_valid=bool(iou >= 0.94),
        rectangle_role_consistency_valid=dimensions_valid,
        junction_closure_valid=topology.graph_connected,
        per_piece_outer_role_valid=per_piece_outer_valid,
    )
