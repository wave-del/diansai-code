"""有向接口上下文、局部两侧关系和完整候选矩形角色验证。

本模块只使用公开观测和候选放置几何。它刻意把三类证据分开：

1. pairwise 阶段的有向端点弱相容性；
2. 接口两片局部内部是否位于支撑线相反侧；
3. fully-placed 阶段的全局正交矩形、外边角色和节点闭环。

局部相反侧只是必要条件，不能替代完整候选的左右角色判断。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from ..transforms import transform_points
from .atomic_segments import AtomicSegment
from .observation_uncertainty import ObservationUncertaintyProfile
from .preprocess import ProcessedPiece
from .uncertain_geometry import UncertainPieceGeometry


ANGLE_MODEL_FLOOR_DEG = 1.0
LENGTH_MODEL_FLOOR_MM = 0.25


def _wrap_signed_degrees(value: float) -> float:
    return float((float(value) + 180.0) % 360.0 - 180.0)


def _line_angle_error_degrees(first: float, second: float) -> float:
    difference = abs(_wrap_signed_degrees(first - second))
    return float(min(difference, abs(180.0 - difference)))


def _oriented_internal_angle_deg(
    vertices: np.ndarray, vertex_index: int
) -> float:
    """返回 0～360° 内角，保留凹顶点而不是 arccos 的 0～180°。"""

    points = np.asarray(vertices, dtype=float)
    previous = points[(vertex_index - 1) % len(points)]
    current = points[vertex_index]
    following = points[(vertex_index + 1) % len(points)]
    incoming = current - previous
    outgoing = following - current
    cross = float(
        incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
    )
    dot = float(np.dot(incoming, outgoing))
    signed_turn = math.degrees(math.atan2(cross, dot))
    return float((180.0 - signed_turn) % 360.0)


def _edge_record(piece: ProcessedPiece, edge_index: int) -> dict[str, Any]:
    count = len(piece.vertices)
    for row in piece.edges:
        if int(row.get("start_vertex_index", -1)) % count == edge_index % count:
            return row
    return {}


def edge_direction_sigma_deg(
    piece: ProcessedPiece,
    edge_index: int,
    profile: ObservationUncertaintyProfile,
    *,
    model_floor_deg: float = ANGLE_MODEL_FLOOR_DEG,
    uncertain_geometry: UncertainPieceGeometry | None = None,
) -> float:
    """由公开边拟合或端点位置不确定度估计方向标准差。"""

    count = len(piece.vertices)
    index = edge_index % count
    record = _edge_record(piece, index)
    explicit = record.get("direction_std_deg")
    if explicit is not None and np.isfinite(float(explicit)):
        return float(max(model_floor_deg, float(explicit)))

    typical_position_sigma = float(
        profile.typical_sigma_mm
        if profile.typical_sigma_mm is not None
        else 1.0
    )
    uncertain = (
        uncertain_geometry
        if uncertain_geometry is not None
        else UncertainPieceGeometry.from_piece(piece, typical_position_sigma)
    )
    return float(max(
        model_floor_deg,
        float(uncertain.edge_direction_sigma_deg[index]),
    ))


def vertex_angle_sigma_deg(
    piece: ProcessedPiece,
    vertex_index: int,
    profile: ObservationUncertaintyProfile,
    *,
    model_floor_deg: float = ANGLE_MODEL_FLOOR_DEG,
    uncertain_geometry: UncertainPieceGeometry | None = None,
) -> float:
    count = len(piece.vertices)
    previous_sigma = edge_direction_sigma_deg(
        piece, (vertex_index - 1) % count, profile,
        model_floor_deg=model_floor_deg,
        uncertain_geometry=uncertain_geometry,
    )
    next_sigma = edge_direction_sigma_deg(
        piece, vertex_index % count, profile,
        model_floor_deg=model_floor_deg,
        uncertain_geometry=uncertain_geometry,
    )
    return float(math.sqrt(
        previous_sigma * previous_sigma
        + next_sigma * next_sigma
        + model_floor_deg * model_floor_deg
    ))


@dataclass(frozen=True)
class InterfaceEndpointContext:
    vertex_index: int
    internal_angle_deg: float
    angle_sigma_deg: float
    signed_turn_deg: float
    signed_turn_sigma_deg: float
    adjacent_edge_length_mm: float
    adjacent_edge_length_sigma_mm: float
    local_interior_side: int
    vertex_confidence: float
    possible_outer_corner_probability: float
    possible_outer_side_probability: float

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DirectedInterfaceContext:
    piece_id: str
    edge_index: int
    start: InterfaceEndpointContext
    end: InterfaceEndpointContext
    edge_length_mm: float
    edge_length_sigma_mm: float
    edge_direction_deg: float
    edge_direction_sigma_deg: float
    cyclic_shape_signature: tuple[tuple[float, float], ...]

    def as_json(self) -> dict[str, Any]:
        result = asdict(self)
        result["cyclic_shape_signature"] = [
            list(row) for row in self.cyclic_shape_signature
        ]
        return result


def cyclic_ordered_shape_signature(
    piece: ProcessedPiece,
) -> tuple[tuple[float, float], ...]:
    """保持轮廓方向的循环签名；调用方可循环移位，但不能反转。"""

    perimeter = max(float(np.sum(piece.edge_lengths)), 1e-12)
    angles = np.asarray(piece.edge_angles, dtype=float)
    rows: list[tuple[float, float]] = []
    for index, length in enumerate(piece.edge_lengths):
        previous = angles[(index - 1) % len(angles)]
        turn = _wrap_signed_degrees(float(angles[index] - previous))
        rows.append((round(float(length) / perimeter, 8), round(turn, 6)))
    return tuple(rows)


def cyclic_signature_distance(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
    *,
    allow_reverse: bool = False,
) -> float:
    """最小循环移位距离；默认明确禁止镜像式反转。"""

    if len(first) != len(second) or not first:
        return math.inf
    first_array = np.asarray(first, dtype=float)
    variants = [np.roll(np.asarray(second, dtype=float), shift, axis=0)
                for shift in range(len(second))]
    if allow_reverse:
        reversed_rows = np.asarray(second, dtype=float)[::-1].copy()
        # 反转轮廓时有符号转角也必须反号。
        reversed_rows[:, 1] *= -1.0
        variants.extend(
            np.roll(reversed_rows, shift, axis=0)
            for shift in range(len(second))
        )
    return float(min(np.sqrt(np.mean(np.square(first_array - row)))
                     for row in variants))


def _right_angle_probability(angle_deg: float, sigma_deg: float) -> float:
    z_score = abs(float(angle_deg) - 90.0) / max(float(sigma_deg), 1e-9)
    return float(math.exp(-0.5 * z_score * z_score))


def build_directed_interface_context(
    piece: ProcessedPiece,
    segment: AtomicSegment,
    profile: ObservationUncertaintyProfile,
    *,
    model_floor_deg: float = ANGLE_MODEL_FLOOR_DEG,
    uncertain_geometry: UncertainPieceGeometry | None = None,
) -> DirectedInterfaceContext:
    count = len(piece.vertices)
    edge_index = int(segment.edge_index) % count
    start_index = edge_index
    end_index = (edge_index + 1) % count
    direction_sigma = edge_direction_sigma_deg(
        piece,
        edge_index,
        profile,
        model_floor_deg=model_floor_deg,
        uncertain_geometry=uncertain_geometry,
    )
    typical_length_sigma = max(
        LENGTH_MODEL_FLOOR_MM,
        float(profile.typical_sigma_mm or LENGTH_MODEL_FLOOR_MM) * math.sqrt(2.0),
    )

    def endpoint(
        vertex_index: int,
        adjacent_edge_index: int,
        signed_turn: float,
    ) -> InterfaceEndpointContext:
        angle = _oriented_internal_angle_deg(
            piece.vertices, vertex_index
        )
        angle_sigma = vertex_angle_sigma_deg(
            piece,
            vertex_index,
            profile,
            model_floor_deg=model_floor_deg,
            uncertain_geometry=uncertain_geometry,
        )
        probability = _right_angle_probability(angle, angle_sigma)
        adjacent_length = float(
            piece.edge_lengths[adjacent_edge_index % count]
        )
        return InterfaceEndpointContext(
            vertex_index=vertex_index,
            internal_angle_deg=angle,
            angle_sigma_deg=angle_sigma,
            signed_turn_deg=float(signed_turn),
            signed_turn_sigma_deg=angle_sigma,
            adjacent_edge_length_mm=adjacent_length,
            # pairwise 阶段无法预知该端点最终属于哪类节点。邻边长度只作
            # 弱的有向上下文证据，模型方差必须覆盖不同合法扇区。
            adjacent_edge_length_sigma_mm=max(
                typical_length_sigma, 0.15 * adjacent_length
            ),
            # preprocess 保证视觉顺时针；在数学坐标中这是正鞋带面积，
            # 因而有向边的局部内部位于左侧。
            local_interior_side=1,
            vertex_confidence=float(piece.vertex_confidence[vertex_index]),
            possible_outer_corner_probability=probability,
            possible_outer_side_probability=float(max(0.05, probability)),
        )

    edge_angles = np.asarray(piece.edge_angles, dtype=float)
    start_turn = _wrap_signed_degrees(
        edge_angles[edge_index] - edge_angles[(edge_index - 1) % count]
    )
    end_turn = _wrap_signed_degrees(
        edge_angles[(edge_index + 1) % count] - edge_angles[edge_index]
    )
    return DirectedInterfaceContext(
        piece_id=piece.piece_id,
        edge_index=edge_index,
        start=endpoint(start_index, edge_index - 1, start_turn),
        end=endpoint(end_index, edge_index + 1, end_turn),
        edge_length_mm=float(segment.length_mm),
        edge_length_sigma_mm=typical_length_sigma,
        edge_direction_deg=float(piece.edge_angles[edge_index]),
        edge_direction_sigma_deg=direction_sigma,
        cyclic_shape_signature=cyclic_ordered_shape_signature(piece),
    )


def directed_endpoint_context_cost(
    first: DirectedInterfaceContext,
    second: DirectedInterfaceContext,
    *,
    model_floor_deg: float = ANGLE_MODEL_FLOOR_DEG,
) -> tuple[float, dict[str, float]]:
    """比较 A.start↔B.end、A.end↔B.start，不预设唯一节点类型。"""

    endpoint_pairs = ((first.start, second.end), (first.end, second.start))
    length_z: list[float] = []
    turn_z: list[float] = []
    angle_z: list[float] = []
    for left, right in endpoint_pairs:
        length_sigma = math.sqrt(
            left.adjacent_edge_length_sigma_mm ** 2
            + right.adjacent_edge_length_sigma_mm ** 2
            + LENGTH_MODEL_FLOOR_MM ** 2
        )
        length_z.append(
            abs(left.adjacent_edge_length_mm - right.adjacent_edge_length_mm)
            / max(length_sigma, 1e-9)
        )
        angle_sigma = math.sqrt(
            left.angle_sigma_deg ** 2
            + right.angle_sigma_deg ** 2
            + model_floor_deg ** 2
        )
        # pairwise 阶段仅计算多模型弱似然。最终应使用哪一个 90/180/360
        # 关系，必须等完整候选识别出节点类型后再决定。
        relation_residual = min(
            abs(left.internal_angle_deg - right.internal_angle_deg),
            abs(left.internal_angle_deg + right.internal_angle_deg - 90.0),
            abs(left.internal_angle_deg + right.internal_angle_deg - 180.0),
            abs(left.internal_angle_deg + right.internal_angle_deg - 360.0),
        )
        angle_z.append(relation_residual / max(angle_sigma, 1e-9))
        turn_residual = min(
            abs(_wrap_signed_degrees(
                left.signed_turn_deg - right.signed_turn_deg
            )),
            abs(_wrap_signed_degrees(
                left.signed_turn_deg + right.signed_turn_deg
            )),
        )
        turn_z.append(turn_residual / max(angle_sigma, 1e-9))

    components = {
        "ordered_adjacent_length_z": float(np.mean(np.minimum(length_z, 8.0))),
        "multi_model_angle_z": float(np.mean(np.minimum(angle_z, 8.0))),
        "signed_turn_z": float(np.mean(np.minimum(turn_z, 8.0))),
    }
    cost = (
        0.45 * components["ordered_adjacent_length_z"]
        + 0.30 * components["multi_model_angle_z"]
        + 0.25 * components["signed_turn_z"]
    )
    return float(cost), components


@dataclass(frozen=True)
class OppositeSideResult:
    valid: bool
    ambiguous: bool
    signed_distance_a_mm: float
    signed_distance_b_mm: float
    probability: float
    reason: str

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def _interior_probe(
    vertices: np.ndarray,
    segment: AtomicSegment,
) -> np.ndarray | None:
    points = np.asarray(vertices, dtype=float)
    start = points[segment.edge_index]
    end = points[(segment.edge_index + 1) % len(points)]
    segment_start = start + float(segment.t_start) * (end - start)
    segment_end = start + float(segment.t_end) * (end - start)
    midpoint = 0.5 * (segment_start + segment_end)
    direction = end - start
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return None
    left_normal = np.asarray([-direction[1], direction[0]]) / norm
    polygon = Polygon(points)
    base = max(1e-4, min(2.0, 0.05 * norm))
    for distance in (base, base * 0.5, base * 0.2, base * 0.05):
        for sign in (1.0, -1.0):
            probe = midpoint + sign * distance * left_normal
            point = Point(probe)
            if polygon.contains(point) and point.distance(polygon.boundary) > 1e-8:
                return probe
    representative = polygon.representative_point()
    return np.asarray([representative.x, representative.y], dtype=float)


def evaluate_opposite_side(
    first_piece: ProcessedPiece,
    first_segment: AtomicSegment,
    second_piece: ProcessedPiece,
    second_segment: AtomicSegment,
    relative_angle_deg: float,
    relative_translation: Sequence[float],
    *,
    numerical_tolerance_mm: float = 1e-7,
) -> OppositeSideResult:
    """真实计算 second→first 放置后两片局部内部的支撑线侧别。"""

    first_probe = _interior_probe(first_piece.vertices, first_segment)
    second_probe = _interior_probe(second_piece.vertices, second_segment)
    if first_probe is None or second_probe is None:
        return OppositeSideResult(
            valid=False,
            ambiguous=True,
            signed_distance_a_mm=0.0,
            signed_distance_b_mm=0.0,
            probability=0.5,
            reason="interior_probe_unavailable",
        )
    first_start = np.asarray(first_segment.start_point, dtype=float)
    first_end = np.asarray(first_segment.end_point, dtype=float)
    direction = first_end - first_start
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return OppositeSideResult(
            False, True, 0.0, 0.0, 0.5, "degenerate_support_line"
        )
    second_probe_placed = transform_points(
        np.asarray([second_probe], dtype=float),
        float(relative_angle_deg),
        np.asarray(relative_translation, dtype=float),
    )[0]

    def signed_distance(point: np.ndarray) -> float:
        delta = point - first_start
        return float(
            (direction[0] * delta[1] - direction[1] * delta[0]) / norm
        )

    first_distance = signed_distance(first_probe)
    second_distance = signed_distance(second_probe_placed)
    if (
        abs(first_distance) <= numerical_tolerance_mm
        or abs(second_distance) <= numerical_tolerance_mm
    ):
        return OppositeSideResult(
            valid=False,
            ambiguous=True,
            signed_distance_a_mm=first_distance,
            signed_distance_b_mm=second_distance,
            probability=0.5,
            reason="probe_near_support_line",
        )
    valid = first_distance * second_distance < 0.0
    return OppositeSideResult(
        valid=valid,
        ambiguous=False,
        signed_distance_a_mm=first_distance,
        signed_distance_b_mm=second_distance,
        probability=1.0 if valid else 0.0,
        reason="opposite_local_interiors" if valid else "same_side_interface",
    )


@dataclass(frozen=True)
class RectangleFrameFit:
    valid: bool
    center: tuple[float, float]
    axis_u: tuple[float, float]
    axis_v: tuple[float, float]
    width_mm: float
    height_mm: float
    orthogonality_error_deg: float
    opposite_side_parallel_error_deg: float
    support_line_rms_mm: float
    corner_closure_rms_mm: float
    exposed_edge_coverage_ratio: float
    side_coverage: dict[str, float]
    confidence: float
    rejection_reasons: tuple[str, ...]
    direction_rms_deg: float = 0.0
    normalized_cost: float = 0.0
    support_coordinates: dict[str, float] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PieceOuterRole:
    piece_id: str
    supported_sides: tuple[str, ...]
    side_coverage_mm: dict[str, float]
    role_confidence: float
    ambiguous: bool

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RectangleRoleConsistency:
    valid: bool
    parallelogram_like: bool
    left_right_parallel_error_deg: float
    top_bottom_parallel_error_deg: float
    adjacent_orthogonality_error_deg: float
    corner_closure_error_mm: float
    support_line_error_mm: float
    side_coverage_ratio: float
    per_piece_outer_role_valid: bool
    normalized_cost: float
    reasons: tuple[str, ...]

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JunctionClosureMetric:
    junction_id: str
    junction_type: str
    observed_angle_sum_deg: float
    expected_angle_sum_deg: float
    combined_sigma_deg: float
    normalized_residual: float
    cyclic_order_valid: bool
    continuation_valid: bool
    valid: bool
    ambiguous: bool
    location: tuple[float, float]

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def _weighted_line_mean_degrees(
    angles_deg: Sequence[float], weights: Sequence[float]
) -> float:
    if not angles_deg:
        return 0.0
    angles = np.deg2rad(np.asarray(angles_deg, dtype=float) * 2.0)
    rows = np.asarray(weights, dtype=float)
    return float(
        0.5 * math.degrees(math.atan2(
            float(np.sum(rows * np.sin(angles))),
            float(np.sum(rows * np.cos(angles))),
        ))
    )


def fit_rectangle_frame(
    placed_vertices: Mapping[str, np.ndarray],
    profile: ObservationUncertaintyProfile,
    *,
    excluded_piece_edges: Iterable[tuple[str, int]] = (),
    angle_model_floor_deg: float = ANGLE_MODEL_FLOOR_DEG,
    hard_rejection_z: float = 3.0,
    minimum_side_coverage_ratio: float = 0.72,
) -> tuple[
    RectangleFrameFit,
    dict[str, PieceOuterRole],
    RectangleRoleConsistency,
]:
    """对暴露外边拟合允许整体旋转、但内部严格正交的矩形框架。"""

    excluded = set(excluded_piece_edges)
    polygons = {
        piece_id: Polygon(np.asarray(vertices, dtype=float))
        for piece_id, vertices in placed_vertices.items()
    }
    union = unary_union(list(polygons.values()))
    boundary = union.boundary
    exposed: list[dict[str, Any]] = []
    for piece_id, vertices in placed_vertices.items():
        points = np.asarray(vertices, dtype=float)
        for edge_index in range(len(points)):
            if (piece_id, edge_index) in excluded:
                continue
            start = points[edge_index]
            end = points[(edge_index + 1) % len(points)]
            line = LineString([start, end])
            exposed_length = float(line.intersection(boundary).length)
            if exposed_length <= 1e-5:
                continue
            delta = end - start
            length = float(np.linalg.norm(delta))
            if length <= 1e-9:
                continue
            exposed.append({
                "piece_id": piece_id,
                "edge_index": edge_index,
                "start": start,
                "end": end,
                "midpoint": 0.5 * (start + end),
                "length": min(length, exposed_length),
                "angle": math.degrees(math.atan2(delta[1], delta[0])),
            })
    if not exposed:
        empty = RectangleFrameFit(
            False, (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
            0.0, 0.0, math.inf, math.inf, math.inf, math.inf, 0.0,
            {"TOP": 0.0, "BOTTOM": 0.0, "LEFT": 0.0, "RIGHT": 0.0},
            0.0, ("no_exposed_edges",), math.inf, math.inf, {},
        )
        consistency = RectangleRoleConsistency(
            False, False, math.inf, math.inf, math.inf, math.inf,
            math.inf, 0.0, False, math.inf, ("no_exposed_edges",),
        )
        return empty, {}, consistency

    candidate_angles = sorted({
        round(float(row["angle"]) % 90.0, 8) for row in exposed
    })
    best_theta = 0.0
    best_cost = math.inf
    for theta in candidate_angles:
        residuals = [
            min(
                _line_angle_error_degrees(row["angle"], theta),
                _line_angle_error_degrees(row["angle"], theta + 90.0),
            )
            for row in exposed
        ]
        cost = float(np.average(
            np.square(np.minimum(residuals, 30.0)),
            weights=[row["length"] for row in exposed],
        ))
        if cost < best_cost:
            best_cost = cost
            best_theta = theta
    radians = math.radians(best_theta)
    axis_u = np.asarray([math.cos(radians), math.sin(radians)], dtype=float)
    axis_v = np.asarray([-axis_u[1], axis_u[0]], dtype=float)
    all_points = np.vstack([
        np.asarray(vertices, dtype=float)
        for vertices in placed_vertices.values()
    ])
    projections_u = all_points @ axis_u
    projections_v = all_points @ axis_v
    min_u, max_u = float(projections_u.min()), float(projections_u.max())
    min_v, max_v = float(projections_v.min()), float(projections_v.max())
    center_vector = (
        0.5 * (min_u + max_u) * axis_u
        + 0.5 * (min_v + max_v) * axis_v
    )
    support_tolerance = max(
        0.25,
        2.5 * float(profile.typical_sigma_mm or 0.20),
    )
    direction_sigma = max(
        angle_model_floor_deg,
        math.degrees(math.atan2(
            math.sqrt(2.0) * float(profile.typical_sigma_mm or 0.35),
            20.0,
        )),
    )
    if not profile.absolute_measurement_sigma_known:
        # 没有公开 covariance/direction_std 时不能把 1° 模型下限冒充
        # 已知测量精度。这里仅扩大“角色残差是否明确”的统计尺度；
        # 不修改多边形、IoU、重叠或赛题门槛。
        direction_sigma = max(direction_sigma, 3.0)
        support_tolerance = max(support_tolerance, 2.0)
    side_coverage = {
        "TOP": 0.0, "BOTTOM": 0.0, "LEFT": 0.0, "RIGHT": 0.0,
    }
    per_piece_coverage: dict[str, dict[str, float]] = {
        piece_id: dict(side_coverage) for piece_id in placed_vertices
    }
    direction_residuals: list[float] = []
    support_residuals: list[float] = []
    weights: list[float] = []
    group_angles: dict[str, list[float]] = {"u": [], "v": []}
    group_weights: dict[str, list[float]] = {"u": [], "v": []}
    accepted_length = 0.0
    total_length = float(sum(row["length"] for row in exposed))
    for row in exposed:
        error_u = _line_angle_error_degrees(row["angle"], best_theta)
        error_v = _line_angle_error_degrees(row["angle"], best_theta + 90.0)
        midpoint = np.asarray(row["midpoint"], dtype=float)
        if error_u <= error_v:
            direction_error = error_u
            coordinate = float(midpoint @ axis_v)
            side = "TOP" if abs(coordinate - min_v) <= abs(coordinate - max_v) else "BOTTOM"
            support_error = min(abs(coordinate - min_v), abs(coordinate - max_v))
            group = "u"
        else:
            direction_error = error_v
            coordinate = float(midpoint @ axis_u)
            side = "LEFT" if abs(coordinate - min_u) <= abs(coordinate - max_u) else "RIGHT"
            support_error = min(abs(coordinate - min_u), abs(coordinate - max_u))
            group = "v"
        direction_residuals.append(direction_error)
        support_residuals.append(support_error)
        weights.append(row["length"])
        group_angles[group].append(row["angle"])
        group_weights[group].append(row["length"])
        if (
            direction_error <= hard_rejection_z * direction_sigma
            and support_error <= hard_rejection_z * support_tolerance
        ):
            side_coverage[side] += row["length"]
            per_piece_coverage[row["piece_id"]][side] += row["length"]
            accepted_length += row["length"]

    direction_rms = float(math.sqrt(np.average(
        np.square(direction_residuals), weights=weights
    )))
    support_rms = float(math.sqrt(np.average(
        np.square(support_residuals), weights=weights
    )))
    mean_u = _weighted_line_mean_degrees(
        group_angles["u"], group_weights["u"]
    )
    mean_v = _weighted_line_mean_degrees(
        group_angles["v"], group_weights["v"]
    )
    adjacent_orthogonality = abs(
        90.0 - abs(_wrap_signed_degrees(mean_v - mean_u))
    )
    adjacent_orthogonality = min(
        adjacent_orthogonality, abs(180.0 - adjacent_orthogonality)
    )

    def parallel_spread(group: str) -> float:
        if len(group_angles[group]) <= 1:
            return 0.0
        mean = _weighted_line_mean_degrees(
            group_angles[group], group_weights[group]
        )
        return float(max(
            _line_angle_error_degrees(value, mean)
            for value in group_angles[group]
        ))

    u_spread = parallel_spread("u")
    v_spread = parallel_spread("v")
    coverage_ratio = accepted_length / max(total_length, 1e-12)
    piece_roles: dict[str, PieceOuterRole] = {}
    minimum_piece_coverage = max(1.0, 2.0 * support_tolerance)
    for piece_id, coverage in per_piece_coverage.items():
        supported = tuple(
            side for side in ("TOP", "BOTTOM", "LEFT", "RIGHT")
            if coverage[side] > minimum_piece_coverage
        )
        total = sum(coverage.values())
        confidence = float(min(1.0, total / max(minimum_piece_coverage, 1e-9)))
        piece_roles[piece_id] = PieceOuterRole(
            piece_id, supported, coverage, confidence, not bool(supported)
        )
    per_piece_valid = all(
        not role.ambiguous for role in piece_roles.values()
    )
    direction_z = direction_rms / max(direction_sigma, 1e-9)
    support_z = support_rms / max(support_tolerance, 1e-9)
    orthogonality_z = adjacent_orthogonality / max(direction_sigma, 1e-9)
    normalized_cost = float(max(direction_z, support_z, orthogonality_z))
    reasons: list[str] = []
    if direction_z > hard_rejection_z:
        reasons.append("rectangle_axis_inconsistent")
    if orthogonality_z > hard_rejection_z:
        reasons.append("parallelogram_layout")
    if coverage_ratio < minimum_side_coverage_ratio:
        reasons.append("outer_side_role_mismatch")
    if not per_piece_valid:
        reasons.append("piece_without_outer_side_role")
    all_sides_present = all(
        side_coverage[side] > minimum_piece_coverage
        for side in side_coverage
    )
    if not all_sides_present:
        reasons.append("incomplete_rectangle_side_coverage")
    parallelogram_like = bool(
        orthogonality_z > hard_rejection_z
        and max(u_spread, v_spread) <= hard_rejection_z * direction_sigma
    )
    valid = not reasons
    confidence = float(math.exp(-0.5 * min(normalized_cost, 8.0) ** 2))
    frame = RectangleFrameFit(
        valid=valid,
        center=(float(center_vector[0]), float(center_vector[1])),
        axis_u=(float(axis_u[0]), float(axis_u[1])),
        axis_v=(float(axis_v[0]), float(axis_v[1])),
        width_mm=float(max_u - min_u),
        height_mm=float(max_v - min_v),
        orthogonality_error_deg=float(adjacent_orthogonality),
        opposite_side_parallel_error_deg=float(max(u_spread, v_spread)),
        support_line_rms_mm=support_rms,
        corner_closure_rms_mm=support_rms,
        exposed_edge_coverage_ratio=float(coverage_ratio),
        side_coverage=side_coverage,
        confidence=confidence,
        rejection_reasons=tuple(reasons),
        direction_rms_deg=direction_rms,
        normalized_cost=normalized_cost,
        support_coordinates={
            "LEFT": min_u, "RIGHT": max_u,
            "TOP": min_v, "BOTTOM": max_v,
        },
    )
    consistency = RectangleRoleConsistency(
        valid=valid,
        parallelogram_like=parallelogram_like,
        left_right_parallel_error_deg=float(v_spread),
        top_bottom_parallel_error_deg=float(u_spread),
        adjacent_orthogonality_error_deg=float(adjacent_orthogonality),
        corner_closure_error_mm=support_rms,
        support_line_error_mm=support_rms,
        side_coverage_ratio=float(coverage_ratio),
        per_piece_outer_role_valid=per_piece_valid,
        normalized_cost=normalized_cost,
        reasons=tuple(reasons),
    )
    return frame, piece_roles, consistency


def evaluate_junction_closure(
    placed_vertices: Mapping[str, np.ndarray],
    frame: RectangleFrameFit,
    profile: ObservationUncertaintyProfile,
    *,
    source_pieces: Mapping[str, ProcessedPiece] | None = None,
    cluster_tolerance_mm: float = 1e-3,
    angle_model_floor_deg: float = ANGLE_MODEL_FLOOR_DEG,
    hard_rejection_z: float = 4.0,
) -> list[JunctionClosureMetric]:
    """在完整放置状态中按 90/180/360 度语义复核节点闭环。"""

    records: list[tuple[str, int, np.ndarray]] = []
    for piece_id, vertices in placed_vertices.items():
        for vertex_index, point in enumerate(np.asarray(vertices, dtype=float)):
            records.append((piece_id, vertex_index, point))
    groups: list[list[tuple[str, int, np.ndarray]]] = []
    for record in records:
        for group in groups:
            if np.linalg.norm(record[2] - group[0][2]) <= cluster_tolerance_mm:
                group.append(record)
                break
        else:
            groups.append([record])

    axis_u = np.asarray(frame.axis_u, dtype=float)
    axis_v = np.asarray(frame.axis_v, dtype=float)
    supports = frame.support_coordinates
    support_tolerance = max(
        0.25, 2.5 * float(profile.typical_sigma_mm or 0.20)
    )
    metrics: list[JunctionClosureMetric] = []
    for group_index, group in enumerate(groups):
        location = np.mean([row[2] for row in group], axis=0)
        sectors: dict[str, tuple[float, float]] = {}
        for piece_id, vertex_index, _point in group:
            vertices = np.asarray(placed_vertices[piece_id], dtype=float)
            angle = _oriented_internal_angle_deg(vertices, vertex_index)
            source = (source_pieces or {}).get(piece_id)
            sigma = (
                vertex_angle_sigma_deg(
                    source, vertex_index, profile,
                    model_floor_deg=angle_model_floor_deg,
                )
                if source is not None
                else math.sqrt(3.0) * angle_model_floor_deg
            )
            sectors[piece_id] = (float(angle), float(sigma))
        # T 节点中长边在交点处没有物理顶点，需要补一个 180° 扇区。
        for piece_id, vertices in placed_vertices.items():
            if piece_id in sectors:
                continue
            points = np.asarray(vertices, dtype=float)
            for edge_index in range(len(points)):
                line = LineString([
                    points[edge_index], points[(edge_index + 1) % len(points)]
                ])
                point = Point(location)
                if point.distance(line) > cluster_tolerance_mm:
                    continue
                parameter = float(
                    np.dot(
                        location - points[edge_index],
                        points[(edge_index + 1) % len(points)] - points[edge_index],
                    )
                    / max(line.length * line.length, 1e-12)
                )
                if 1e-5 < parameter < 1.0 - 1e-5:
                    sectors[piece_id] = (
                        180.0, math.sqrt(2.0) * angle_model_floor_deg
                    )
                    break
        if len(sectors) < 2:
            continue
        projection_u = float(location @ axis_u)
        projection_v = float(location @ axis_v)
        on_u_side = min(
            abs(projection_u - supports.get("LEFT", projection_u)),
            abs(projection_u - supports.get("RIGHT", projection_u)),
        ) <= support_tolerance
        on_v_side = min(
            abs(projection_v - supports.get("TOP", projection_v)),
            abs(projection_v - supports.get("BOTTOM", projection_v)),
        ) <= support_tolerance
        if on_u_side and on_v_side:
            junction_type, expected = "RECTANGLE_CORNER", 90.0
        elif on_u_side or on_v_side:
            junction_type, expected = "OUTER_SIDE", 180.0
        else:
            junction_type, expected = "INTERNAL", 360.0
        observed = float(sum(row[0] for row in sectors.values()))
        combined_sigma = math.sqrt(
            sum(row[1] * row[1] for row in sectors.values())
            + angle_model_floor_deg * angle_model_floor_deg
        )
        normalized = abs(observed - expected) / max(combined_sigma, 1e-9)
        continuation = not (
            junction_type == "INTERNAL"
            and any(abs(row[0] - 180.0) <= 1e-6 for row in sectors.values())
            and len(sectors) < 3
        )
        # 正面积重叠由 admission 另行检查；在无重叠前提下，唯一 piece
        # sector 加上角度闭环足以排除重复循环占位。
        cyclic_valid = len(sectors) == len(set(sectors))
        valid = (
            normalized <= hard_rejection_z
            and cyclic_valid
            and continuation
        )
        metrics.append(JunctionClosureMetric(
            junction_id=f"junction_{group_index:05d}",
            junction_type=junction_type,
            observed_angle_sum_deg=observed,
            expected_angle_sum_deg=expected,
            combined_sigma_deg=float(combined_sigma),
            normalized_residual=float(normalized),
            cyclic_order_valid=cyclic_valid,
            continuation_valid=continuation,
            valid=valid,
            ambiguous=not valid and normalized <= hard_rejection_z * 1.5,
            location=(float(location[0]), float(location[1])),
        ))
    return metrics


def physically_equivalent_permutation(
    first_layout: Mapping[str, np.ndarray],
    second_layout: Mapping[str, np.ndarray],
    *,
    tolerance_mm: float = 1e-5,
) -> bool:
    """忽略白片 ID 后比较最终占据区域的多重集合与完整 union。"""

    if len(first_layout) != len(second_layout):
        return False
    first_polygons = [Polygon(np.asarray(row, dtype=float))
                      for row in first_layout.values()]
    second_polygons = [Polygon(np.asarray(row, dtype=float))
                       for row in second_layout.values()]
    first_union = unary_union(first_polygons)
    second_union = unary_union(second_polygons)
    if first_union.symmetric_difference(second_union).area > tolerance_mm ** 2:
        return False
    unmatched = list(second_polygons)
    for polygon in first_polygons:
        candidates = [
            (index, polygon.symmetric_difference(other).area)
            for index, other in enumerate(unmatched)
        ]
        if not candidates:
            return False
        index, difference = min(candidates, key=lambda row: row[1])
        if difference > tolerance_mm ** 2:
            return False
        del unmatched[index]
    return not unmatched
