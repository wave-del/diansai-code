"""遮挡鲁棒的蓝纸粗定位。

完整四边形失败时回退到“边界直线拟合”：
- 对蓝 mask 最大连通域的轮廓做分段 + 方向/共线聚类，得到若干虚拟边；
- 用“矩形结构先验”（两两平行 + 正交）筛选出 3~4 条真实边；
- 4 条边 → 相邻边求交得 4 角；3 条边 → 仿射补全第 4 边（遮挡边与对边平行）；
- 每个角用角点邻域蓝像素覆盖率判定“实测/估计”，供 Resize 放宽校验。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import cv2
import numpy as np

try:
    from . import _paths
except ImportError:
    import _paths
_paths.ensure()

import test_a4_detection as a4d  # 复用 HSV 参数、_order_corners、PARAM_PRESETS

# 检测在 640×360 上完成，与 a4d._blue_a4_seed 一致
DETECT_W, DETECT_H = 640, 360
# 现场蓝纸实测尺寸（不是标准 A4），用于期望边长估算
A4_X_MM = 206.5
A4_Y_MM = 292.0

_MIN_FALLBACK_AREA_RATIO = 0.15  # 遮挡后蓝 mask 最小面积占比
_POLY_EPSILON_RATIO = 0.025      # 轮廓粗分段系数
_MIN_SEGMENT_LEN = 12.0          # 分段最小长度，px
_CLUSTER_ANGLE_TOL_DEG = 8.0     # 同一条边的分段角度容差
_CLUSTER_LINE_TOL_PX = 10.0      # 同一条边的分段线距容差
_MIN_EDGE_SPAN_RATIO = 0.35      # 虚拟边相对期望短边的最小跨度
_MAX_EDGES = 6                   # 参与组合搜索的边数上限
_MAX_EDGE_RESIDUAL_PX = 1.6      # 虚拟边直线拟合最大中位残差（拒绝手部等不规则边界）
_STRUCTURE_PARALLEL_TOL_DEG = 10.0
_STRUCTURE_ORTHOGONAL_TOL_DEG = 15.0
_CORNER_PATCH_RADIUS = 12        # 角点内侧蓝覆盖率检查半径，px
_CORNER_PATCH_MIN_COVERAGE = 0.5
_CORNER_PATCH_DILATE = 7         # 补偿 MORPH_CLOSE 对角的腐蚀
_CORNER_INWARD_OFFSET = 14       # 内侧采样点沿角平分线内移距离，px
_CORNER_OUTWARD_SAMPLES = (8, 14, 20)  # 角点外侧沿两条边方向的采样距离
_CORNER_OUTWARD_MAX_BLUE = 0.35  # 外侧蓝占比高于此值 → 角点未到纸边
_WORLD_ASPECT_RATIO = A4_Y_MM / A4_X_MM  # 292/206.5 ≈ 1.414
_ASPECT_TOLERANCE = 0.04         # 组合四边形校正长宽比相对世界值允许偏差
                                 # （干净帧实测偏差 ~0.4%，遮挡边组合 ~5.6%）


@dataclass
class EdgeLine:
    """一条拟合出的矩形边直线。"""

    direction: np.ndarray  # 单位方向 (2,)
    point: np.ndarray      # 直线上一点 (2,)
    span: float            # 沿方向的实际跨度，px
    pts: np.ndarray        # 参与拟合的轮廓点 (N,2)


@dataclass
class SeedResult:
    """蓝纸 seed 结果：四角 + 逐角可靠性 + 诊断信息。"""

    corners: np.ndarray                    # 4×2 源图像素坐标，TL,TR,BR,BL
    reliability: tuple[bool, bool, bool, bool]  # True=实测角，False=估计角
    mask: np.ndarray                       # 640×360 蓝 mask（诊断用）
    area_ratio: float                      # 蓝 mask 面积占比
    mode: str                              # "full" | "partial4" | "partial3"
    estimated_gap: float                   # 最大角点缺口，px（诊断用）
    aspect_error: float = 0.0              # 四边形校正长宽比相对世界值偏差


def _fit_segments(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """对蓝 mask 最大连通域轮廓做分段，返回 (方向, 线上点, 轮廓点集) 列表。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    pts = contour.reshape(-1, 2).astype(np.float32)
    perimeter = float(cv2.arcLength(contour, True))
    poly = cv2.approxPolyDP(
        contour, _POLY_EPSILON_RATIO * perimeter, True
    ).reshape(-1, 2).astype(np.float32)
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    count = len(poly)
    for index in range(count):
        start, end = poly[index], poly[(index + 1) % count]
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length < _MIN_SEGMENT_LEN:
            continue
        unit = vector / length
        normal = np.asarray([-unit[1], unit[0]])
        relative = pts - start
        projection = relative @ unit
        distance = np.abs(relative @ normal)
        member = (
            (projection >= -0.05 * length)
            & (projection <= 1.05 * length)
            & (distance <= max(4.0, 0.15 * length))
        )
        if int(np.count_nonzero(member)) < 12:
            continue
        member_points = pts[member]
        vx, vy, x0, y0 = cv2.fitLine(
            member_points, cv2.DIST_HUBER, 0, 0.01, 0.01
        ).reshape(4)
        direction = np.asarray([vx, vy], dtype=np.float64)
        direction /= np.linalg.norm(direction) + 1e-12
        point = np.asarray([x0, y0], dtype=np.float64)
        for _ in range(2):  # 残余修剪，剔除遮挡物边界杂点
            residuals = np.abs((member_points - point) @ np.asarray([-direction[1], direction[0]]))
            median = float(np.median(residuals))
            mad = float(np.median(np.abs(residuals - median)))
            keep = residuals <= max(1.25, median + 3.0 * 1.4826 * mad)
            if int(np.count_nonzero(keep)) < 12:
                break
            member_points = member_points[keep]
        segments.append((direction, point, member_points))
    return segments


def _same_edge(cluster_points: np.ndarray, candidate: tuple[np.ndarray, np.ndarray, np.ndarray]) -> bool:
    """候选分段是否与簇内点集属于同一条边（方向 + 线距）。"""
    direction, point, _ = candidate
    vx, vy, x0, y0 = cv2.fitLine(
        cluster_points, cv2.DIST_HUBER, 0, 0.01, 0.01
    ).reshape(4)
    fitted = np.asarray([vx, vy], dtype=np.float64)
    fitted /= np.linalg.norm(fitted) + 1e-12
    fitted_point = np.asarray([x0, y0], dtype=np.float64)
    angle_diff = abs(
        math.degrees(math.atan2(fitted[1], fitted[0]))
        - math.degrees(math.atan2(direction[1], direction[0]))
    )
    angle_diff = min(angle_diff % 180.0, 180.0 - angle_diff % 180.0)
    if angle_diff > _CLUSTER_ANGLE_TOL_DEG:
        return False
    distance = np.abs((point - fitted_point) @ np.asarray([-fitted[1], fitted[0]]))
    return float(distance) <= _CLUSTER_LINE_TOL_PX


def _cluster_segments(segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> list[EdgeLine]:
    """按方向 + 共线性把分段合并成虚拟边。"""
    used = [False] * len(segments)
    edges: list[EdgeLine] = []
    for index in range(len(segments)):
        if used[index]:
            continue
        members: list[np.ndarray] = [segments[index][2]]
        used[index] = True
        changed = True
        while changed:
            changed = False
            cluster_points = np.vstack(members)
            for other in range(len(segments)):
                if used[other] or not _same_edge(cluster_points, segments[other]):
                    continue
                members.append(segments[other][2])
                used[other] = True
                changed = True
                cluster_points = np.vstack(members)
        cluster_points = np.vstack(members)
        vx, vy, x0, y0 = cv2.fitLine(
            cluster_points, cv2.DIST_HUBER, 0, 0.01, 0.01
        ).reshape(4)
        direction = np.asarray([vx, vy], dtype=np.float64)
        direction /= np.linalg.norm(direction) + 1e-12
        point = np.asarray([x0, y0], dtype=np.float64)
        projection = (cluster_points - point) @ direction
        residuals = np.abs(
            (cluster_points - point) @ np.asarray([-direction[1], direction[0]])
        )
        median_residual = float(np.median(residuals))
        if median_residual > _MAX_EDGE_RESIDUAL_PX:
            # 手部/不规则遮挡物的边界直线拟合残差高，不能当作纸边
            continue
        edges.append(
            EdgeLine(
                direction=direction,
                point=point,
                span=float(np.ptp(projection)),
                pts=cluster_points,
            )
        )
    return edges


def _angle_deg(direction: np.ndarray) -> float:
    return math.degrees(math.atan2(direction[1], direction[0]))


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    """Scalar planar cross product, including NumPy versions without 2D cross."""
    return float(first[0] * second[1] - first[1] * second[0])


def _orient_edges(edges: list[EdgeLine], center: np.ndarray) -> list[EdgeLine]:
    """统一方向使多边形中心位于有向边左侧（便于按角度排序）。"""
    oriented: list[EdgeLine] = []
    for edge in edges:
        direction = edge.direction
        if _cross_2d(direction, center - edge.point) < 0.0:
            direction = -direction
        oriented.append(EdgeLine(direction, edge.point, edge.span, edge.pts))
    return oriented


def _line_intersection(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
) -> np.ndarray | None:
    """两条参数直线求交；近平行时返回 None。"""
    point_a, direction_a = first
    point_b, direction_b = second
    matrix = np.column_stack((direction_a, -direction_b))
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 0.08:
        return None
    parameters = np.linalg.solve(matrix, point_b - point_a)
    return point_a + parameters[0] * direction_a


def _edge_line(edge: EdgeLine) -> tuple[np.ndarray, np.ndarray]:
    return edge.point, edge.direction


def _structure_valid(edges: list[EdgeLine]) -> bool:
    """矩形结构先验：4 边时两对近平行且对间近正交；3 边时有一对平行 + 第三边正交。"""
    angles = [_angle_deg(edge.direction) for edge in edges]

    def diff(a: float, b: float) -> float:
        return min((a - b) % 180.0, 180.0 - (a - b) % 180.0)

    if len(edges) == 4:
        pairs = [(i, j, diff(angles[i], angles[j])) for i in range(4) for j in range(i + 1, 4)]
        pairs.sort(key=lambda item: item[2])
        i, j, parallel = pairs[0]
        if parallel > _STRUCTURE_PARALLEL_TOL_DEG:
            return False
        rest = [k for k in range(4) if k not in (i, j)]
        other_parallel = diff(angles[rest[0]], angles[rest[1]])
        if other_parallel > _STRUCTURE_PARALLEL_TOL_DEG:
            return False
        cross = diff(angles[i], angles[rest[0]])
        return abs(cross - 90.0) <= _STRUCTURE_ORTHOGONAL_TOL_DEG
    for i in range(3):
        for j in range(i + 1, 3):
            if diff(angles[i], angles[j]) > _STRUCTURE_PARALLEL_TOL_DEG:
                continue
            k = 3 - i - j
            cross = diff(angles[i], angles[k])
            if abs(cross - 90.0) <= _STRUCTURE_ORTHOGONAL_TOL_DEG:
                return True
    return False


def _quad_aspect_error_from_corners(corners: np.ndarray) -> float | None:
    """四边形校正后长宽比相对世界值的偏差；非法四边形返回 None。

    俯视相机下四边形图像长宽比 ≈ 世界长宽比（实测偏差 <1%）。
    遮挡物边界冒充纸边时（如黑条盖住整边），四边形明显变窄，
    长宽比偏差可到 5% 以上，据此区分真实纸边组合与假边组合。
    """
    if not np.all(np.isfinite(corners)):
        return None
    if not cv2.isContourConvex(np.rint(corners).astype(np.int32).reshape(-1, 1, 2)):
        return None
    sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    if float(np.min(sides)) < 5.0:
        return None
    pair_a = float(np.mean((sides[0], sides[2])))
    pair_b = float(np.mean((sides[1], sides[3])))
    if pair_a < 1e-6 or pair_b < 1e-6:
        return None
    aspect = max(pair_a / pair_b, pair_b / pair_a)
    return abs(aspect - _WORLD_ASPECT_RATIO) / _WORLD_ASPECT_RATIO


def _quad_aspect_error_from_selection(selection: list[EdgeLine]) -> float | None:
    """4 边组合的四边形宽高比偏差；3 边组合不适用（补全自带宽高比约束）。"""
    if len(selection) != 4:
        return None
    try:
        corners = _corners_from_four(selection)
    except ValueError:
        return None
    return _quad_aspect_error_from_corners(corners)


_COMPLETION_MIN_CONTAINMENT = 0.95  # 补全四边形必须包含绝大多数蓝 mask
_COMPLETION_MAX_AREA_RATIO = 1.5    # 补全四边形面积相对 mask 面积上限
_COMPLETION_BORDER_MARGIN = 20.0    # 补全角允许越出 mask 边界的最大像素


def _completion_score(
    selection: list[EdgeLine],
    mask: np.ndarray,
) -> tuple[float, float, float] | None:
    """3 边组合补全结果的合理性评分。

    返回 (mask 包含率, 四边形面积/mask 面积, 实测边无支撑外延比例)。
    外延比例 = 实测边在四边形中的长度相对其 mask 拟合跨度的超出部分；
    正确组合的外延只等于遮挡深度，错误组合（把遮挡边界当纸边）外延更大。
    """
    try:
        corners, _ = _complete_three(selection, mask)
    except ValueError:
        return None
    height, width = mask.shape[:2]
    if (
        np.any(corners[:, 0] < -_COMPLETION_BORDER_MARGIN)
        or np.any(corners[:, 0] > width - 1.0 + _COMPLETION_BORDER_MARGIN)
        or np.any(corners[:, 1] < -_COMPLETION_BORDER_MARGIN)
        or np.any(corners[:, 1] > height - 1.0 + _COMPLETION_BORDER_MARGIN)
    ):
        return None
    dilated = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    quad_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(
        quad_mask,
        [np.rint(corners).astype(np.int32).reshape(-1, 1, 2)],
        255,
    )
    total = int(cv2.countNonZero(dilated))
    if total == 0:
        return None
    containment = float(
        cv2.countNonZero(cv2.bitwise_and(dilated, quad_mask))
    ) / total
    area_ratio = float(cv2.countNonZero(quad_mask)) / total
    # 补全角点 [A, B, C, D] 的三条实测边（A→B, B→C, C→D），
    # 与 selection 中的边按方向匹配后计算无支撑外延比例
    quad_edge_vectors = [
        corners[1] - corners[0],
        corners[2] - corners[1],
        corners[3] - corners[2],
    ]
    unsupported = 0.0
    for edge in selection:
        best_ratio = 1.0
        for vector in quad_edge_vectors:
            length = float(np.linalg.norm(vector))
            if length < 1e-6:
                continue
            direction = vector / length
            if abs(_cross_2d(edge.direction, direction)) > 0.45:
                continue  # 方向差 > ~27°，不是同一条边
            ratio = abs(length - edge.span) / length
            best_ratio = min(best_ratio, ratio)
        unsupported = max(unsupported, best_ratio)
    return containment, area_ratio, unsupported


def _select_rectangle_edges(
    edges: list[EdgeLine],
    short_side_expected: float,
    mask: np.ndarray,
) -> tuple[list[EdgeLine] | None, str]:
    """按长度过滤后组合搜索满足矩形结构的 4 边或 3 边组合。

    - 4 边组合：矩形结构 + 校正宽高比接近世界值；
    - 3 边组合：矩形结构 + 补全四边形包含大部分蓝 mask 且面积不过大；
    - 优先 4 边（实测角），无有效 4 边组合时才用 3 边补全。
    """
    strong = [edge for edge in edges if edge.span >= _MIN_EDGE_SPAN_RATIO * short_side_expected]
    strong.sort(key=lambda edge: -edge.span)
    strong = strong[:_MAX_EDGES]
    best_four: tuple[float, list[EdgeLine]] | None = None
    for combo in itertools.combinations(range(len(strong)), 4):
        selection = [strong[index] for index in combo]
        if not _structure_valid(selection):
            continue
        error = _quad_aspect_error_from_selection(selection)
        if error is not None and error <= _ASPECT_TOLERANCE:
            if best_four is None or error < best_four[0]:
                best_four = (error, selection)
    if best_four is not None:
        return best_four[1], "four"
    best_three: tuple[tuple[float, float], list[EdgeLine]] | None = None
    for combo in itertools.combinations(range(len(strong)), 3):
        selection = [strong[index] for index in combo]
        if not _structure_valid(selection):
            continue
        score = _completion_score(selection, mask)
        if (
            score is None
            or score[0] < _COMPLETION_MIN_CONTAINMENT
            or score[1] > _COMPLETION_MAX_AREA_RATIO
        ):
            continue
        # 正确补全 = 蓝 mask 的最紧外包络：面积最小者优先（外延比作次级）
        key = (score[1], score[2])
        if best_three is None or key < best_three[0]:
            best_three = (key, selection)
    if best_three is not None:
        return best_three[1], "three"
    return None, ""


def _corners_from_four(edges: list[EdgeLine]) -> np.ndarray:
    """4 条定向边按角度排序后相邻求交。"""
    oriented = _orient_edges(edges, np.mean([edge.point for edge in edges], axis=0))
    oriented.sort(key=lambda edge: _angle_deg(edge.direction))
    corners: list[np.ndarray] = []
    for index in range(4):
        intersection = _line_intersection(
            _edge_line(oriented[index - 1]),
            _edge_line(oriented[index]),
        )
        if intersection is None:
            raise ValueError("相邻矩形边近平行，无法求交")
        corners.append(intersection)
    return np.asarray(corners, dtype=np.float64)


def _complete_three(
    edges: list[EdgeLine],
    mask: np.ndarray,
) -> tuple[np.ndarray, tuple[bool, bool, bool, bool]]:
    """3 条边用“平行对 + 已知宽高比”补全第 4 边。

    3 条已知边 = 世界平行对 (P1, P2) + 横边 E（缺边与 E 平行）。
    仿射模型下缺边位置有一个自由参数（纸宽），由已知宽高比约束：
    平行边长度 = 宽高比 × |E|。补全角 A = B ± u·d、D = C ± u·d，
    其中 B = P1∩E、C = P2∩E、u ∥ P1、d = 宽高比 × |C−B|。
    u 的符号取“补全四边形包含更多蓝 mask”者。
    """
    parallel_pair = min(
        itertools.combinations(edges, 2),
        key=lambda pair: min(
            abs(_angle_deg(pair[0].direction) - _angle_deg(pair[1].direction)) % 180.0,
            180.0 - abs(_angle_deg(pair[0].direction) - _angle_deg(pair[1].direction)) % 180.0,
        ),
    )
    first, second = parallel_pair
    cross = next(edge for edge in edges if edge is not first and edge is not second)
    b = _line_intersection(_edge_line(first), _edge_line(cross))
    c = _line_intersection(_edge_line(second), _edge_line(cross))
    if b is None or c is None:
        raise ValueError("3 边补全时实测角近平行")
    cross_length = float(np.linalg.norm(c - b))
    if cross_length < 5.0:
        raise ValueError("3 边补全时横边过短")
    parallel_direction = first.direction
    if _cross_2d(parallel_direction, c - b) < 0.0:
        parallel_direction = -parallel_direction
    d_offset = _WORLD_ASPECT_RATIO * cross_length

    best: tuple[float, np.ndarray] | None = None
    for sign in (1.0, -1.0):
        offset = parallel_direction * (sign * d_offset)
        corners = np.asarray(
            [b + offset, b, c, c + offset], dtype=np.float64
        )
        quad_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(
            quad_mask,
            [np.rint(corners).astype(np.int32).reshape(-1, 1, 2)],
            255,
        )
        total = int(cv2.countNonZero(mask))
        containment = (
            float(cv2.countNonZero(cv2.bitwise_and(mask, quad_mask))) / total
            if total else 0.0
        )
        if best is None or containment > best[0]:
            best = (containment, corners)
    if best is None or best[0] < 0.5:
        raise ValueError("3 边补全结果不包含蓝 mask，疑似遮挡过重")
    corners = best[1]
    reliability = (False, True, True, False)
    return corners, reliability


def _corner_reliability(
    mask: np.ndarray,
    corners: np.ndarray,
) -> tuple[bool, bool, bool, bool]:
    """判定角点是否真实落在纸角上。

    真实纸角同时满足：
    1) 角点内侧（沿角平分线内移）邻域基本是蓝纸；
    2) 角点外侧沿两条边方向延伸处基本不是蓝纸（纸边已到尽头）。
    遮挡物角点/假角（落在遮挡物边缘上的多边形角）至少破坏其一：
    - 遮挡角：内侧被遮挡物占据 → 条件 1 失败；
    - 落在遮挡物边上的假角：外侧延伸仍落在纸上 → 条件 2 失败。
    """
    dilated = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_CORNER_PATCH_DILATE, _CORNER_PATCH_DILATE)),
    )
    height, width = mask.shape[:2]
    center = np.mean(corners, axis=0)
    reliability: list[bool] = []
    for index, corner in enumerate(corners):
        # 条件 1：角平分线内侧采样（方向从角点指向四边形中心）
        bisector = center - corner
        bisector /= np.linalg.norm(bisector) + 1e-12
        inward = corner + bisector * _CORNER_INWARD_OFFSET
        x, y = int(round(inward[0])), int(round(inward[1]))
        x0, x1 = max(0, x - _CORNER_PATCH_RADIUS), min(width, x + _CORNER_PATCH_RADIUS + 1)
        y0, y1 = max(0, y - _CORNER_PATCH_RADIUS), min(height, y + _CORNER_PATCH_RADIUS + 1)
        patch = dilated[y0:y1, x0:x1]
        inward_blue = (
            float((patch > 0).mean()) >= _CORNER_PATCH_MIN_COVERAGE
            if patch.size else False
        )
        # 条件 2：沿两条边方向向角点外侧延伸采样
        outward_blue_ratio = 0.0
        samples: list[tuple[float, float]] = []
        for neighbor in (corners[(index - 1) % 4], corners[(index + 1) % 4]):
            edge_dir = corner - neighbor
            edge_dir /= np.linalg.norm(edge_dir) + 1e-12
            for distance in _CORNER_OUTWARD_SAMPLES:
                point = corner + edge_dir * distance
                samples.append((float(point[0]), float(point[1])))
        if samples:
            xs = np.asarray([point[0] for point in samples], dtype=np.float32)
            ys = np.asarray([point[1] for point in samples], dtype=np.float32)
            xi = np.clip(np.rint(xs).astype(int), 0, width - 1)
            yi = np.clip(np.rint(ys).astype(int), 0, height - 1)
            outward_blue_ratio = float((dilated[yi, xi] > 0).mean())
        outward_dark = outward_blue_ratio <= _CORNER_OUTWARD_MAX_BLUE
        reliability.append(bool(inward_blue and outward_dark))
    return tuple(reliability)


def _order_with_reliability(
    corners: np.ndarray,
    reliability: tuple[bool, bool, bool, bool],
) -> tuple[np.ndarray, tuple[bool, bool, bool, bool]]:
    """按 a4d._order_corners 同规则排序角点并同步重排可靠性。"""
    center = np.mean(corners, axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    order = np.argsort(angles)
    ordered = corners[order]
    ordered_rel = tuple(reliability[index] for index in order)
    if cv2.contourArea(ordered.reshape(-1, 1, 2).astype(np.float32), oriented=True) < 0:
        ordered = ordered[::-1]
        ordered_rel = tuple(reversed(ordered_rel))
    roll = -int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, roll, axis=0), tuple(np.roll(list(ordered_rel), roll))


def _build_mask(frame: np.ndarray, args: object) -> np.ndarray:
    """构造 640×360 蓝 mask（与 a4d._blue_a4_seed 相同的阈值与形态学）。"""
    image = cv2.resize(frame, (DETECT_W, DETECT_H), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.asarray((args.hue_min, args.sat_min, args.value_min), dtype=np.uint8)
    upper = np.asarray((args.hue_max, 255, 255), dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )


def _seed_from_mask(
    mask: np.ndarray,
    source_width: int,
    source_height: int,
) -> SeedResult | None:
    """回退路径：边界直线拟合 + 矩形结构/宽高比筛选 + 3 边补全。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area_ratio = float(cv2.contourArea(contour)) / float(DETECT_W * DETECT_H)
    if area_ratio < _MIN_FALLBACK_AREA_RATIO:
        return None
    segments = _fit_segments(mask)
    if not segments:
        return None
    edges = _cluster_segments(segments)
    short_side_expected = math.sqrt(
        area_ratio * DETECT_W * DETECT_H / _WORLD_ASPECT_RATIO
    )
    selection, mode = _select_rectangle_edges(edges, short_side_expected, mask)
    if selection is None:
        return None
    if mode == "four":
        corners = _corners_from_four(selection)
        reliability = (True, True, True, True)
    else:
        corners, reliability = _complete_three(selection, mask)
    reliability = _corner_reliability(mask, corners)
    scale = np.asarray(
        [(source_width - 1) / (DETECT_W - 1), (source_height - 1) / (DETECT_H - 1)],
        dtype=np.float64,
    )
    corners, reliability = _order_with_reliability(corners * scale, reliability)
    aspect_error = _quad_aspect_error_from_corners(corners) or 1.0
    return SeedResult(
        corners.astype(np.float32),
        reliability,
        mask,
        area_ratio,
        f"partial{mode[-1]}",
        0.0,
        aspect_error=aspect_error,
    )


def _seed_from_chroma(frame: np.ndarray) -> SeedResult | None:
    """Recover illumination-split HSV regions using the existing Resize blue criterion.

    This is a measured four-corner path, not a three-side extrapolation. Reuse
    Resize's absolute cyan chroma floor and existing fast-path contour criteria.
    """
    from ec_26_e.vision.resize.resizer import _paper_chroma_image, _MIN_PAPER_CHROMA

    small = cv2.resize(frame, (DETECT_W, DETECT_H), interpolation=cv2.INTER_AREA)
    mask = (_paper_chroma_image(small) >= _MIN_PAPER_CHROMA).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area_ratio = float(cv2.contourArea(contour)) / (DETECT_W * DETECT_H)
    if area_ratio < 0.25:  # existing formal minimum, not a fitted threshold
        return None
    polygon = cv2.approxPolyDP(contour, .015 * cv2.arcLength(contour, True), True).reshape(-1, 2)
    if len(polygon) != 4 or not cv2.isContourConvex(polygon.reshape(-1, 1, 2)):
        return None
    if (np.any(polygon[:, 0] <= 1) or np.any(polygon[:, 0] >= DETECT_W-2)
            or np.any(polygon[:, 1] <= 1) or np.any(polygon[:, 1] >= DETECT_H-2)):
        return None
    ordered = a4d._order_corners(polygon.astype(np.float32))
    height, width = frame.shape[:2]
    scale = np.asarray(((width-1)/(DETECT_W-1), (height-1)/(DETECT_H-1)))
    corners = ordered * scale
    error = _quad_aspect_error_from_corners(corners)
    return SeedResult(corners.astype(np.float32), _corner_reliability(mask, ordered), mask,
                      area_ratio, "chroma-four", 0.0, 1.0 if error is None else error)


def seed_a4_robust(frame: np.ndarray, args: object) -> SeedResult:
    """遮挡鲁棒蓝纸定位；返回 4 角（源图像素）与逐角可靠性。"""
    source_height, source_width = frame.shape[:2]
    scale_to_detect = np.asarray(
        [(DETECT_W - 1) / (source_width - 1), (DETECT_H - 1) / (source_height - 1)],
        dtype=np.float64,
    )
    # 快速路径：完整四边形（与现有 _blue_a4_seed 完全一致）
    try:
        corners, area_ratio, mask = a4d._blue_a4_seed(frame, args)
        reliability = _corner_reliability(mask, corners * scale_to_detect)
        full_result = SeedResult(
            corners,
            reliability,
            mask,
            area_ratio,
            "full",
            0.0,
            aspect_error=_quad_aspect_error_from_corners(corners) or 1.0,
        )
    except ValueError:
        full_result = None
        mask = _build_mask(frame, args)
    # 回退路径：边拟合 + 结构/宽高比筛选。full 成功时也运行——
    # 遮挡物可使完整多边形“合法但扭曲”（如黑条盖边时假边导致
    # 角偏差 40px），此时按宽高比偏差选更接近真实纸形的结果。
    fallback = _seed_from_mask(mask, source_width, source_height)
    # ResizeConfig requires in-frame reference corners. Do not pass an
    # extrapolated out-of-frame quad into a configuration error downstream.
    candidates = [item for item in (full_result, fallback) if item is not None
                  and np.all(np.isfinite(item.corners))
                  and np.all(item.corners >= 0)
                  and np.all(item.corners <= np.asarray((source_width-1, source_height-1)))]
    if candidates:
        return min(candidates, key=lambda item: item.aspect_error)
    chroma = _seed_from_chroma(frame)
    if chroma is not None:
        return chroma
    if full_result is not None or fallback is not None:
        raise ValueError("蓝纸补全角越出画面；缺少完整可见边界，拒绝透视校正")
    raise ValueError("蓝纸边界未能拟合出有效矩形；请检查蓝纸颜色、完整可见性或 HSV 参数")
