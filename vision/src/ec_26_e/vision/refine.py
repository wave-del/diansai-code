"""在 YOLO 实例局部搜索区内定位纸面可见边并稳健拟合多边形。"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ec_26_e.vision._blue import normalized_blue_score_u8
from ec_26_e.vision.contracts import (
    InstanceDiagnostics,
    PieceDetection,
    SegInstance,
    VisionConfig,
)
from ec_26_e.vision.foreground import prepare_foreground, _workspace_mask
from ec_26_e.vision.geometry import (
    canonicalize_corners,
    is_simple_polygon,
    polygon_corner_depths,
    polygon_edge_lengths,
    polygon_iou,
    polygon_turn_angles_deg,
    relative_point,
)


def _mm_per_pixel(width: int, height: int) -> float:
    """返回横向 A4 图两个方向毫米像素比的均值。"""

    return 0.5 * (292.0 / max(width - 1, 1) + 207.0 / max(height - 1, 1))


def _largest_core_component(mask: np.ndarray, core: np.ndarray) -> np.ndarray:
    """仅保留与 YOLO 核心交叠最多的连通部分。"""

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask)
    best_label = max(
        range(1, count),
        key=lambda label: (
            int(np.count_nonzero((labels == label) & (core > 0))),
            int(stats[label, cv2.CC_STAT_AREA]),
        ),
    )
    selected = np.zeros_like(mask)
    selected[labels == best_label] = 255
    return selected


def _recover_local_foreground(
    image_bgr: np.ndarray,
    coarse: np.ndarray,
    core: np.ndarray,
    search: np.ndarray,
) -> np.ndarray:
    """以 YOLO 只作前景种子和搜索限制，用 GrabCut 恢复局部颜色边界。"""

    x, y, width, height = cv2.boundingRect(search)
    if width <= 0 or height <= 0:
        return np.zeros_like(coarse)
    slices = (slice(y, y + height), slice(x, x + width))
    local_search = search[slices]
    local_coarse = coarse[slices]
    local_core = core[slices]
    local_image = image_bgr[slices]
    grab_mask = np.full(local_coarse.shape, cv2.GC_BGD, dtype=np.uint8)
    grab_mask[local_search > 0] = cv2.GC_PR_BGD
    grab_mask[local_coarse > 0] = cv2.GC_PR_FGD
    grab_mask[local_core > 0] = cv2.GC_FGD
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        # GrabCut initializes its colour mixtures by random sampling. A fixed
        # image and instance seed must yield the same physical boundary on
        # every frame, independent of preceding frames' initialization.
        cv2.setRNGSeed(0)
        cv2.grabCut(
            local_image,
            grab_mask,
            None,
            background_model,
            foreground_model,
            2,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return np.zeros_like(coarse)
    local_recovered = np.where(
        (grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    local_recovered = cv2.bitwise_and(local_recovered, local_search)
    recovered = np.zeros_like(coarse)
    recovered[slices] = local_recovered
    return _largest_core_component(recovered, core)


def _choose_polygon(
    contour: np.ndarray,
    config: VisionConfig,
    mm_px: float,
) -> tuple[np.ndarray | None, int]:
    """选择证据充分的 3～5 边简单多边形，并清理低显著度杂边。"""

    perimeter = cv2.arcLength(contour, True)
    best: np.ndarray | None = None
    best_score = -math.inf
    candidates: dict[tuple[int, ...], tuple[np.ndarray, int, float]] = {}
    for epsilon in np.linspace(0.004, 0.065, 50):
        points = cv2.approxPolyDP(
            contour,
            float(epsilon * perimeter),
            True,
        ).reshape(-1, 2)
        if not config.min_vertices <= len(points) <= config.max_vertices + 2:
            continue
        points = canonicalize_corners(points).astype(np.float64)
        removed_short_edges = 0
        while len(points) > config.min_vertices:
            turns = np.abs(polygon_turn_angles_deg(points))
            depths_mm = polygon_corner_depths(points) * mm_px
            weak = np.flatnonzero(
                (turns < config.minimum_corner_turn_deg)
                | (depths_mm < config.minimum_corner_depth_mm)
            )
            if len(weak):
                weakness = np.minimum(
                    turns[weak] / config.minimum_corner_turn_deg,
                    depths_mm[weak] / config.minimum_corner_depth_mm,
                )
                # Several small bevel/rounding vertices may be weak at once.
                # Removing the shallowest first can discard a necessary end of
                # the bevel, leaving a triangle below the existing IoU gate.
                # Preserve the most observed contour before using salience as
                # a tie-break; no vertex-count or physical threshold is relaxed.
                remove = max(range(len(weak)), key=lambda slot: (
                    polygon_iou(contour, np.delete(points, int(weak[slot]), axis=0)),
                    -float(weakness[slot]),
                ))
                points = np.delete(points, int(weak[remove]), axis=0)
                points = canonicalize_corners(points).astype(np.float64)
                continue
            lengths_mm = polygon_edge_lengths(points) * mm_px
            shortest = int(np.argmin(lengths_mm))
            if float(lengths_mm[shortest]) > config.short_edge_merge_mm:
                break
            endpoints = np.asarray(
                [shortest, (shortest + 1) % len(points)],
                dtype=np.int64,
            )
            salience = (
                polygon_corner_depths(points)[endpoints] * mm_px
                * np.abs(polygon_turn_angles_deg(points)[endpoints])
            )
            points = np.delete(points, int(endpoints[np.argmin(salience)]), axis=0)
            points = canonicalize_corners(points).astype(np.float64)
            removed_short_edges += 1
        lengths_mm = polygon_edge_lengths(points) * mm_px
        turns = np.abs(polygon_turn_angles_deg(points))
        depths_mm = polygon_corner_depths(points) * mm_px
        if (
            not config.min_vertices <= len(points) <= config.max_vertices
            or not is_simple_polygon(points)
            or float(np.min(lengths_mm)) <= config.minimum_physical_edge_mm
            or float(np.min(turns)) < config.minimum_corner_turn_deg
            or float(np.min(depths_mm)) < config.minimum_corner_depth_mm
        ):
            continue
        iou = polygon_iou(contour, points)
        if iou < config.minimum_polygon_iou:
            continue
        key = tuple(np.rint(points).astype(np.int32).reshape(-1).tolist())
        candidates[key] = (points, removed_short_edges, iou)
    best_removed_short_edges = 0
    for points, removed_short_edges, iou in candidates.values():
        score = (
            iou - config.extra_vertex_iou_gain * len(points)
        )
        if score > best_score:
            best_score = score
            best = points
            best_removed_short_edges = removed_short_edges
    if best is None:
        return None, 0
    return canonicalize_corners(best).astype(np.float64), best_removed_short_edges


def _polygon_geometry_is_acceptable(
    polygon: np.ndarray,
    config: VisionConfig,
    mm_px: float,
) -> bool:
    """检查精修或稳定前轮廓满足简单性及全部物理角边约束。"""

    points = canonicalize_corners(np.asarray(polygon, dtype=np.float32))
    return bool(
        config.min_vertices <= len(points) <= config.max_vertices
        and is_simple_polygon(points)
        and float(np.min(polygon_edge_lengths(points)) * mm_px)
        > config.minimum_physical_edge_mm
        and float(np.min(np.abs(polygon_turn_angles_deg(points))))
        >= config.minimum_corner_turn_deg
        and float(np.min(polygon_corner_depths(points)) * mm_px)
        >= config.minimum_corner_depth_mm
    )


def _retain_feasible_refinement(
    initial: np.ndarray,
    fitted: np.ndarray,
    contour: np.ndarray,
    config: VisionConfig,
    mm_px: float,
) -> np.ndarray:
    """Retain supported progress when unconstrained intersections break geometry.

    Returning the entire initial polygon discards useful fits on its other
    edges as soon as one bevel becomes too short. Backtrack towards that known
    feasible polygon, retaining a step only if all existing physical constraints
    hold and its observed-contour IoU is no worse. Vertex count is unchanged.
    Eight halvings are a bounded line search, not a physical acceptance gate.
    """
    initial=np.asarray(initial,np.float32)
    fitted=np.asarray(fitted,np.float32)
    if initial.shape!=fitted.shape or not np.isfinite(fitted).all():
        return initial
    baseline_iou=polygon_iou(contour,initial)
    for step in range(1,9):
        candidate=initial+(fitted-initial)*(0.5**step)
        if (_polygon_geometry_is_acceptable(candidate,config,mm_px)
                and polygon_iou(contour,candidate)>=max(baseline_iou,config.minimum_polygon_iou)):
            return canonicalize_corners(candidate).astype(np.float32)
    return initial


def _sample_profile(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """用双线性插值采样浮点坐标的单通道图。"""

    map_x = points[:, 0].astype(np.float32).reshape(1, -1)
    map_y = points[:, 1].astype(np.float32).reshape(1, -1)
    return cv2.remap(
        image.astype(np.float32), map_x, map_y,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    ).reshape(-1)


def _visible_edge_points(
    blue_score: np.ndarray,
    initial: np.ndarray,
    search_px: int,
) -> list[np.ndarray]:
    """沿粗边外法线搜索由纸面到蓝底的首次连续稳定跃迁。"""

    edges: list[np.ndarray] = []
    distances = np.arange(-search_px, search_px + 1, dtype=np.float64)
    for index, start in enumerate(initial):
        end = initial[(index + 1) % len(initial)]
        vector = end - start
        length = float(np.linalg.norm(vector))
        direction = vector / max(length, 1e-9)
        outward = np.array([direction[1], -direction[0]], dtype=np.float64)
        sample_count = max(24, int(length / 3.0))
        fractions = np.linspace(0.12, 0.88, sample_count)
        bases = start[None, :] + fractions[:, None] * vector[None, :]
        profile_points = bases[:, None, :] + distances[None, :, None] * outward[None, None, :]
        profiles = cv2.remap(
            blue_score.astype(np.float32),
            profile_points[..., 0].astype(np.float32),
            profile_points[..., 1].astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        profiles = cv2.GaussianBlur(profiles, (3, 1), 0)
        gradients = np.gradient(profiles, axis=1)
        # 用累计和一次算出全部内外局部均值，避免逐采样点的 Python 循环。
        slots = np.arange(2, len(distances) - 2)
        cumulative = np.pad(np.cumsum(profiles, axis=1), ((0, 0), (1, 0)))
        inside_start = np.maximum(0, slots - 4)
        inside = (cumulative[:, slots] - cumulative[:, inside_start]) / (slots - inside_start)
        outside_end = np.minimum(len(distances), slots + 5)
        outside = (cumulative[:, outside_end] - cumulative[:, slots + 1]) / (outside_end - slots - 1)
        valid = (outside - inside >= 3.0) & (gradients[:, slots] > 0.0)
        has_edge = np.any(valid, axis=1)
        # 在粗前景边界内侧约 1.5 px 优先寻找最强跃迁：足以避开阴影外缘，
        # 又不会把纸面渐变或印刷纹理当作更早的“边”。
        quality = gradients[:, slots] - 0.8 * np.abs(distances[slots] + 1.5)[None, :]
        local_peak = np.argmax(np.where(valid, quality, -np.inf), axis=1)
        peaks = slots[local_peak]
        rows = np.arange(len(bases))
        denominator = (
            gradients[rows, peaks - 1]
            - 2.0 * gradients[rows, peaks]
            + gradients[rows, peaks + 1]
        )
        delta = np.zeros(len(bases), dtype=np.float64)
        usable = np.abs(denominator) > 1e-6
        delta[usable] = np.clip(
            0.5
            * (gradients[rows[usable], peaks[usable] - 1] - gradients[rows[usable], peaks[usable] + 1])
            / denominator[usable],
            -0.5,
            0.5,
        )
        locations = bases[has_edge] + (
            distances[peaks[has_edge]] + delta[has_edge]
        )[:, None] * outward[None, :]
        edges.append(locations.astype(np.float64))
    return edges


def _foreground_contour_edge_points(
    contour: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    """从传统前景轮廓中选取某条边主体的支持点。"""

    points = contour.reshape(-1, 2).astype(np.float64)
    vector = end - start
    length = float(np.linalg.norm(vector))
    direction = vector / max(length, 1e-9)
    relative = points - start
    projection = relative @ direction
    normal_distance = np.abs(
        relative[:, 0] * direction[1] - relative[:, 1] * direction[0]
    )
    band = max(1.5, min(4.0, 0.035 * length))
    return points[
        (projection >= 0.08 * length)
        & (projection <= 0.92 * length)
        & (normal_distance <= band)
    ]


def _fit_robust_line(points: np.ndarray) -> tuple[tuple[float, float, float], float, float]:
    """用 RANSAC/MAD 去异常后执行 Huber/TLS 加权直线拟合。"""

    if len(points) < 6:
        raise ValueError("真实边支持点不足")
    # 固定随机种子使相同输入可复现；RANSAC 只负责剔除明显离群点，
    # 最终参数仍由后续 Huber/TLS 使用全部合格支持点估计。
    generator = np.random.default_rng(0)
    best_inliers = np.ones(len(points), dtype=bool)
    pair_count = min(12, len(points))
    pairs = generator.integers(0, len(points), size=(pair_count, 2))
    pairs[:, 1] = (pairs[:, 1] + (pairs[:, 1] == pairs[:, 0])) % len(points)
    directions = points[pairs[:, 1]] - points[pairs[:, 0]]
    lengths = np.linalg.norm(directions, axis=1)
    usable_pairs = lengths >= 1.0
    directions /= np.maximum(lengths[:, None], 1e-9)
    normals = np.column_stack((directions[:, 1], -directions[:, 0]))
    all_residuals = np.abs(
        np.einsum(
            "pni,pi->pn",
            points[None, :, :] - points[pairs[:, 0], None, :],
            normals,
        )
    )
    all_inliers = all_residuals <= 1.5
    counts = np.count_nonzero(all_inliers, axis=1)
    counts[~usable_pairs] = 0
    mean_residuals = np.sum(all_residuals * all_inliers, axis=1) / np.maximum(counts, 1)
    maximum_count = int(np.max(counts))
    if maximum_count >= 6:
        choices = np.flatnonzero(counts == maximum_count)
        selected_pair = int(choices[np.argmin(mean_residuals[choices])])
        best_inliers = all_inliers[selected_pair]
    ransac_points = points[best_inliers] if np.count_nonzero(best_inliers) >= 6 else points
    vx, vy, x0, y0 = cv2.fitLine(ransac_points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(4)
    direction = np.array([float(vx), float(vy)])
    normal = np.array([direction[1], -direction[0]])
    offset = -float(normal @ np.array([x0, y0]))
    residuals = ransac_points @ normal + offset
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    limit = max(1.0, 2.5 * 1.4826 * mad)
    selected = ransac_points[np.abs(residuals - median) <= limit]
    if len(selected) < 6:
        selected = ransac_points
    center = np.median(selected, axis=0)
    centered = selected - center
    initial_dist = np.abs(centered @ normal)
    huber = max(0.75, 1.5 * float(np.median(initial_dist)))
    weights = np.minimum(1.0, huber / np.maximum(initial_dist, 1e-6))
    covariance = (centered * weights[:, None]).T @ centered / max(float(np.sum(weights)), 1e-9)
    values, vectors = np.linalg.eigh(covariance)
    fitted_normal = vectors[:, int(np.argmin(values))]
    if fitted_normal @ normal < 0:
        fitted_normal *= -1.0
    fitted_offset = -float(fitted_normal @ center)
    final_residuals = np.abs(selected @ fitted_normal + fitted_offset)
    angles = np.arctan2(
        selected[:, 1] - center[1], selected[:, 0] - center[0]
    )
    return (
        (float(fitted_normal[0]), float(fitted_normal[1]), fitted_offset),
        float(np.mean(final_residuals)),
        float(np.degrees(np.std(np.unwrap(angles)))) / max(len(selected), 1),
    )


def _intersection(first: tuple[float, float, float], second: tuple[float, float, float]) -> np.ndarray | None:
    """求相邻法线式直线交点，近平行时返回空。"""

    matrix = np.asarray([first[:2], second[:2]], dtype=np.float64)
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 0.05:
        return None
    return np.linalg.solve(matrix, -np.asarray([first[2], second[2]], dtype=np.float64))


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    """计算两个二值掩膜的 IoU。"""

    intersection = int(np.count_nonzero((left > 0) & (right > 0)))
    union = int(np.count_nonzero((left > 0) | (right > 0)))
    return intersection / union if union else 0.0


def _consistent_foreground_component(
    foreground: np.ndarray,
    coarse: np.ndarray,
    core: np.ndarray,
    minimum_iou: float,
) -> np.ndarray | None:
    """Keep an observed component when it agrees with the instance seed.

    A shrunken model mask can put a real corner outside the local search band.
    Cutting that component creates an artificial edge before polygon fitting.
    Agreement is measured over the full component, so attached desk/background
    pixels count against it. This is an alternative to clipping, not permission
    to widen the 3/4 mm edge searches or skip the later geometry/IoU checks.
    """
    component = _largest_core_component(foreground, core)
    if _mask_iou(coarse, component) < minimum_iou:
        return None
    return component


def _foreground_connected_to_workspace_edge(
    foreground: np.ndarray,
    core: np.ndarray,
) -> bool:
    """Detect an interior instance joined to the workspace boundary by foreground.

    Global colour thresholding can classify a washed-out part of the blue board
    as foreground. Area alone cannot detect this: its intersection with the
    instance search band is large but contains artificial clipping edges.
    Use the existing workspace boundary, without changing colour thresholds.
    """
    height, width = foreground.shape
    workspace = _workspace_mask((width, height))
    boundary = cv2.subtract(workspace, cv2.erode(workspace, np.ones((3, 3), np.uint8)))
    if np.any((core > 0) & (boundary > 0)):
        return False
    component = _largest_core_component(foreground, core)
    return bool(np.any((component > 0) & (boundary > 0)))


def refine_instance(
    image_bgr: np.ndarray,
    instance: SegInstance,
    group_id: int,
    config: VisionConfig,
    foreground: np.ndarray | None = None,
    blue_score: np.ndarray | None = None,
) -> tuple[PieceDetection | None, InstanceDiagnostics]:
    """在单个 YOLO 实例搜索区内精修，并返回现有 PieceDetection 契约。"""

    height, width = image_bgr.shape[:2]
    mm_px = _mm_per_pixel(width, height)
    coarse = np.where(instance.mask > 0, 255, 0).astype(np.uint8)
    diagnostic = InstanceDiagnostics(instance.score, coarse.copy(), np.zeros_like(coarse))
    if cv2.countNonZero(coarse) < 20:
        diagnostic.failure = "YOLO mask 为空或过小"
        return None, diagnostic
    core_px = max(1, int(round(1.0 / mm_px)))
    core = cv2.erode(coarse, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * core_px + 1,) * 2))
    if cv2.countNonZero(core) == 0:
        core = coarse
    search_px = max(1, int(round(config.dilation_mm / mm_px)))
    search = cv2.dilate(coarse, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * search_px + 1,) * 2))

    if foreground is None:
        foreground, _, _ = prepare_foreground(image_bgr)
    candidate = cv2.bitwise_and(foreground, search)
    refined = _largest_core_component(candidate, core)
    boundary_source = "global_foreground"
    component = _consistent_foreground_component(
        foreground, coarse, core, config.minimum_polygon_iou,
    )
    if component is not None and not np.array_equal(component, refined):
        refined = component
        boundary_source = "consistent_foreground_component"
    contaminated = component is None and _foreground_connected_to_workspace_edge(foreground, core)
    fallback = (contaminated or
                cv2.countNonZero(refined) < max(20, int(0.15 * cv2.countNonZero(coarse))))
    if fallback:
        # 回退仍以 YOLO 实例为唯一候选，只在其 4 mm 局部区域恢复轮廓。
        fallback_px = max(search_px, int(round(config.fallback_dilation_mm / mm_px)))
        fallback_search = cv2.dilate(coarse, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * fallback_px + 1,) * 2))
        if contaminated:
            # Expanding the same polluted global mask cannot supply a true
            # boundary. Recover local colour evidence using the existing 4 mm
            # GrabCut fallback instead of waiting for an area failure.
            refined = _recover_local_foreground(image_bgr, coarse, core, fallback_search)
            boundary_source = "local_grabcut"
        else:
            refined = _largest_core_component(cv2.bitwise_and(foreground, fallback_search), core)
            boundary_source = "expanded_foreground"
        if not contaminated and cv2.countNonZero(refined) < max(20, int(0.15 * cv2.countNonZero(coarse))):
            refined = _recover_local_foreground(
                image_bgr,
                coarse,
                core,
                fallback_search,
            )
            boundary_source = "local_grabcut"
    diagnostic.refined_mask = refined.copy()
    diagnostic.fallback_used = fallback
    diagnostic.boundary_source = boundary_source
    diagnostic.coarse_refined_iou = _mask_iou(coarse, refined)

    if cv2.countNonZero(refined) < 20:
        diagnostic.failure = "传统视觉未在 YOLO 搜索区内恢复有效前景"
        return None, diagnostic

    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        diagnostic.failure = "无法形成前景轮廓"
        return None, diagnostic
    contour = max(contours, key=cv2.contourArea)
    initial, merged_short_edges = _choose_polygon(contour, config, mm_px)
    diagnostic.merged_below_minimum_edges = merged_short_edges
    if initial is None:
        diagnostic.failure = (
            "无法形成满足边长、角点显著度和 IoU 限制的 3～5 边简单多边形"
        )
        return None, diagnostic

    if blue_score is None:
        blue_score = normalized_blue_score_u8(cv2.GaussianBlur(image_bgr, (5, 5), 0))
    edge_points = _visible_edge_points(blue_score, initial, max(search_px, 3))
    expanded_edge_points: list[np.ndarray] | None = None
    lines: list[tuple[float, float, float]] = []
    residuals: list[float] = []
    direction_spreads: list[float] = []
    edge_sources: list[str] = []
    support = 0
    expected = 0
    for index, points in enumerate(edge_points):
        expected += max(24, int(np.linalg.norm(initial[(index + 1) % len(initial)] - initial[index]) / 3.0))
        source = "visible_edge_3mm"
        if len(points) < 6:
            if expanded_edge_points is None:
                fallback_edge_px = max(
                    search_px,
                    int(round(config.fallback_dilation_mm / mm_px)),
                )
                expanded_edge_points = _visible_edge_points(
                    blue_score,
                    initial,
                    fallback_edge_px,
                )
            points = expanded_edge_points[index]
            source = "visible_edge_4mm"
            diagnostic.fallback_used = True
        if len(points) < 6 and boundary_source != "local_grabcut":
            points = _foreground_contour_edge_points(
                contour,
                initial[index],
                initial[(index + 1) % len(initial)],
            )
            source = "foreground_contour"
            diagnostic.fallback_used = True
        if len(points) < 6:
            diagnostic.failure = f"第 {index} 条边缺少真实图像边缘支持"
            return None, diagnostic
        support += len(points)
        try:
            line, residual, spread = _fit_robust_line(points)
        except ValueError:
            diagnostic.failure = f"第 {index} 条边无法由真实像素稳健拟合"
            return None, diagnostic
        lines.append(line)
        residuals.append(residual)
        direction_spreads.append(spread)
        edge_sources.append(source)

    corners: list[np.ndarray] = []
    uncertainties: list[float] = []
    for index in range(len(lines)):
        point = _intersection(lines[index - 1], lines[index])
        if point is None or not np.all(np.isfinite(point)):
            diagnostic.failure = "相邻拟合边近平行，无法求交"
            return None, diagnostic
        if np.linalg.norm(point - initial[index]) > max(2.0 * search_px, 16.0):
            point = initial[index]
            diagnostic.fallback_used = True
        angle = abs(float(np.linalg.det(np.asarray([lines[index - 1][:2], lines[index][:2]]))))
        uncertainties.append((residuals[index - 1] + residuals[index]) / max(angle, 0.05))
        corners.append(point)
    polygon = canonicalize_corners(np.asarray(corners, dtype=np.float32))
    if not _polygon_geometry_is_acceptable(polygon, config, mm_px):
        polygon = _retain_feasible_refinement(
            initial,np.asarray(corners,np.float32),contour,config,mm_px,
        )
        diagnostic.fallback_used = True
    if not _polygon_geometry_is_acceptable(polygon, config, mm_px):
        diagnostic.failure = "拟合交点不能形成满足物理约束的简单多边形"
        return None, diagnostic

    diagnostic.lines = tuple(lines)
    diagnostic.edge_sources = tuple(edge_sources)
    diagnostic.edge_support = support / max(expected, 1)
    diagnostic.normal_residual_px = float(np.mean(residuals))
    diagnostic.direction_std_deg = float(np.mean(direction_spreads))
    diagnostic.intersection_uncertainty_px = float(np.mean(uncertainties))
    diagnostic.minimum_edge_mm = float(
        np.min(np.linalg.norm(polygon - np.roll(polygon, -1, axis=0), axis=1))
        * mm_px
    )
    area = float(cv2.contourArea(polygon))
    moments = cv2.moments(polygon)
    center = (
        moments["m10"] / moments["m00"],
        moments["m01"] / moments["m00"],
    ) if abs(moments["m00"]) > 1e-9 else tuple(np.mean(polygon, axis=0))
    corners_px = tuple((round(float(x), 2), round(float(y), 2)) for x, y in polygon)
    center_px = (round(float(center[0]), 2), round(float(center[1]), 2))
    fitted_iou = polygon_iou(contour, polygon)
    if fitted_iou < config.minimum_polygon_iou:
        fallback_polygon = canonicalize_corners(initial.astype(np.float32))
        fallback_iou = polygon_iou(contour, fallback_polygon)
        if fallback_iou < config.minimum_polygon_iou:
            diagnostic.failure = "精修多边形与真实轮廓 IoU 不足"
            return None, diagnostic
        polygon = fallback_polygon
        fitted_iou = fallback_iou
        area = float(cv2.contourArea(polygon))
        moments = cv2.moments(polygon)
        center = (
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        )
        corners_px = tuple(
            (round(float(x), 2), round(float(y), 2))
            for x, y in polygon
        )
        center_px = (round(float(center[0]), 2), round(float(center[1]), 2))
        diagnostic.minimum_edge_mm = float(
            np.min(polygon_edge_lengths(polygon)) * mm_px
        )
        diagnostic.fallback_used = True
    confidence = float(
        np.clip(0.65 * instance.score + 0.35 * fitted_iou, 0.0, 1.0)
    )
    piece = PieceDetection(
        group_id=group_id,
        corners_px=corners_px,
        center_px=center_px,
        corners_relative=tuple(
            relative_point(point, width, height)
            for point in corners_px
        ),
        center_relative=relative_point(center_px, width, height),
        area_px=round(area, 2),
        area_relative=round(area / (width * height), 6),
        polygon_iou=round(fitted_iou, 4),
        line_residual=round(diagnostic.normal_residual_px, 3),
        confidence=round(confidence, 4),
    )
    return piece, diagnostic
