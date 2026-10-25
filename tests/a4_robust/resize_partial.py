"""部分遮挡放行的 Resize。

复用参考工程 resizer 的全部内部函数，仅放宽：
- 每条边独立精拟合，允许失败边用 seed 边线兜底（仅限估计角相邻边）；
- 估计角相邻边的拟合搜索带加宽（角点先验不确定）；
- 几何校验：估计角允许贴近/少量越出图像边缘；
- 外观校验：估计角相邻边跳过外暗内亮抽样；若全部边都涉及估计角，
  退回纸面内部采样（亮度区间 + 青蓝色差）。
不修改参考工程任何文件。
"""

from __future__ import annotations

from math import cos, radians

import cv2
import numpy as np

try:
    from . import _paths
except ImportError:
    import _paths
_paths.ensure()

from ec_26_e.vision._blue import validate_bgr_image
from ec_26_e.vision.resize.contracts import (
    ASPECT_RATIO_TOLERANCE,
    ResizeResult,
    aspect_error,
)
from ec_26_e.vision.resize.resizer import (
    Resize,
    _EdgeFit,
    _FINE_SEARCH_BAND_RATIO,
    _MAX_COARSE_RESIDUAL_PX,
    _MAX_FINE_RESIDUAL_PX,
    _MAX_RELIABLE_PAPER_LUMINANCE,
    _MIN_EDGE_LUMINANCE_ADVANTAGE,
    _MIN_EDGE_LUMINANCE_SAMPLE_ADVANTAGE,
    _MIN_EDGE_LUMINANCE_SUPPORT_RATIO,
    _MIN_PAPER_CHROMA,
    _MIN_PAPER_CHROMA_RATIO,
    _MIN_RELIABLE_PAPER_LUMINANCE,
    _MIN_RELIABLE_PAPER_SAMPLE_RATIO,
    _fit_transition_edge,
    _luminance_image,
    _paper_chroma_image,
    _reference_corners,
)

# 估计角相邻边：精拟合搜索带放大倍数
_ESTIMATED_BAND_SCALE = 2.0
# 估计角允许越出图像边缘的比例（相对短边）
_ESTIMATED_BORDER_TOLERANCE = 0.08


class ResizePartial(Resize):
    """与 Resize 输出契约一致，但对估计角放宽校验。"""

    def __init__(
        self,
        config=None,
        *,
        reliability: tuple[bool, bool, bool, bool] = (True, True, True, True),
        estimated_band_scale: float = _ESTIMATED_BAND_SCALE,
    ) -> None:
        super().__init__(config)
        if len(reliability) != 4:
            raise ValueError("reliability 必须是 4 个布尔值")
        self._reliability = tuple(bool(value) for value in reliability)
        self._band_scale = estimated_band_scale

    def resize(self, image_bgr: np.ndarray) -> ResizeResult:
        """与参考 Resize.resize 相同，但工作区定位使用部分遮挡放行路径。"""
        validate_bgr_image(image_bgr)
        source_height, source_width = image_bgr.shape[:2]
        if (
            self.config.detect_width > source_width
            or self.config.detect_height > source_height
        ):
            raise ValueError("工作区定位分辨率不能大于输入图像")
        source_ratio = source_width / source_height
        detect_ratio = self.config.detect_width / self.config.detect_height
        if aspect_error(detect_ratio, source_ratio) > ASPECT_RATIO_TOLERANCE:
            raise ValueError(
                "工作区定位分辨率必须保持输入图像宽高比，允许的相对误差为 0.5%"
            )
        (
            source_corners,
            threshold,
            workspace_ratio,
            line_residual,
        ) = _locate_workspace_partial(image_bgr, self.config, self._reliability, self._band_scale)
        target_width = self.config.target_width
        target_height = self.config.target_height
        target_corners = np.asarray(
            [
                [0.0, 0.0],
                [target_width - 1.0, 0.0],
                [target_width - 1.0, target_height - 1.0],
                [0.0, target_height - 1.0],
            ],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(
            source_corners.astype(np.float32),
            target_corners,
        )
        rectified = cv2.warpPerspective(
            image_bgr,
            homography,
            (target_width, target_height),
            flags=cv2.INTER_LINEAR,
        )
        return ResizeResult(
            image_bgr=rectified,
            source_size=(source_width, source_height),
            target_size=(target_width, target_height),
            source_corners_px=tuple(
                (round(float(x), 2), round(float(y), 2))
                for x, y in source_corners
            ),
            homography=homography,
            workspace_threshold=round(threshold, 4),
            workspace_ratio=round(workspace_ratio, 4),
            line_residual=round(line_residual, 3),
        )


def _locate_workspace_partial(
    image_bgr: np.ndarray,
    config: object,
    reliability: tuple[bool, bool, bool, bool],
    band_scale: float,
) -> tuple[np.ndarray, float, float, float]:
    """粗定位 + 精拟合；失败边用 seed 边线兜底；校验按估计角放宽。"""
    source_height, source_width = image_bgr.shape[:2]
    detect_size = (config.detect_width, config.detect_height)
    detection_image = cv2.resize(
        image_bgr,
        detect_size,
        interpolation=cv2.INTER_AREA,
    )
    detect_reference = _reference_corners(config, detect_size)
    coarse_band = max(10.0, config.search_band_ratio * min(detect_size))
    coarse_edges = _fit_edges_tolerant(
        _luminance_image(detection_image),
        detect_reference,
        coarse_band,
        _MAX_COARSE_RESIDUAL_PX,
        reliability,
        band_scale=1.0,
    )
    scale = np.asarray(
        [
            (source_width - 1) / max(config.detect_width - 1, 1),
            (source_height - 1) / max(config.detect_height - 1, 1),
        ],
        dtype=np.float32,
    )
    source_seed = _corners_from_edge_fits(coarse_edges, detect_reference) * scale
    fine_band = max(6.0, _FINE_SEARCH_BAND_RATIO * min(source_width, source_height))
    fine_edges = _fit_edges_tolerant(
        _luminance_image(image_bgr),
        source_seed,
        fine_band,
        _MAX_FINE_RESIDUAL_PX,
        reliability,
        band_scale=band_scale,
    )
    source_corners = _corners_from_edge_fits(fine_edges, source_seed)
    source_reference = _reference_corners(config, (source_width, source_height))
    _validate_geometry_partial(
        source_corners,
        source_reference,
        (source_width, source_height),
        config.min_workspace_ratio,
        config.search_band_ratio,
        reliability,
    )
    appearance_score = _validate_workspace_appearance_partial(
        image_bgr,
        source_corners,
        reliability,
    )
    workspace_ratio = abs(
        cv2.contourArea(source_corners)
    ) / float(source_width * source_height)
    fitted_residuals = [edge.residual for edge in fine_edges if edge is not None]
    line_residual = float(np.mean(fitted_residuals)) if fitted_residuals else 0.0
    gradient_thresholds = [
        edge.gradient_threshold
        for edge in coarse_edges + fine_edges
        if edge is not None
    ]
    gradient_threshold = float(np.mean(gradient_thresholds)) if gradient_thresholds else 0.0
    return (
        source_corners,
        min(gradient_threshold, appearance_score),
        workspace_ratio,
        line_residual,
    )


def _fit_edges_tolerant(
    luminance: np.ndarray,
    seed_corners: np.ndarray,
    search_band_px: float,
    max_residual_px: float,
    reliability: tuple[bool, bool, bool, bool],
    *,
    band_scale: float,
) -> list[_EdgeFit | None]:
    """逐边独立拟合；失败的边返回 None（seed 边线兜底）。

    仅当该边两个端点都被判为实测角时才把失败当作致命错误；
    估计角相邻边失败时容忍并返回 None。
    """
    fits: list[_EdgeFit | None] = []
    for index in range(4):
        start = seed_corners[index]
        end = seed_corners[(index + 1) % 4]
        band = search_band_px
        if band_scale > 1.0 and (not reliability[index] or not reliability[(index + 1) % 4]):
            band *= band_scale
        try:
            fits.append(
                _fit_transition_edge(
                    luminance,
                    start,
                    end,
                    band,
                    max_residual_px,
                )
            )
        except ValueError:
            if reliability[index] and reliability[(index + 1) % 4]:
                raise
            fits.append(None)
    return fits


def _point_direction_intersection(
    point_a: np.ndarray,
    direction_a: np.ndarray,
    point_b: np.ndarray,
    direction_b: np.ndarray,
) -> np.ndarray | None:
    """两条 (点, 方向) 直线求交；近平行时返回 None。"""
    matrix = np.column_stack((direction_a, -direction_b))
    if abs(float(np.linalg.det(matrix))) < 0.08:
        return None
    parameters = np.linalg.solve(matrix, point_b - point_a)
    return point_a + parameters[0] * direction_a


def _corners_from_edge_fits(
    edges: list[_EdgeFit | None],
    seed_corners: np.ndarray,
) -> np.ndarray:
    """逐边求交得四角；失败的边用 seed 两端点连线（seed 边线）代替。"""
    resolved: list[tuple[np.ndarray, np.ndarray]] = []
    for index, edge in enumerate(edges):
        if edge is not None:
            resolved.append((edge.point, edge.direction))
            continue
        start = seed_corners[index]
        end = seed_corners[(index + 1) % 4]
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length < 1e-6:
            raise ValueError("seed 边退化")
        resolved.append((start, vector / length))
    corners: list[np.ndarray] = []
    for index in range(4):
        point_a, direction_a = resolved[index - 1]
        point_b, direction_b = resolved[index]
        intersection = _point_direction_intersection(
            point_a, direction_a, point_b, direction_b
        )
        if intersection is None:
            raise ValueError("相邻边近平行，无法稳定求交")
        corners.append(intersection)
    return np.asarray(corners, dtype=np.float32)


def _validate_geometry_partial(
    corners: np.ndarray,
    reference: np.ndarray,
    image_size: tuple[int, int],
    min_workspace_ratio: float,
    search_band_ratio: float,
    reliability: tuple[bool, bool, bool, bool],
) -> None:
    """参考 _validate_geometry，但估计角允许贴近/少量越出图像边缘。"""
    width, height = image_size
    if not np.all(np.isfinite(corners)):
        raise ValueError("A4 四角包含非有限坐标")
    border_tolerance = _ESTIMATED_BORDER_TOLERANCE * min(width, height)
    for index, corner in enumerate(corners):
        allowed = border_tolerance if not reliability[index] else 2.0
        if (
            corner[0] < -allowed
            or corner[0] > width - 1.0 + allowed
            or corner[1] < -allowed
            or corner[1] > height - 1.0 + allowed
        ):
            raise ValueError("A4 边界不完整或接触图像边缘")
    integer_corners = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(integer_corners):
        raise ValueError("A4 四角未形成凸四边形")
    signed_area = 0.5 * float(
        np.sum(corners[:, 0] * np.roll(corners[:, 1], -1) - corners[:, 1] * np.roll(corners[:, 0], -1))
    )
    if signed_area <= 0.0:
        raise ValueError("A4 四角环绕方向错误")
    area_ratio = abs(cv2.contourArea(corners)) / float(width * height)
    if area_ratio < min_workspace_ratio:
        raise ValueError(
            f"A4 工作区占比过小：{area_ratio:.3f} < {min_workspace_ratio:.3f}"
        )
    side_lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    if float(np.min(side_lengths)) < 0.15 * min(width, height):
        raise ValueError("A4 四边形包含过短边")
    maximum_normal_offset = 1.15 * search_band_ratio * min(width, height)
    minimum_direction_dot = cos(radians(20.0))
    for index in range(4):
        reference_edge = reference[(index + 1) % 4] - reference[index]
        candidate_edge = corners[(index + 1) % 4] - corners[index]
        reference_length = float(np.linalg.norm(reference_edge))
        candidate_length = float(np.linalg.norm(candidate_edge))
        reference_direction = reference_edge / reference_length
        candidate_direction = candidate_edge / candidate_length
        if float(reference_direction @ candidate_direction) < minimum_direction_dot:
            raise ValueError("A4 边方向偏离参考姿态过大")
        normal = np.asarray([-reference_direction[1], reference_direction[0]])
        midpoint_offset = abs(
            float(
                (
                    (corners[index] + corners[(index + 1) % 4]) / 2.0
                    - (reference[index] + reference[(index + 1) % 4]) / 2.0
                )
                @ normal
            )
        )
        if midpoint_offset > maximum_normal_offset:
            raise ValueError("A4 边位置超出参考搜索范围")


def _validate_workspace_appearance_partial(
    image_bgr: np.ndarray,
    corners: np.ndarray,
    reliability: tuple[bool, bool, bool, bool],
) -> float:
    """外观校验：估计角相邻边跳过外暗内亮抽样；全部跳过时退回内部采样。"""
    luminance = _luminance_image(image_bgr)
    paper_chroma = _paper_chroma_image(image_bgr)
    short_side = min(image_bgr.shape[:2])
    near_offset = max(5.0, 0.008 * short_side)
    far_offset = max(10.0, 0.018 * short_side)
    offsets = np.linspace(near_offset, far_offset, 6, dtype=np.float32)

    luminance_advantages: list[float] = []
    inside_luminance_samples: list[np.ndarray] = []
    inside_chroma_samples: list[np.ndarray] = []
    checked_any = False
    for index in range(4):
        if not reliability[index] or not reliability[(index + 1) % 4]:
            continue
        start = corners[index]
        end = corners[(index + 1) % 4]
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length < 20.0:
            continue
        direction = edge / length
        inward = np.asarray([-direction[1], direction[0]], dtype=np.float32)
        positions = np.linspace(
            0.12 * length,
            0.88 * length,
            max(80, int(round(0.25 * length))),
            dtype=np.float32,
        )
        bases = start[np.newaxis, :] + positions[:, np.newaxis] * direction[np.newaxis, :]
        inside_points = (
            bases[:, np.newaxis, :]
            + offsets[np.newaxis, :, np.newaxis] * inward[np.newaxis, np.newaxis, :]
        ).astype(np.float32)
        outside_points = (
            bases[:, np.newaxis, :]
            - offsets[np.newaxis, :, np.newaxis] * inward[np.newaxis, np.newaxis, :]
        ).astype(np.float32)
        inside_luminance = cv2.remap(
            luminance,
            inside_points[..., 0],
            inside_points[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        outside_luminance = cv2.remap(
            luminance,
            outside_points[..., 0],
            outside_points[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        luminance_difference = inside_luminance - outside_luminance
        luminance_advantage = float(np.median(luminance_difference))
        luminance_support_ratio = float(
            np.mean(luminance_difference >= _MIN_EDGE_LUMINANCE_SAMPLE_ADVANTAGE)
        )
        if luminance_advantage < _MIN_EDGE_LUMINANCE_ADVANTAGE:
            raise ValueError("A4 边界外暗内亮差不足")
        if luminance_support_ratio < _MIN_EDGE_LUMINANCE_SUPPORT_RATIO:
            raise ValueError("A4 边界外暗内亮支持不足")
        inside_chroma = cv2.remap(
            paper_chroma,
            inside_points[..., 0],
            inside_points[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        checked_any = True
        luminance_advantages.append(luminance_advantage)
        inside_luminance_samples.append(inside_luminance.reshape(-1))
        inside_chroma_samples.append(inside_chroma.reshape(-1))

    if checked_any:
        combined_luminance = np.concatenate(inside_luminance_samples)
        combined_chroma = np.concatenate(inside_chroma_samples)
    else:
        # 全部边都涉及估计角（严重遮挡）：退回纸面内部网格采样
        mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        cv2.fillPoly(
            mask,
            [np.rint(corners).astype(np.int32).reshape(-1, 1, 2)],
            255,
        )
        kernel_size = int(short_side * 0.03) | 1
        interior = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
        if cv2.countNonZero(interior) < int(0.10 * mask.size):
            raise ValueError("A4 可见纸面过少")
        combined_luminance = luminance[interior > 0]
        combined_chroma = paper_chroma[interior > 0]
    reliable = (
        (combined_luminance >= _MIN_RELIABLE_PAPER_LUMINANCE)
        & (combined_luminance <= _MAX_RELIABLE_PAPER_LUMINANCE)
    )
    reliable_ratio = float(np.mean(reliable))
    if reliable_ratio < _MIN_RELIABLE_PAPER_SAMPLE_RATIO:
        raise ValueError("A4 青蓝纸色的可靠曝光区域不足")
    reliable_chroma = combined_chroma[reliable]
    paper_chroma_median = float(np.median(reliable_chroma))
    paper_chroma_ratio = float(np.mean(reliable_chroma >= _MIN_PAPER_CHROMA))
    if paper_chroma_median < _MIN_PAPER_CHROMA:
        raise ValueError("A4 内侧没有稳定的青蓝卡纸色差")
    if paper_chroma_ratio < _MIN_PAPER_CHROMA_RATIO:
        raise ValueError("A4 青蓝卡纸有效区域不足")
    if checked_any:
        return min(min(luminance_advantages), paper_chroma_median)
    return paper_chroma_median
