"""定义蓝色 A4 工作区定位与透视校正的公共契约。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


A4_LANDSCAPE_RATIO = 292.0 / 207.0
ASPECT_RATIO_TOLERANCE = 0.005
DEFAULT_REFERENCE_CORNERS_RELATIVE = (
    (0.1375, 0.0650),
    (0.7780, 0.0780),
    (0.7710, 0.8700),
    (0.1375, 0.8440),
)


def _validate_positive_integer(name: str, value: int) -> None:
    """校验配置值是正整数且不是布尔值。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须是整数")
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0")


def _validate_ratio(name: str, value: float) -> None:
    """校验比例是零到一之间的有限实数。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是实数")
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"{name} 必须位于 (0, 1] 范围内")


def _signed_polygon_area(points: np.ndarray) -> float:
    """计算相对坐标四边形在图像坐标系中的有向面积。"""

    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(
        np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
    )


def _validate_reference_corners(
    value: object,
) -> tuple[tuple[float, float], ...]:
    """校验并冻结按 TL、TR、BR、BL 排列的参考四角。"""

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise TypeError(
            "reference_corners_relative 必须包含 TL、TR、BR、BL 四点"
        )
    checked: list[tuple[float, float]] = []
    for index, point in enumerate(value):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise TypeError(
                "reference_corners_relative"
                f"[{index}] 必须是二维坐标"
            )
        coordinates: list[float] = []
        for coordinate in point:
            if isinstance(coordinate, bool) or not isinstance(
                coordinate,
                (int, float),
            ):
                raise TypeError("参考四角坐标必须是实数")
            converted = float(coordinate)
            if not np.isfinite(converted) or not 0.0 <= converted <= 1.0:
                raise ValueError("参考四角坐标必须位于 [0, 1]")
            coordinates.append(converted)
        checked.append((coordinates[0], coordinates[1]))

    points = np.asarray(checked, dtype=np.float32)
    integer_points = np.rint(points * 10000.0).astype(np.int32)
    if not cv2.isContourConvex(integer_points.reshape(-1, 1, 2)):
        raise ValueError("参考四角必须形成凸四边形")
    if _signed_polygon_area(points) <= 0.0:
        raise ValueError("参考四角必须按 TL、TR、BR、BL 顺时针排列")
    return tuple(checked)


def aspect_error(actual: float, expected: float) -> float:
    """返回实际宽高比相对期望宽高比的误差。"""

    return abs(actual - expected) / expected


@dataclass(frozen=True, slots=True)
class ResizeConfig:
    """描述工作区定位分辨率、A4 输出尺寸和面积下限。"""

    detect_width: int = 640
    detect_height: int = 360
    target_width: int = 762
    target_height: int = 540
    min_workspace_ratio: float = 0.25
    reference_corners_relative: tuple[tuple[float, float], ...] = (
        DEFAULT_REFERENCE_CORNERS_RELATIVE
    )
    search_band_ratio: float = 0.14
    require_observed_boundary: bool = False

    def __post_init__(self) -> None:
        """在访问 OpenCV 前校验独立字段和 A4 比例约束。"""

        if not isinstance(self.require_observed_boundary, bool):
            raise TypeError("require_observed_boundary 必须为布尔值")
        _validate_positive_integer("detect_width", self.detect_width)
        _validate_positive_integer("detect_height", self.detect_height)
        _validate_positive_integer("target_width", self.target_width)
        _validate_positive_integer("target_height", self.target_height)
        _validate_ratio("min_workspace_ratio", self.min_workspace_ratio)
        _validate_ratio("search_band_ratio", self.search_band_ratio)
        if self.search_band_ratio > 0.25:
            raise ValueError("search_band_ratio 不能大于 0.25")
        reference_corners = _validate_reference_corners(
            self.reference_corners_relative
        )
        object.__setattr__(
            self,
            "reference_corners_relative",
            reference_corners,
        )
        if self.target_width <= self.target_height:
            raise ValueError("A4 resize 输出必须为横向，宽度应大于高度")
        actual_ratio = self.target_width / self.target_height
        if aspect_error(actual_ratio, A4_LANDSCAPE_RATIO) > (
            ASPECT_RATIO_TOLERANCE
        ):
            raise ValueError(
                "A4 resize 宽高比必须符合横向 292:207，"
                "允许的相对误差为 0.5%"
            )


@dataclass(frozen=True, slots=True)
class ResizeResult:
    """描述透视校正图、源图四角和工作区边缘诊断信息。"""

    image_bgr: np.ndarray
    source_size: tuple[int, int]
    target_size: tuple[int, int]
    source_corners_px: tuple[tuple[float, float], ...]
    homography: np.ndarray
    workspace_threshold: float
    workspace_ratio: float
    line_residual: float

    def to_dict(self) -> dict[str, object]:
        """转换为不包含图像像素的 JSON 兼容诊断字典。"""

        return {
            "source_size": list(self.source_size),
            "target_size": list(self.target_size),
            "source_corners_px": [
                list(corner) for corner in self.source_corners_px
            ],
            "homography": self.homography.tolist(),
            "diagnostics": {
                "workspace_threshold": self.workspace_threshold,
                "workspace_ratio": self.workspace_ratio,
                "line_residual": self.line_residual,
            },
        }
