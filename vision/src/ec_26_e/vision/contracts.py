"""定义校正后横向 A4 图像上的纸片检测公共契约。"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np

from ec_26_e.vision.resize.contracts import (
    A4_LANDSCAPE_RATIO,
    ASPECT_RATIO_TOLERANCE,
    aspect_error,
)


def _validate_real_number(name: str, value: float) -> None:
    """校验配置值是有限实数。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是实数")
    if not np.isfinite(value):
        raise ValueError(f"{name} 必须是有限实数")


def _validate_ratio(
    name: str,
    value: float,
    *,
    allow_zero: bool = False,
) -> None:
    """校验配置比例位于约定范围。"""

    _validate_real_number(name, value)
    lower_valid = value >= 0.0 if allow_zero else value > 0.0
    if not lower_valid or value > 1.0:
        lower_symbol = "[" if allow_zero else "("
        raise ValueError(f"{name} 必须位于 {lower_symbol}0, 1] 范围内")


def _validate_positive_integer(name: str, value: int) -> None:
    """校验配置值是正整数。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须是整数")
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0")


@dataclass(frozen=True, slots=True)
class Detect1Config:
    """描述 A4 安全边缘和纸片轮廓筛选参数。"""

    workspace_erode_ratio: float = 0.007
    min_piece_area_ratio: float = 0.004
    max_piece_area_ratio: float = 0.20
    polygon_epsilon_ratio: float = 0.014
    short_edge_merge_mm: float = 20.0
    minimum_physical_edge_mm: float = 20.0
    minimum_corner_turn_deg: float = 12.0
    minimum_corner_depth_mm: float = 4.0
    minimum_polygon_iou: float = 0.85
    min_vertices: int = 3
    max_vertices: int = 6
    max_pieces: int = 12
    piece_mode: int = 2

    def __post_init__(self) -> None:
        """校验比例、顶点数量和候选数量约束。"""

        if isinstance(self.piece_mode, bool) or not isinstance(
            self.piece_mode,
            int,
        ):
            raise TypeError("piece_mode 必须是整数")
        if self.piece_mode not in {1, 2}:
            raise ValueError("piece_mode 只能为 1（白纸）或 2（图案纸片）")
        _validate_ratio(
            "workspace_erode_ratio",
            self.workspace_erode_ratio,
            allow_zero=True,
        )
        _validate_ratio("min_piece_area_ratio", self.min_piece_area_ratio)
        _validate_ratio("max_piece_area_ratio", self.max_piece_area_ratio)
        _validate_ratio("polygon_epsilon_ratio", self.polygon_epsilon_ratio)
        if isinstance(self.short_edge_merge_mm, bool):
            raise TypeError("short_edge_merge_mm 必须是实数")
        if isinstance(self.minimum_physical_edge_mm, bool):
            raise TypeError("minimum_physical_edge_mm 必须是实数")
        _validate_real_number("short_edge_merge_mm", self.short_edge_merge_mm)
        _validate_real_number(
            "minimum_physical_edge_mm",
            self.minimum_physical_edge_mm,
        )
        _validate_real_number(
            "minimum_corner_turn_deg",
            self.minimum_corner_turn_deg,
        )
        _validate_real_number(
            "minimum_corner_depth_mm",
            self.minimum_corner_depth_mm,
        )
        _validate_ratio("minimum_polygon_iou", self.minimum_polygon_iou)
        if self.minimum_physical_edge_mm <= 0.0:
            raise ValueError("minimum_physical_edge_mm 必须大于 0 mm")
        if self.short_edge_merge_mm < self.minimum_physical_edge_mm:
            raise ValueError(
                "short_edge_merge_mm 不能小于 minimum_physical_edge_mm"
            )
        if not 0.0 < self.minimum_corner_turn_deg < 180.0:
            raise ValueError("minimum_corner_turn_deg 必须位于 (0, 180) 度")
        if self.minimum_corner_depth_mm <= 0.0:
            raise ValueError("minimum_corner_depth_mm 必须大于 0 mm")
        _validate_positive_integer("min_vertices", self.min_vertices)
        _validate_positive_integer("max_vertices", self.max_vertices)
        _validate_positive_integer("max_pieces", self.max_pieces)
        if self.min_piece_area_ratio >= self.max_piece_area_ratio:
            raise ValueError(
                "min_piece_area_ratio 必须小于 max_piece_area_ratio"
            )
        if self.min_vertices < 3:
            raise ValueError("min_vertices 不能小于 3")
        if self.min_vertices > self.max_vertices:
            raise ValueError("min_vertices 不能大于 max_vertices")


@dataclass(frozen=True, slots=True)
class PieceDetection:
    """描述纸片在校正 A4 图上的像素坐标和相对坐标。"""

    group_id: int
    corners_px: tuple[tuple[float, float], ...]
    center_px: tuple[float, float]
    corners_relative: tuple[tuple[float, float], ...]
    center_relative: tuple[float, float]
    area_px: float
    area_relative: float
    polygon_iou: float
    line_residual: float
    confidence: float

    def to_dict(self) -> dict[str, object]:
        """转换为适合 JSON 序列化的字典。"""

        return {
            "group_id": self.group_id,
            "corners_px": [list(point) for point in self.corners_px],
            "center_px": list(self.center_px),
            "corners_relative": [
                list(point) for point in self.corners_relative
            ],
            "center_relative": list(self.center_relative),
            "area_px": self.area_px,
            "area_relative": self.area_relative,
            "polygon_iou": self.polygon_iou,
            "line_residual": self.line_residual,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class Detect1Result:
    """描述一张横向 A4 图上的全部纸片及前景诊断信息。"""

    image_size: tuple[int, int]
    pieces: tuple[PieceDetection, ...]
    foreground_threshold: float
    workspace_area_px: int
    coordinate_system: str = "a4_landscape_top_left"

    def to_dict(self) -> dict[str, object]:
        """转换为包含坐标系声明的 JSON 兼容字典。"""

        return {
            "image_size": list(self.image_size),
            "coordinate_system": self.coordinate_system,
            "piece_count": len(self.pieces),
            "groups": [piece.to_dict() for piece in self.pieces],
            "diagnostics": {
                "foreground_threshold": self.foreground_threshold,
                "workspace_area_px": self.workspace_area_px,
            },
        }


def validate_a4_image_size(width: int, height: int) -> None:
    """校验 Detect1 输入为符合容差的横向 A4 尺寸。"""

    if width <= height:
        raise ValueError("Detect1 输入必须是横向 A4 图像")
    if aspect_error(width / height, A4_LANDSCAPE_RATIO) > (
        ASPECT_RATIO_TOLERANCE
    ):
        raise ValueError("Detect1 输入宽高比必须符合横向 A4 的 292:207")


@dataclass(frozen=True, slots=True)
class NativeTiming:
    """记录 C ABI 内部各阶段耗时，单位为毫秒。"""

    preprocess_ms: float = 0.0
    input_flush_ms: float = 0.0
    network_ms: float = 0.0
    output_map_ms: float = 0.0
    detection_decode_ms: float = 0.0
    nms_ms: float = 0.0
    mask_decode_ms: float = 0.0
    output_unmap_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """转换为 JSON 兼容的毫秒字典。"""

        return {
            name: round(float(getattr(self, name)), 4)
            for name in self.__slots__
        }


@dataclass(frozen=True, slots=True)
class SegInstance:
    """描述已经逆 letterbox 到 A4 图坐标的单个实例。"""

    box_px: tuple[float, float, float, float]
    score: float
    class_id: int
    mask: np.ndarray


@dataclass(frozen=True, slots=True)
class VisionConfig:
    """描述 NPU 阈值和毫米尺度的传统视觉精修参数。"""

    model_path: Path = Path(__file__).resolve().parents[4] / "model-lib/q6a/puzzle_q6a.bin"
    library_path: Path = Path(__file__).resolve().parents[4] / "model-lib/q6a/libq6a_qnn.so"
    score_threshold: float = 0.15
    nms_threshold: float = 0.45
    mask_threshold: float = 0.50
    dilation_mm: float = 3.0
    fallback_dilation_mm: float = 4.0
    short_edge_merge_mm: float = 20.0
    minimum_physical_edge_mm: float = 20.0
    minimum_corner_turn_deg: float = 12.0
    minimum_corner_depth_mm: float = 4.0
    minimum_polygon_iou: float = 0.85
    extra_vertex_iou_gain: float = 0.025
    min_vertices: int = 3
    max_vertices: int = 5
    max_pieces: int = 4

    def __post_init__(self) -> None:
        """校验用户可调字段，避免把非法值传进原生层。"""

        for name in (
            "score_threshold",
            "nms_threshold",
            "mask_threshold",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(f"{name} 必须是实数")
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须位于 [0, 1]")
        if not 2.0 <= self.dilation_mm <= 4.0:
            raise ValueError("dilation_mm 必须位于 [2, 4] mm")
        if not (
            self.dilation_mm
            <= self.fallback_dilation_mm
            <= 4.0
        ):
            raise ValueError(
                "fallback_dilation_mm 必须不小于 dilation_mm 且不大于 4 mm"
            )
        for name in (
            "short_edge_merge_mm",
            "minimum_physical_edge_mm",
            "minimum_corner_turn_deg",
            "minimum_corner_depth_mm",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} 必须是实数")
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} 必须是大于 0 的有限毫米值")
        if self.short_edge_merge_mm < self.minimum_physical_edge_mm:
            raise ValueError(
                "short_edge_merge_mm 不能小于 minimum_physical_edge_mm"
            )
        if not 0.0 < self.minimum_corner_turn_deg < 180.0:
            raise ValueError("minimum_corner_turn_deg 必须位于 (0, 180) 度")
        if (
            isinstance(self.minimum_polygon_iou, bool)
            or not isinstance(self.minimum_polygon_iou, (int, float))
        ):
            raise TypeError("minimum_polygon_iou 必须是实数")
        if not 0.0 < self.minimum_polygon_iou <= 1.0:
            raise ValueError("minimum_polygon_iou 必须位于 (0, 1]")
        if not 0.0 <= self.extra_vertex_iou_gain <= 0.10:
            raise ValueError("extra_vertex_iou_gain 必须位于 [0, 0.10]")
        if not 3 <= self.min_vertices <= self.max_vertices <= 5:
            raise ValueError("顶点数必须限制在 3～5")
        if not 1 <= self.max_pieces <= 4:
            raise ValueError("max_pieces 必须限制在 1～4")


@dataclass(frozen=True, slots=True)
class StableVisionConfig:
    """描述连续帧稳定窗口和唯一的送控制前几何要求。"""

    window_size: int = 9
    required_frames: int = 6
    max_vertex_jitter_mm: float = 2.0
    max_center_distance_mm: float = 30.0
    min_mask_iou: float = 0.20

    def __post_init__(self) -> None:
        """将稳定窗口和连续帧数限制为约定的 5～10 帧。"""

        if not 5 <= self.window_size <= 10:
            raise ValueError("window_size 必须位于 5～10")
        if not 5 <= self.required_frames <= self.window_size:
            raise ValueError("required_frames 必须位于 5～window_size")
        if (
            self.max_vertex_jitter_mm <= 0
            or self.max_center_distance_mm <= 0
        ):
            raise ValueError("稳定距离必须大于 0")
        if not 0.0 <= self.min_mask_iou <= 1.0:
            raise ValueError("min_mask_iou 必须位于 [0, 1]")


@dataclass(slots=True)
class InstanceDiagnostics:
    """保存单实例的粗/精掩膜、拟合直线和质量信息。"""

    score: float
    coarse_mask: np.ndarray
    refined_mask: np.ndarray
    lines: tuple[tuple[float, float, float], ...] = ()
    edge_support: float = 0.0
    normal_residual_px: float = 0.0
    direction_std_deg: float = 0.0
    intersection_uncertainty_px: float = 0.0
    coarse_refined_iou: float = 0.0
    minimum_edge_mm: float = 0.0
    merged_below_minimum_edges: int = 0
    boundary_source: str = ""
    edge_sources: tuple[str, ...] = ()
    fallback_used: bool = False
    failure: str | None = None


@dataclass(slots=True)
class VisionDiagnostics:
    """保存一次统一视觉调用的全部调试数据和分阶段耗时。"""

    native_timing: NativeTiming = field(default_factory=NativeTiming)
    stage_ms: dict[str, float] = field(default_factory=dict)
    instances: list[InstanceDiagnostics] = field(default_factory=list)
    yolo_count: int = 0
    contained_duplicate_count: int = 0
    refined_count: int = 0
    driver_version: int = 0

    def to_dict(self) -> dict[str, object]:
        """输出不包含大体积掩膜的 JSON 兼容诊断。"""

        return {
            "native_timing": self.native_timing.to_dict(),
            "stage_ms": {
                key: round(value, 4)
                for key, value in self.stage_ms.items()
            },
            "yolo_count": self.yolo_count,
            "contained_duplicate_count": self.contained_duplicate_count,
            "refined_count": self.refined_count,
            "driver_version": self.driver_version,
            "instances": [
                {
                    "score": round(item.score, 4),
                    "edge_support": round(item.edge_support, 4),
                    "normal_residual_px": round(
                        item.normal_residual_px,
                        4,
                    ),
                    "direction_std_deg": round(item.direction_std_deg, 4),
                    "intersection_uncertainty_px": round(
                        item.intersection_uncertainty_px,
                        4,
                    ),
                    "coarse_refined_iou": round(
                        item.coarse_refined_iou,
                        4,
                    ),
                    "minimum_edge_mm": round(item.minimum_edge_mm, 4),
                    "merged_below_minimum_edges": (
                        item.merged_below_minimum_edges
                    ),
                    "boundary_source": item.boundary_source,
                    "edge_sources": list(item.edge_sources),
                    "fallback_used": item.fallback_used,
                    "failure": item.failure,
                }
                for item in self.instances
            ],
        }
