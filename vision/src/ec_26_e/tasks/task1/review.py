"""审查单帧视觉结果并转换为竖向 A4 毫米坐标。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from shapely.geometry import Polygon

from ec_26_e.cam import CameraFrame
from ec_26_e.control import PuzzlePiece
from ec_26_e.tasks.task1.contracts import LandscapeRotation
from ec_26_e.vision import Detect1Result, ResizeResult


A4_WIDTH_MM = 207.0
A4_HEIGHT_MM = 292.0
_MIN_POLYGON_AREA_MM2 = 1e-6


@dataclass(frozen=True, slots=True)
class ReviewedFrame:
    """保存一帧经严格审查后的 A4 角点和毫米纸片轮廓。"""

    sequence_id: int
    received_at_ns: int
    workspace_corners_source_px: tuple[tuple[float, float], ...]
    pieces: tuple[PuzzlePiece, ...]
    piece_confidences: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """校验可选检测置信度与纸片观测一一对应。"""

        if self.piece_confidences and len(self.piece_confidences) != len(
            self.pieces
        ):
            raise ValueError("纸片置信度数量必须与纸片数量一致")
        for confidence in self.piece_confidences:
            if isinstance(confidence, bool) or not isinstance(
                confidence,
                (int, float),
            ):
                raise TypeError("纸片检测置信度必须是实数")
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("纸片检测置信度必须位于 [0, 1]")


def _landscape_point_to_portrait_mm(
    point: tuple[float, float],
    rotation: LandscapeRotation,
) -> tuple[float, float]:
    """把横向 A4 相对坐标旋转为竖向 A4 毫米坐标。"""

    u, v = point
    if rotation == "clockwise":
        return (
            A4_WIDTH_MM * (1.0 - v),
            A4_HEIGHT_MM * u,
        )
    return (
        A4_WIDTH_MM * v,
        A4_HEIGHT_MM * (1.0 - u),
    )


def _review_relative_vertices(
    group_id: int,
    vertices: tuple[tuple[float, float], ...],
    rotation: LandscapeRotation,
) -> PuzzlePiece:
    """校验一个 Detect1 轮廓并创建控制层强类型纸片。"""

    if isinstance(group_id, bool) or not isinstance(group_id, int):
        raise TypeError("Detect1 group_id 必须是整数")
    if not isinstance(vertices, tuple) or len(vertices) < 3:
        raise ValueError(f"group {group_id}: 至少需要 3 个相对坐标顶点")

    checked: list[tuple[float, float]] = []
    for index, point in enumerate(vertices):
        if not isinstance(point, tuple) or len(point) != 2:
            raise TypeError(
                f"group {group_id}: corners_relative[{index}] 必须是二维元组"
            )
        u, v = point
        for value in point:
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    f"group {group_id}: 相对坐标必须只包含实数"
                )
            if not math.isfinite(value):
                raise ValueError(
                    f"group {group_id}: 相对坐标必须只包含有限数"
                )
        if not 0.0 <= u <= 1.0 or not 0.0 <= v <= 1.0:
            raise ValueError(
                f"group {group_id}: 相对坐标必须位于 [0, 1]"
            )
        checked.append((float(u), float(v)))

    if len(set(checked)) != len(checked):
        raise ValueError(f"group {group_id}: 顶点不能重复或重复闭合")
    signed_area = 0.5 * sum(
        x * checked[(index + 1) % len(checked)][1]
        - checked[(index + 1) % len(checked)][0] * y
        for index, (x, y) in enumerate(checked)
    )
    if signed_area <= 0.0:
        raise ValueError(
            f"group {group_id}: 顶点必须按视觉顺时针顺序排列"
        )

    vertices_mm = tuple(
        _landscape_point_to_portrait_mm(point, rotation)
        for point in checked
    )
    polygon = Polygon(vertices_mm)
    if not polygon.is_valid or polygon.area <= _MIN_POLYGON_AREA_MM2:
        raise ValueError(f"group {group_id}: 多边形自相交或退化")
    return PuzzlePiece(
        piece_id=str(group_id),
        vertices_mm=vertices_mm,
    )


def review_visual_frame(
    frame: CameraFrame,
    resize_result: ResizeResult,
    detect_result: Detect1Result,
    rotation: LandscapeRotation,
) -> ReviewedFrame:
    """审查相机、Resize 和 Detect1 的关联契约并转换纸片坐标。"""

    if not isinstance(frame, CameraFrame):
        raise TypeError("camera.read_latest 必须返回 CameraFrame")
    if not isinstance(resize_result, ResizeResult):
        raise TypeError("resizer.resize 必须返回 ResizeResult")
    if not isinstance(detect_result, Detect1Result):
        raise TypeError("detector.detect 必须返回 Detect1Result")

    source_height, source_width = frame.image_bgr.shape[:2]
    if resize_result.source_size != (source_width, source_height):
        raise ValueError("ResizeResult.source_size 与相机帧尺寸不一致")
    target_width, target_height = resize_result.target_size
    if resize_result.image_bgr.shape[:2] != (
        target_height,
        target_width,
    ):
        raise ValueError("ResizeResult.target_size 与校正图尺寸不一致")
    if detect_result.image_size != resize_result.target_size:
        raise ValueError("Detect1Result.image_size 与 Resize 目标尺寸不一致")
    if detect_result.coordinate_system != "a4_landscape_top_left":
        raise ValueError("Detect1Result 使用了不支持的坐标系")

    corners = np.asarray(resize_result.source_corners_px, dtype=float)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("ResizeResult.source_corners_px 必须是四个有限角点")
    source_corners = tuple((float(x), float(y)) for x, y in corners)

    ids = [piece.group_id for piece in detect_result.pieces]
    if len(set(ids)) != len(ids):
        raise ValueError("Detect1Result 的 group_id 不能重复")
    pieces = tuple(
        _review_relative_vertices(
            piece.group_id,
            piece.corners_relative,
            rotation,
        )
        for piece in detect_result.pieces
    )
    return ReviewedFrame(
        sequence_id=frame.sequence_id,
        received_at_ns=frame.received_at_ns,
        workspace_corners_source_px=source_corners,
        pieces=pieces,
        piece_confidences=tuple(
            float(piece.confidence) for piece in detect_result.pieces
        ),
    )
