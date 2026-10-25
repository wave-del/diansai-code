"""
验证统一视觉标注线宽、顶点半径和原图不可变性。

该文件不依赖 NPU 或摄像头，可通过默认单元测试命令运行。
"""

from __future__ import annotations

import numpy as np

from ec_26_e.vision import Detect1Result, PieceDetection, VisionDiagnostics
from ec_26_e.vision.drawing import (
    _drawing_dimensions,
    draw_vision_detection,
)


def _result() -> Detect1Result:
    """构造一个不依赖检测器的四边形结果。"""

    corners = ((100.0, 100.0), (200.0, 100.0), (200.0, 180.0), (100.0, 180.0))
    piece = PieceDetection(
        group_id=0,
        corners_px=corners,
        center_px=(150.0, 140.0),
        corners_relative=tuple((x / 764.0, y / 540.0) for x, y in corners),
        center_relative=(150.0 / 764.0, 140.0 / 540.0),
        area_px=8000.0,
        area_relative=8000.0 / (764.0 * 540.0),
        polygon_iou=1.0,
        line_residual=0.0,
        confidence=0.9,
    )
    return Detect1Result((764, 540), (piece,), 0.15, 764 * 540)


def test_a4_preview_uses_thin_lines_and_small_vertices() -> None:
    """常用 A4 预览尺寸应使用 1 px 线宽和 2 px 顶点。"""

    assert _drawing_dimensions((540, 764, 3)) == (1, 2)
    assert _drawing_dimensions((2160, 3056, 3)) == (2, 4)


def test_draw_vision_does_not_modify_source_image() -> None:
    """绘制分支只能修改副本，不得污染模型输入。"""

    image = np.zeros((540, 764, 3), dtype=np.uint8)
    source = image.copy()
    drawn = draw_vision_detection(image, _result(), VisionDiagnostics())

    assert np.array_equal(image, source)
    assert not np.array_equal(drawn, source)
