"""绘制 YOLO 粗 mask、精修边、交点、稳定结果和耗时诊断。"""

from __future__ import annotations

import cv2
import numpy as np

from ec_26_e.vision.contracts import Detect1Result, VisionDiagnostics


_COLORS = ((50, 210, 255), (255, 120, 40), (90, 230, 90), (220, 90, 220))
_BASE_SHORT_SIDE_PX = 540
_MIN_LINE_THICKNESS = 1
_MAX_LINE_THICKNESS = 2
_MIN_VERTEX_RADIUS = 2
_MAX_VERTEX_RADIUS = 4


def _drawing_dimensions(image_shape: tuple[int, ...]) -> tuple[int, int]:
    """按画面短边返回紧凑的线宽和顶点半径。"""

    short_side = min(image_shape[:2])
    scale = short_side / _BASE_SHORT_SIDE_PX
    thickness = int(np.clip(round(scale), _MIN_LINE_THICKNESS, _MAX_LINE_THICKNESS))
    radius = int(np.clip(round(2.0 * scale), _MIN_VERTEX_RADIUS, _MAX_VERTEX_RADIUS))
    return thickness, radius


def _draw_result(
    canvas: np.ndarray,
    result: Detect1Result,
    color: tuple[int, int, int],
    label: str,
    thickness: int,
    vertex_radius: int,
) -> None:
    """绘制一组多边形边和直线求交顶点。"""

    for piece in result.pieces:
        points = np.rint(piece.corners_px).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], True, color, thickness, cv2.LINE_AA)
        for index, (x, y) in enumerate(piece.corners_px):
            cv2.circle(canvas, (round(x), round(y)), vertex_radius, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, str(index), (round(x) + 4, round(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{label}{piece.group_id} {piece.confidence:.2f}",
            (round(piece.center_px[0]) - 35, round(piece.center_px[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_vision_detection(
    image_bgr: np.ndarray,
    result: Detect1Result,
    diagnostics: VisionDiagnostics,
    stable_result: Detect1Result | None = None,
) -> np.ndarray:
    """返回包含粗 mask、单帧精修、稳定结果和耗时的图像副本。"""

    canvas = image_bgr.copy()
    line_thickness, vertex_radius = _drawing_dimensions(canvas.shape)
    overlay = canvas.copy()
    for index, item in enumerate(diagnostics.instances):
        color = _COLORS[index % len(_COLORS)]
        overlay[item.coarse_mask > 0] = color
    cv2.addWeighted(overlay, 0.24, canvas, 0.76, 0.0, canvas)
    _draw_result(canvas, result, (0, 255, 255), "R-", line_thickness, vertex_radius)
    if stable_result is not None:
        _draw_result(canvas, stable_result, (40, 255, 40), "S-", line_thickness, vertex_radius)
    timing = diagnostics.stage_ms
    status = "GLARE FP" if diagnostics.yolo_count and not result.pieces else ""
    lines = [
        f"YOLO {diagnostics.yolo_count}  refined {len(result.pieces)}  stable {len(stable_result.pieces) if stable_result else 0} {status}",
        f"NPU {diagnostics.native_timing.network_ms:.1f} ms  masks {diagnostics.native_timing.mask_decode_ms:.1f} ms",
        f"foreground {timing.get('traditional_foreground', 0):.1f} ms  refine {timing.get('traditional_refine', 0):.1f} ms  total {timing.get('total', 0):.1f} ms",
    ]
    for index, text in enumerate(lines):
        y = 22 + index * 22
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas
