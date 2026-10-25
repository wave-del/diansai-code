#!/usr/bin/env python3
"""拼图装置上位机：固定拼图优先，随机两问预留入口。

界面线程只负责触摸交互与低帧率相机预览。实际视觉/规划/硬件流程仍在独立
子进程中运行，从而保持与已实测 ``run_task1_tangram.py`` 链路一致。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QProcess, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HANDOVER_DIR = ROOT / "handover-puzzle"
if str(HANDOVER_DIR) not in sys.path:
    sys.path.insert(0, str(HANDOVER_DIR))
from q6a_platform import (VISION_UNAVAILABLE, add_mode_arguments, load_config, validate_motor_port)

VENV_PYTHON = Path(sys.executable)
RUNNER = ROOT / "handover-puzzle" / "run_task1_tangram.py"
TASK_ENV = ROOT / "handover-puzzle" / "task1.tangram.env"
SAFE_PARKING = HERE / "safe_parking.json"
SETTINGS_PATH = HERE / "ui_settings.json"
DEFAULT_WRITER_PORT = ""

BUTTONS = (
    (1, "第 1 问·第 2 小问", "固定四片拼图", True),
    (2, "第 2 问·第 1 小问", "白色随机碎片（自动识别 2～4 片）", True),
    (3, "第 2 问·第 2 小问", "扑克牌图案碎片（外轮廓）", True),
)

TASK_PROFILES = {
    1: "fixed",
    2: "random-white",
    3: "random-patterned",
}

# 单帧 NPU+精修实测约 0.4 秒；3 FPS 已接近实际吞吐上限。任务开始时预览会
# 完全停止，因此该值只影响等待按键期间的检测叠加刷新。
PREVIEW_DETECTION_FPS = 3.0
COMPETITION_TIME_LIMIT_SECONDS = 120.0
CAMERA_DISCOVERY_RETRY_MS = 2000
# USB 摄像头热插拔后，旧 V4L2 文件描述符可能仍可读但永远不再产生帧。
# 不能让界面永久停留在最后一帧；超过此时间即关闭旧句柄并重新发现设备。
CAMERA_FRAME_STALL_SECONDS = 4.0
# VNC 常见为 1280 宽，而现场显示器通常更宽。布局依据真实可用宽度切换，
# 不依赖 DISPLAY 名称或设备类型。
COMPACT_LAYOUT_MAX_WIDTH = 1599
# 仅空闲预览使用。正式任务仍通过 ``test_a4_detection.PARAM_PRESETS`` 的
# current/new 参数运行，绝不会读取本表。
PREVIEW_A4_PROFILES: dict[str, dict[str, int | float]] = {
    "adaptive": {
        # 仅作为兼容旧设置的默认值；实际自适应预览会遍历下面多个候选，
        # 不会把它们合并为同一个会吞入背景的大 mask。
        "hue_min": 90,
        "hue_max": 130,
        "sat_min": 160,
        "value_min": 80,
        "min_a4_area": 0.20,
        "max_refine_shift": 70.0,
    },
}
# 调试用蓝纸候选。每组独立构建 mask 并通过完整矩形结构验证；这样较浅蓝纸
# 可以落入后续候选，而不会与偏蓝机架、阴影合并成一个轮廓。
PREVIEW_A4_CANDIDATES: tuple[tuple[str, dict[str, int | float]], ...] = (
    ("高饱和蓝", PREVIEW_A4_PROFILES["adaptive"]),
    (
        "中等饱和亮蓝",
        {
            "hue_min": 90,
            "hue_max": 132,
            "sat_min": 135,
            "value_min": 115,
            "min_a4_area": 0.20,
            "max_refine_shift": 70.0,
        },
    ),
    (
        "浅亮蓝",
        {
            "hue_min": 88,
            "hue_max": 135,
            "sat_min": 95,
            "value_min": 150,
            "min_a4_area": 0.20,
            "max_refine_shift": 70.0,
        },
    ),
)


def read_env_camera() -> tuple[str, int, int, int]:
    values: dict[str, str] = {}
    for line in TASK_ENV.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return (
        values.get("CAMERA_DEVICE_PATH", "/dev/video0"),
        int(values.get("CAMERA_WIDTH", "1280")),
        int(values.get("CAMERA_HEIGHT", "720")),
        int(values.get("CAMERA_FPS", "30")),
    )


def _preview_a4_args(values: dict[str, int | float]) -> argparse.Namespace:
    """把一个仅预览的 HSV 候选转换为 a4_robust 所需参数。"""

    return argparse.Namespace(
        hue_min=int(values["hue_min"]),
        hue_max=int(values["hue_max"]),
        sat_min=int(values["sat_min"]),
        value_min=int(values["value_min"]),
        min_a4_area=float(values["min_a4_area"]),
        max_refine_shift=float(values["max_refine_shift"]),
    )


def _preview_a4_candidate_args(
    formal_params: dict[str, int | float],
    profile: str,
) -> list[tuple[str, argparse.Namespace]]:
    """返回预览候选，不会修改正式任务使用的参数字典。"""

    if profile == "formal":
        return [
            (
                "正式参数",
                _preview_a4_args(
                    {
                        "hue_min": formal_params["hue_min"],
                        "hue_max": formal_params["hue_max"],
                        "sat_min": formal_params["sat_min"],
                        "value_min": formal_params["value_min"],
                        "min_a4_area": 0.25,
                        "max_refine_shift": 35.0,
                    }
                ),
            )
        ]
    return [(name, _preview_a4_args(values)) for name, values in PREVIEW_A4_CANDIDATES]


def _best_effort_preview_a4(
    frame: np.ndarray,
    candidates: list[tuple[str, argparse.Namespace]],
) -> np.ndarray | None:
    """严格 A4 seed 都失败时，仅预览用的橙色矩形候选。

    不参与透视校正、拼图检测或任何正式执行，仅避免调试界面在困难画面中完全
    没有 A4 反馈。
    """

    source_height, source_width = frame.shape[:2]
    detect_width, detect_height = 640, 360
    image = cv2.resize(frame, (detect_width, detect_height), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    best: tuple[float, np.ndarray] | None = None
    for _name, args in candidates:
        lower = np.asarray((args.hue_min, args.sat_min, args.value_min), dtype=np.uint8)
        upper = np.asarray((args.hue_max, 255, 255), dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area_ratio = cv2.contourArea(contour) / float(detect_width * detect_height)
            if area_ratio < 0.08:
                continue
            rectangle = cv2.minAreaRect(contour)
            width, height = rectangle[1]
            if min(width, height) <= 1.0:
                continue
            aspect = max(width, height) / min(width, height)
            # A4 实物比例约 1.414；透视下只作软排序，不把候选直接拒掉。
            score = area_ratio / (1.0 + abs(float(np.log(aspect / 1.414))))
            corners = cv2.boxPoints(rectangle).astype(np.float32)
            if best is None or score > best[0]:
                best = score, corners
    if best is None:
        return None
    scale = np.asarray(
        [(source_width - 1) / (detect_width - 1), (source_height - 1) / (detect_height - 1)],
        dtype=np.float32,
    )
    return best[1] * scale


def load_parking() -> tuple[float, float]:
    raw = json.loads(SAFE_PARKING.read_text(encoding="utf-8"))
    x, y = raw["blue_paper_xy_mm"]
    return float(x), float(y)


def render_content_flipped_solution(image_path: Path) -> Path:
    """仅上下镜像最终执行坐标，并清晰标注实际拼缝状态。"""

    solution_path = image_path.with_name("solution.json")
    solution = json.loads(solution_path.read_text(encoding="utf-8"))
    moves = solution.get("moves")
    if not isinstance(moves, list) or not moves:
        raise ValueError("solution.json 中没有可显示的拼图块")
    polygons: list[tuple[str, list[tuple[float, float]]]] = []
    for move in moves:
        if not isinstance(move, dict):
            raise ValueError("solution.json 的 move 无效")
        raw_vertices = move.get(
            "planned_final_vertices_a4_mm",
            move.get("target_vertices_a4_mm", []),
        )
        vertices = [(float(x), float(y)) for x, y in raw_vertices]
        if len(vertices) < 3:
            raise ValueError("拼图块顶点不足 3 个")
        polygons.append((str(move.get("piece_id", "?")), vertices))
    min_y = min(y for _piece_id, vertices in polygons for _x, y in vertices)
    max_y = max(y for _piece_id, vertices in polygons for _x, y in vertices)
    mirror_y = min_y + max_y

    # 仅改变各拼图块的 Y 坐标；必须展示 clearance 后的最终 MovePlan 坐标，
    # 不以半透明无描边的理论图掩盖实际拼缝或潜在重叠。
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PolygonPatch

    clearance = solution.get("clearance", {})
    clearance_ok = bool(clearance.get("feasible", False))
    clearance_applied = bool(clearance.get("applied", clearance_ok))
    minimum_gap = float(clearance.get("minimum_gap_mm", 0.0))
    maximum_vertex = float(clearance.get(
        "max_vertex_distance_mm",
        solution.get("nominal_max_vertex_distance_mm", 0.0),
    ))
    overlap = float(clearance.get("overlap_mm2", 0.0))
    figure, axes = plt.subplots(figsize=(7, 5))
    figure.patch.set_facecolor("#111827")
    axes.set_facecolor("#1f2937")
    for index, (piece_id, vertices) in enumerate(polygons):
        flipped_vertices = [(x, mirror_y - y) for x, y in vertices]
        axes.add_patch(
            PolygonPatch(
                flipped_vertices,
                closed=True,
                alpha=0.82,
                facecolor=plt.get_cmap("tab10")(index),
                edgecolor="#f8fafc",
                linewidth=1.6,
                label=piece_id,
            )
        )
    axes.set_aspect("equal")
    axes.autoscale()
    axes.grid(alpha=0.24, color="#94a3b8")
    axes.legend(loc="best")
    for spine in axes.spines.values():
        spine.set_color("#cbd5e1")
    axes.tick_params(colors="#e2e8f0")
    if clearance_ok:
        title = (
            "EXECUTION PREVIEW — clearance OK\n"
            f"min gap={minimum_gap:.2f} mm | vertex≤{maximum_vertex:.2f} mm | "
            f"overlap≈{overlap:.2f} mm²"
        )
    elif clearance_applied:
        title = (
            "EXECUTION PREVIEW — narrow-gap fallback (no overlap)\n"
            f"min gap={minimum_gap:.2f} mm | vertex≤{maximum_vertex:.2f} mm | "
            f"overlap≈{overlap:.2f} mm²"
        )
    else:
        title = "EXECUTION PREVIEW — WARNING: clearance unavailable"
    axes.set_title(title, color="#f8fafc", fontsize=10, fontweight="bold", pad=10)
    figure.tight_layout()
    output_path = image_path.with_name("solution_after_display_vertical_flip.png")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def draw_enhanced_preview(
    frame: np.ndarray,
    result: Any | None,
    status: str,
    best_effort_a4: np.ndarray | None = None,
) -> np.ndarray:
    """在原始相机帧叠加鲁棒 A4、精修 A4 与拼图轮廓。"""

    annotated = frame.copy()
    if result is None and best_effort_a4 is not None:
        corners = np.rint(best_effort_a4).astype(np.int32)
        cv2.polylines(
            annotated, [corners.reshape(-1, 1, 2)], True, (0, 165, 255), 3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            "A4 BEST EFFORT | NOT EXECUTABLE",
            tuple(corners[0] + np.asarray((8, 24))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
    if result is not None:
        seed = result.seed
        for index, corner in enumerate(seed.corners):
            color = (0, 165, 255) if not seed.reliability[index] else (255, 0, 255)
            cv2.circle(annotated, tuple(np.rint(corner).astype(int)), 7, color, -1, cv2.LINE_AA)
        cv2.polylines(
            annotated,
            [np.rint(seed.corners).astype(np.int32).reshape(-1, 1, 2)],
            True,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
        refined = np.asarray(result.resize_result.source_corners_px, dtype=np.float32)
        cv2.polylines(
            annotated,
            [np.rint(refined).astype(np.int32).reshape(-1, 1, 2)],
            True,
            (255, 255, 0),
            3,
            cv2.LINE_AA,
        )
        inverse = np.linalg.inv(result.resize_result.homography)
        width, height = result.detect_result.image_size
        divider = np.asarray(
            [[width * 0.52, 0.0], [width * 0.52, height - 1.0]],
            dtype=np.float32,
        ).reshape(1, -1, 2)
        divider_source = cv2.perspectiveTransform(divider, inverse)[0]
        cv2.line(
            annotated,
            tuple(np.rint(divider_source[0]).astype(int)),
            tuple(np.rint(divider_source[1]).astype(int)),
            (0, 255, 255), 2, cv2.LINE_AA,
        )
        display = result.detect_result if result.detect_result.pieces else result.single_result
        if display is not None and display.pieces:
            for piece in display.pieces:
                points = np.asarray(piece.corners_px, dtype=np.float32).reshape(1, -1, 2)
                source = cv2.perspectiveTransform(points, inverse)[0]
                center = cv2.perspectiveTransform(
                    np.asarray(piece.center_px, dtype=np.float32).reshape(1, 1, 2),
                    inverse,
                )[0, 0]
                cv2.polylines(
                    annotated,
                    [np.rint(source).astype(np.int32).reshape(-1, 1, 2)],
                    True,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.circle(annotated, tuple(np.rint(center).astype(int)), 6, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    annotated,
                    f"LEFT {piece.group_id}",
                    tuple(np.rint(center + np.asarray((9, -9))).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
        for piece in result.rejected_pieces:
            points = np.asarray(piece.corners_px, dtype=np.float32).reshape(1, -1, 2)
            source = cv2.perspectiveTransform(points, inverse)[0]
            center = cv2.perspectiveTransform(
                np.asarray(piece.center_px, dtype=np.float32).reshape(1, 1, 2), inverse,
            )[0, 0]
            cv2.polylines(
                annotated,
                [np.rint(source).astype(np.int32).reshape(-1, 1, 2)],
                True, (0, 0, 255), 3, cv2.LINE_AA,
            )
            cv2.putText(
                annotated, "IGNORED RIGHT",
                tuple(np.rint(center + np.asarray((9, -9))).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA,
            )
    color = (0, 255, 0) if status.startswith("OK") else (0, 165, 255)
    cv2.putText(annotated, status[:120], (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    cv2.putText(
        annotated,
        "green=LEFT PUZZLE | red=IGNORED RIGHT | yellow=52% boundary",
        (20, max(65, annotated.shape[0] - 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def _frozen_points(
    overlay: dict[str, Any],
    value: Any,
    frame: np.ndarray,
) -> np.ndarray | None:
    """按当前相机分辨率缩放一次冻结结果中的原始像素坐标。"""

    try:
        points = np.asarray(value, dtype=float).reshape(-1, 2)
        source_width, source_height = overlay["source_image_size"]
        if source_width <= 0 or source_height <= 0 or len(points) == 0:
            return None
        scale = np.asarray(
            [frame.shape[1] / float(source_width), frame.shape[0] / float(source_height)]
        )
        return points * scale
    except (KeyError, TypeError, ValueError):
        return None


def draw_frozen_preview(
    frame: np.ndarray, overlay: dict[str, Any] | None, status: str
) -> np.ndarray:
    """显示实时原图和已经冻结的 YOLO 结果，不进行任何新的推理。"""

    annotated = frame.copy()
    if overlay is not None:
        seed = _frozen_points(overlay, overlay.get("seed_corners_px"), frame)
        reliability = overlay.get("seed_reliability", ())
        if seed is not None:
            for index, corner in enumerate(seed):
                reliable = index < len(reliability) and bool(reliability[index])
                color = (255, 0, 255) if reliable else (0, 165, 255)
                cv2.circle(
                    annotated, tuple(np.rint(corner).astype(int)), 7, color, -1,
                    cv2.LINE_AA,
                )
            cv2.polylines(
                annotated,
                [np.rint(seed).astype(np.int32).reshape(-1, 1, 2)],
                True, (255, 0, 255), 2, cv2.LINE_AA,
            )
        refined = _frozen_points(
            overlay, overlay.get("refined_a4_corners_px"), frame
        )
        if refined is not None:
            cv2.polylines(
                annotated,
                [np.rint(refined).astype(np.int32).reshape(-1, 1, 2)],
                True, (255, 255, 0), 3, cv2.LINE_AA,
            )
        divider = _frozen_points(
            overlay, overlay.get("divider_source_px"), frame
        )
        if divider is not None and len(divider) == 2:
            cv2.line(
                annotated,
                tuple(np.rint(divider[0]).astype(int)),
                tuple(np.rint(divider[1]).astype(int)),
                (0, 255, 255), 2, cv2.LINE_AA,
            )
        for key, color, label in (
            ("accepted_pieces", (0, 255, 0), "FROZEN LEFT"),
            ("rejected_pieces", (0, 0, 255), "FROZEN IGNORED RIGHT"),
        ):
            for piece in overlay.get(key, []):
                if not isinstance(piece, dict):
                    continue
                corners = _frozen_points(overlay, piece.get("corners_px"), frame)
                center = _frozen_points(overlay, piece.get("center_px"), frame)
                if corners is not None:
                    cv2.polylines(
                        annotated,
                        [np.rint(corners).astype(np.int32).reshape(-1, 1, 2)],
                        True, color, 3, cv2.LINE_AA,
                    )
                if center is not None and len(center):
                    point = center[0]
                    cv2.circle(
                        annotated, tuple(np.rint(point).astype(int)), 6,
                        (0, 0, 255), -1, cv2.LINE_AA,
                    )
                    cv2.putText(
                        annotated,
                        f"{label} {piece.get('piece_id', '?')}",
                        tuple(np.rint(point + np.asarray((9, -9))).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
                    )
    cv2.putText(
        annotated, status[:120], (20, 36), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        "frozen YOLO overlay | raw live camera | YOLO OFF",
        (20, max(65, annotated.shape[0] - 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return annotated


def draw_raw_preview(frame: np.ndarray, status: str) -> np.ndarray:
    """显示任务执行期间的轻量原始预览。"""

    annotated = frame.copy()
    cv2.putText(
        annotated, status[:120], (20, 36), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return annotated


class PreviewThread(QThread):
    """空闲 YOLO 或任务期间轻量原始相机预览。"""

    frame_ready = Signal(QImage)
    failed = Signal(str)
    notice = Signal(str)
    vision_state = Signal(int, int, bool, str)

    def __init__(
        self,
        camera: str,
        width: int,
        height: int,
        fps: int,
        params_name: str,
        preview_a4_profile: str,
        detection_fps: float = PREVIEW_DETECTION_FPS,
        preview_mode: str = "live_yolo",
        frozen_overlay: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.camera = camera
        self.width = width
        self.height = height
        self.fps = fps
        self.params_name = params_name
        if preview_a4_profile not in {"formal", *PREVIEW_A4_PROFILES}:
            raise ValueError(f"未知预览蓝纸参数：{preview_a4_profile}")
        self.preview_a4_profile = preview_a4_profile
        self.detection_fps = detection_fps
        if preview_mode not in {"live_yolo", "frozen_overlay", "raw"}:
            raise ValueError(f"未知预览模式：{preview_mode}")
        self.preview_mode = preview_mode
        self._frozen_overlay = frozen_overlay
        self._overlay_lock = threading.Lock()
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def clear_frozen_overlay(self) -> None:
        """机械开始动作后停止绘制静态框，但保持相机预览不断流。"""

        with self._overlay_lock:
            self._frozen_overlay = None
            self.preview_mode = "raw"

    def run(self) -> None:
        try:
            handover_dir = str(ROOT / "handover-puzzle")
            if handover_dir not in sys.path:
                sys.path.insert(0, handover_dir)
            from hardware_discovery import open_camera_with_fallback

            capture, actual_camera = open_camera_with_fallback(self.camera, cv2)
        except Exception as error:
            self.failed.emit(f"无法打开相机：{error}")
            return
        if actual_camera != self.camera:
            message = (
                f"预览相机已从配置 {self.camera} 自动切换至 {actual_camera}"
            )
            print(message, flush=True)
            self.notice.emit(message)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # 与正式任务的相机打开方式对齐，避免预览使用默认格式时出现偶发的
        # 异常帧，进而让 A4 外推角点落到图像边界之外。
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        frames = _LatestFrameReader(capture)
        frames.start()
        detector: Any | None = None
        latest: Any | None = None
        best_effort_a4: np.ndarray | None = None
        status = "VISION LOADING..."
        next_process = -float("inf")
        last_frame_sequence = -1
        active_a4_candidate: str | None = None
        last_detection_notice: str | None = None
        preview_a4_candidates: list[tuple[str, argparse.Namespace]] = []
        seed_a4_robust: Any | None = None
        try:
            if self.preview_mode == "live_yolo":
                try:
                    handover_dir = str(ROOT / "handover-puzzle")
                    if handover_dir not in sys.path:
                        sys.path.insert(0, handover_dir)
                    from task1_yolo_flow import _prepare_import_paths

                    _prepare_import_paths()
                    import test_a4_detection as a4_detection
                    from seed import seed_a4_robust as seed_a4_robust_function
                    from yolo_pipeline import YoloRobustDetector

                    preset_args = argparse.Namespace(
                        params=self.params_name,
                        hue_min=None,
                        hue_max=None,
                        sat_min=None,
                        value_min=None,
                    )
                    formal_params = a4_detection._resolve_params(preset_args)
                    preview_a4_candidates = _preview_a4_candidate_args(
                        formal_params, self.preview_a4_profile
                    )
                    if self.preview_a4_profile != "formal":
                        self.notice.emit(
                            "预览蓝纸参数：多候选自适应（仅预览）；"
                            "正式任务仍使用正式视觉预设。"
                        )
                    seed_a4_robust = seed_a4_robust_function
                    # 白色碎片检测继续使用正式预设；只有蓝纸 seed 使用独立
                    # 的预览参数，避免调试卡纸污染正式碎片识别参数。
                    detector = YoloRobustDetector(
                        formal_params, preview_a4_candidates[0][1], stable=True
                    )
                    status = "OK: VISION READY, WAITING FOR STABLE PIECES"
                except Exception as error:
                    status = "RAW CAMERA ONLY: VISION UNAVAILABLE"
                    self.notice.emit(f"增强实时预览初始化失败，已降级为原始相机：{error}")
            elif self.preview_mode == "frozen_overlay":
                status = "FROZEN YOLO RESULT | RAW LIVE CAMERA | YOLO OFF"
                self.vision_state.emit(0, 0, False, "冻结识别结果叠加中；未运行 YOLO")
            else:
                status = "RAW LIVE CAMERA | MECHANISM MOVING | YOLO OFF"
                self.vision_state.emit(0, 0, False, "机械执行中：原始实时预览")
            while not self._stop.is_set():
                sequence, frame = frames.latest()
                if frame is None or sequence == last_frame_sequence:
                    if frames.frame_age_seconds() >= CAMERA_FRAME_STALL_SECONDS:
                        self.failed.emit(
                            "相机连续 "
                            f"{frames.frame_age_seconds():.1f} 秒没有新画面；"
                            "可能已热插拔或 V4L2 句柄失效，正在重新打开。"
                        )
                        return
                    self.msleep(10)
                    continue
                last_frame_sequence = sequence
                now = time.monotonic()
                if detector is not None and now >= next_process:
                    next_process = now + 1.0 / self.detection_fps
                    try:
                        valid_candidates: list[tuple[str, argparse.Namespace, Any]] = []
                        assert seed_a4_robust is not None
                        for name, a4_args in preview_a4_candidates:
                            try:
                                seed = seed_a4_robust(frame, a4_args)
                            except ValueError:
                                continue
                            valid_candidates.append((name, a4_args, seed))
                        if not valid_candidates:
                            latest = None
                            active_a4_candidate = None
                            best_effort_a4 = _best_effort_preview_a4(
                                frame, preview_a4_candidates
                            )
                            status = "A4 BEST EFFORT | GEOMETRY UNVERIFIED"
                            detail = (
                                "未通过严格 A4 矩形门；已显示橙色最佳努力候选"
                                if best_effort_a4 is not None
                                else "未找到可显示的 A4 候选，继续尝试多组蓝纸参数"
                            )
                            self.vision_state.emit(0, 0, False, detail)
                            if detail != last_detection_notice:
                                self.notice.emit(detail)
                                last_detection_notice = detail
                        else:
                            selected_name, selected_args, _selected_seed = max(
                                valid_candidates,
                                key=lambda item: (
                                    sum(bool(value) for value in item[2].reliability),
                                    -float(item[2].aspect_error),
                                    float(item[2].area_ratio),
                                ),
                            )
                            if selected_name != active_a4_candidate:
                                # YoloRobustDetector 的 seed 参数只在预览线程中切换；
                                # 正式任务的实例和参数完全不受影响。切换时清空五帧
                                # 稳定器，避免两套透视坐标混合。
                                detector._a4_args = selected_args
                                detector.reset()
                                active_a4_candidate = selected_name
                                self.notice.emit(f"预览 A4 候选已锁定：{selected_name}")
                            latest = detector.detect(frame)
                            best_effort_a4 = None
                            last_detection_notice = None
                            stable_count = len(latest.detect_result.pieces)
                            single_count = (
                                len(latest.single_result.pieces)
                                if latest.single_result is not None
                                else 0
                            )
                            status = f"OK: A4 LOCKED | STABLE={stable_count} | SINGLE={single_count}"
                            ready = 2 <= stable_count <= 4
                            self.vision_state.emit(
                                stable_count,
                                single_count,
                                ready,
                                "已完成连续五帧稳定验证" if ready else "正在收集连续五帧稳定结果",
                            )
                    except Exception as error:
                        latest = None
                        best_effort_a4 = _best_effort_preview_a4(
                            frame, preview_a4_candidates
                        )
                        status = "VISION ERROR: SEE STATUS LOG"
                        self.vision_state.emit(0, 0, False, "蓝纸或拼图检测暂不可用")
                        message = f"增强实时预览检测失败：{error}"
                        if message != last_detection_notice:
                            self.notice.emit(message)
                            last_detection_notice = message
                if detector is not None:
                    display = draw_enhanced_preview(
                        frame, latest, status, best_effort_a4
                    )
                else:
                    with self._overlay_lock:
                        frozen_overlay = self._frozen_overlay
                        preview_mode = self.preview_mode
                    display = (
                        draw_frozen_preview(frame, frozen_overlay, status)
                        if preview_mode == "frozen_overlay"
                        else draw_raw_preview(
                            frame,
                            "RAW LIVE CAMERA | MECHANISM MOVING | YOLO OFF",
                        )
                    )
                image = QImage(
                    display.data,
                    display.shape[1],
                    display.shape[0],
                    display.strides[0],
                    QImage.Format.Format_BGR888,
                ).copy()
                self.frame_ready.emit(image)
                self.msleep(100)  # 原始帧可更快显示，模型推理仍限制为约 2 FPS。
        finally:
            frames.close()
            if detector is not None:
                try:
                    detector.close()
                except Exception as error:
                    self.notice.emit(f"关闭增强视觉资源失败：{error}")


class _LatestFrameReader:
    """持续清空 V4L2 队列，只向推理和界面提供最新一帧。"""

    def __init__(self, capture: Any) -> None:
        self._capture = capture
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._sequence = 0
        self._last_frame_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, name="tangram-camera-latest", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest = frame
                self._sequence += 1
                self._last_frame_at = time.monotonic()

    def latest(self) -> tuple[int, np.ndarray | None]:
        with self._lock:
            return self._sequence, self._latest

    def frame_age_seconds(self) -> float:
        """最后一帧距现在的时长；供预览热插拔看门狗使用。"""

        with self._lock:
            return max(0.0, time.monotonic() - self._last_frame_at)

    def close(self) -> None:
        """Stop the reader before releasing its OpenCV capture.

        ``VideoCapture.release()`` must not race a simultaneous ``read()`` in
        another Python thread: some V4L2/OpenCV backends release native buffers
        immediately and can abort the whole Qt process.  At normal camera frame
        rates the reader observes ``_stop`` after at most one more frame, so
        joining it first is both safe and quick.
        """
        self._stop.set()
        self._thread.join(2.0)
        if self._thread.is_alive():
            # A stalled USB camera is exceptional.  Do not silently leave its
            # native handle open, but keep this last-resort release separate
            # from the normal stop path above.
            self._capture.release()
            self._thread.join(0.5)
            return
        self._capture.release()


class ParameterSlider(QWidget):
    """带实时数值读数的定步长滑动参数控件。"""

    valueChanged = Signal(float)

    def __init__(
        self,
        low: float,
        high: float,
        step: float,
        value: float,
        *,
        suffix: str = "",
    ) -> None:
        super().__init__()
        if step <= 0.0:
            raise ValueError("滑动参数步长必须为正数")
        if high < low:
            raise ValueError("滑动参数上限不能小于下限")
        # QSlider 的内部值必须是整数。此前用 ``round(1 / step)`` 转换，
        # 在 XY/Z 这类 100 mm/min 步长时会得到 0，导致上位机启动即崩溃。
        # 改为“第几个步长”的整数索引，任意正步长均可安全表示。
        self._low = float(low)
        self._step = float(step)
        self._steps = max(0, round((float(high) - self._low) / self._step))
        self._suffix = suffix
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, self._steps)
        self._slider.setSingleStep(1)
        self._value_label = QLabel()
        self._value_label.setMinimumWidth(74)
        self._value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._value_label)
        self._slider.valueChanged.connect(self._on_value_changed)
        self.setValue(value)

    def value(self) -> float:
        return self._low + self._slider.value() * self._step

    def setValue(self, value: float) -> None:
        index = round((float(value) - self._low) / self._step)
        self._slider.setValue(max(0, min(self._steps, index)))

    def _on_value_changed(self, _raw: int) -> None:
        value = self.value()
        self._value_label.setText(f"{value:g}{self._suffix}")
        self.valueChanged.emit(value)


class MainWindow(QMainWindow):
    LAST_PIECE_RELEASED_SIGNAL = "拼图完成信号：最后一片已释放。"

    def __init__(self, *, hardware: bool = False) -> None:
        super().__init__()
        self._hardware = hardware
        self._preview = None
        self._q6a_config = load_config()
        self.setWindowTitle("拼图装置上位机")
        self.resize(1280, 800)
        self._image: QImage | None = None
        self._solution_image: QImage | None = None
        self._runner: QProcess | None = None
        self._run_sent = False
        self._completion_signaled = False
        self._stopping = False
        self._terminal_interrupt_requested = False
        # 只在本次上位机会话内保持。应用重启后必须重新按“首次任务”流程
        # 建立原点，避免把安全停靠点误定义成机械零点。
        self._machine_origin_established = False
        self._runner_kind = ""
        self._preview_stable_pieces = 0
        self._preview_single_pieces = 0
        self._preview_vision_ready = False
        self._preview_vision_detail = "正在启动相机与稳定视觉…"
        self._frozen_overlay: dict[str, Any] | None = None
        self._competition_started_at: float | None = None
        self._competition_elapsed_seconds = 0.0
        self._competition_timer = QTimer(self)
        self._competition_timer.setInterval(100)
        self._competition_timer.timeout.connect(self._update_competition_timer)
        self._settings = self._load_settings()
        configured_camera, self._camera_width, self._camera_height, self._camera_fps = (
            read_env_camera()
        )
        # 只保存稳定 by-id 路径或用户明确选择的节点。绝不把一次临时分配的
        # ttyUSB0/video0 当作唯一真相；每次启动仍会重新枚举并校验其存在性。
        self._camera = str(self._q6a_config["camera_path"] or self._settings.get("camera_path", ""))
        self._writer_port = str(
            self._q6a_config["writer_port"] or self._settings.get("writer_port", DEFAULT_WRITER_PORT)
        )
        self._refreshing_devices = False
        self._task_buttons: dict[int, QPushButton] = {}
        self._compact_layout: bool | None = None
        self._build_ui()
        self._preview: PreviewThread | None = None
        # 相机可在上位机已启动后才接入。失败时只重试相机发现，不访问任何
        # 运动硬件或写字机串口。
        self._preview_retry_timer = QTimer(self)
        self._preview_retry_timer.setSingleShot(True)
        self._preview_retry_timer.setInterval(CAMERA_DISCOVERY_RETRY_MS)
        self._preview_retry_timer.timeout.connect(self._retry_preview)
        self._preview_retry_attempt = 0
        self._preview_retry_mode = "live_yolo"
        self._preview_retry_overlay: dict[str, Any] | None = None
        self._refresh_hardware_devices(restart_preview=False, announce=False)
        self._start_preview()
        self._update_origin_status()
        self._set_running_widgets(False)
        self._update_vision_readiness()
        self._set_status("硬件模式" if self._hardware else "无外设模式",
                         "Q6A HTP 支持本地图片验证；摄像头、写字机待接入。")

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        left = QVBoxLayout()
        visual_layout = QHBoxLayout()
        visual_layout.setSpacing(12)

        preview_box = QGroupBox("实时检测画面")
        preview_layout = QVBoxLayout(preview_box)
        self.preview_label = QLabel("正在启动相机预览…")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 右侧状态栏固定保留宽度；左侧双画面中实时检测占更大比例。
        self.preview_label.setMinimumSize(760, 500)
        self.preview_label.setStyleSheet("background:#000; color:#fff; border:2px solid #000;")
        preview_layout.addWidget(self.preview_label, 1)
        self.preview_caption = QLabel("实时检测：相机预览")
        self.preview_caption.setWordWrap(True)
        preview_layout.addWidget(self.preview_caption)
        self.vision_readiness = QLabel()
        self.vision_readiness.setWordWrap(True)
        self.vision_readiness.setStyleSheet(
            "font-size:17px; font-weight:700; padding:8px; "
            "background:#fff; color:#000; border:2px solid #000;"
        )
        preview_layout.addWidget(self.vision_readiness)
        self._update_vision_readiness()
        visual_layout.addWidget(preview_box, 2)

        solution_box = QGroupBox("本次求解结果")
        solution_layout = QVBoxLayout(solution_box)
        self.solution_label = QLabel("尚无求解结果\n求解完成后显示在这里")
        self.solution_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.solution_label.setWordWrap(True)
        self.solution_label.setMinimumSize(340, 260)
        self.solution_label.setStyleSheet(
            "background:#fff; color:#000; border:2px solid #000;"
        )
        solution_layout.addWidget(self.solution_label, 1)
        self.solution_caption = QLabel("结果区：等待本次求解")
        self.solution_caption.setWordWrap(True)
        solution_layout.addWidget(self.solution_caption)
        visual_layout.addWidget(solution_box, 1)
        left.addLayout(visual_layout, 1)
        task_box = QGroupBox("任务选择")
        task_grid = QGridLayout(task_box)
        self.plan_only = QCheckBox("仅检测与解算（硬件隔离：绝不控制电机）")
        self.plan_only.setChecked(True if not self._hardware else bool(self._settings.get("plan_only", True)))
        self.plan_only.toggled.connect(self._plan_only_toggled)
        task_grid.addWidget(self.plan_only, 0, 0, 1, len(BUTTONS))
        for number, title, detail, enabled in BUTTONS:
            button = QPushButton(f"{number}. {title}\n{detail}")
            button.setMinimumHeight(96)
            button.setEnabled(enabled)
            button.setProperty("available", enabled)
            button.clicked.connect(lambda checked=False, n=number: self._start_task(n))
            task_grid.addWidget(button, 1, number - 1)
            self._task_buttons[number] = button
        left.addWidget(task_box)
        layout.addLayout(left, 1)

        sidebar = QScrollArea()
        sidebar.setWidgetResizable(True)
        sidebar.setMinimumWidth(580)
        sidebar.setMaximumWidth(680)
        self._sidebar = sidebar
        side = QWidget()
        side_layout = QVBoxLayout(side)
        self.task_label = QLabel("当前任务：未选择")
        self.task_label.setWordWrap(True)
        self.state_label = QLabel("状态：初始化")
        self.state_label.setWordWrap(True)
        for label in (self.task_label, self.state_label):
            label.setStyleSheet(
                "font-size:18px; font-weight:600; padding:8px; "
                "background:#fff; color:#000; border:2px solid #000;"
            )
            side_layout.addWidget(label)

        timer_box = QGroupBox("本问 2 分钟计时")
        timer_layout = QVBoxLayout(timer_box)
        self.competition_timer_label = QLabel()
        self.competition_timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.competition_timer_label.setStyleSheet(
            "font-size:30px; font-weight:700; padding:10px; "
            "background:#fff; color:#000; border:2px solid #000;"
        )
        timer_layout.addWidget(self.competition_timer_label)
        self.competition_timer_hint = QLabel("尚未启动；每次选择新的题目会清零计时。")
        self.competition_timer_hint.setWordWrap(True)
        timer_layout.addWidget(self.competition_timer_hint)
        side_layout.addWidget(timer_box)
        self._render_competition_timer("尚未启动")

        completion_box = QGroupBox("完成状态")
        completion_layout = QHBoxLayout(completion_box)
        self.completion_light = QLabel()
        self.completion_light.setFixedSize(44, 44)
        completion_layout.addWidget(self.completion_light)
        self.completion_hint = QLabel("红色：未完成 / 运行中\n绿色：最后一片已放下")
        self.completion_hint.setWordWrap(True)
        completion_layout.addWidget(self.completion_hint, 1)
        side_layout.addWidget(completion_box)
        self._set_completion_indicator(False)

        action_box = QGroupBox("安全控制")
        action_layout = QVBoxLayout(action_box)
        self.origin_status = QLabel()
        self.origin_status.setWordWrap(True)
        action_layout.addWidget(self.origin_status)
        self.park_button = QPushButton("回安全停靠点")
        self.park_button.setMinimumHeight(54)
        self.park_button.clicked.connect(self._park_safely)
        action_layout.addWidget(self.park_button)
        self.stop_button = QPushButton("暂停 / 安全终止并回停靠点")
        self.stop_button.setEnabled(False)
        self.stop_button.setMinimumHeight(54)
        self.stop_button.clicked.connect(self._safe_stop)
        action_layout.addWidget(self.stop_button)
        action_layout.addWidget(QLabel("运行中不允许修改参数；安全终止会向任务进程发送 SIGINT。"))
        side_layout.addWidget(action_box)

        settings_box = QGroupBox("视觉 / 现场参数（空闲时可调）")
        form = QFormLayout(settings_box)
        self.preview_a4_profile = QComboBox()
        self.preview_a4_profile.addItem("宽松自适应（仅预览）", "adaptive")
        self.preview_a4_profile.addItem("跟随正式参数（仅预览）", "formal")
        saved_preview_profile = str(self._settings.get("preview_a4_profile", "adaptive"))
        saved_index = self.preview_a4_profile.findData(saved_preview_profile)
        self.preview_a4_profile.setCurrentIndex(max(0, saved_index))
        self.preview_a4_profile.currentIndexChanged.connect(
            self._restart_preview_for_preview_profile
        )
        self.preview_a4_profile.currentIndexChanged.connect(self._save_settings)
        form.addRow("预览蓝纸参数", self.preview_a4_profile)
        self.params_preset = QComboBox()
        self.params_preset.addItems(["new", "current"])
        self.params_preset.setCurrentText(str(self._settings.get("params", "new")))
        self.params_preset.currentTextChanged.connect(self._restart_preview_for_params)
        self.params_preset.currentTextChanged.connect(self._save_settings)
        form.addRow("正式视觉预设", self.params_preset)
        self.process_fps = self._slider(1.0, 12.0, 0.5, "process_fps", 6.0, " Hz")
        form.addRow("视觉处理 FPS", self.process_fps)
        self.acquisition_timeout = self._slider(
            5.0, 15.0, 1.0, "acquisition_timeout", 10.0, " 秒"
        )
        form.addRow("稳定采集窗口", self.acquisition_timeout)
        self.travel_lift = self._slider(0.0, 15.0, 0.5, "travel_lift", 10.0, " mm")
        self.pickup_lift = self._slider(0.0, 15.0, 0.5, "pickup_lift", 2.0, " mm")
        self.place_lift = self._slider(0.0, 15.0, 0.5, "place_lift", 2.0, " mm")
        self.rest_lift = self._slider(0.0, 15.0, 0.5, "rest_lift", 10.0, " mm")
        form.addRow("搬运抬高 mm", self.travel_lift)
        form.addRow("吸取抬高 mm", self.pickup_lift)
        form.addRow("放置抬高 mm", self.place_lift)
        form.addRow("结束抬高 mm", self.rest_lift)
        self.continuous = QCheckBox("连续搬运 + F32C 并行回零")
        self.continuous.setChecked(bool(self._settings.get("continuous", True)))
        form.addRow(self.continuous)
        self.max_moves = QSpinBox()
        self.max_moves.setRange(0, 4)
        self.max_moves.setSpecialValueText("全部")
        self.max_moves.setValue(int(self._settings.get("max_moves", 0)))
        form.addRow("测试片数", self.max_moves)
        parking_x, parking_y = load_parking()
        form.addRow("安全停靠点", QLabel(f"({parking_x:g}, {parking_y:g}) mm"))
        side_layout.addWidget(settings_box)

        devices_box = QGroupBox("现场设备选择（空闲时可刷新）")
        devices_form = QFormLayout(devices_box)
        self.camera_device = QComboBox()
        self.camera_device.setToolTip("扫描到的 V4L2 摄像头；选择后预览与正式任务使用同一台。")
        self.camera_device.currentIndexChanged.connect(self._camera_device_changed)
        devices_form.addRow("摄像头", self.camera_device)
        self.writer_device = QComboBox()
        self.writer_device.setToolTip("扫描到的 USB 串口；正式执行时会以 GRBL 通信确认。")
        self.writer_device.currentIndexChanged.connect(self._writer_device_changed)
        devices_form.addRow("写字机 USB 串口", self.writer_device)
        self.refresh_devices_button = QPushButton("刷新摄像头 / 串口")
        self.refresh_devices_button.setMinimumHeight(42)
        self.refresh_devices_button.clicked.connect(self._refresh_hardware_devices_from_ui)
        devices_form.addRow(self.refresh_devices_button)
        self.device_status = QLabel("正在扫描现场设备…")
        self.device_status.setWordWrap(True)
        devices_form.addRow(self.device_status)
        side_layout.addWidget(devices_box)

        execution_box = QGroupBox("高级执行参数（现场可调）")
        execution_form = QFormLayout(execution_box)
        warning = QLabel(
            "默认值为当前实测配置。提高速度前请先用单片实测；"
            "任务运行中参数会锁定。"
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color:#8a4b00;")
        execution_form.addRow(warning)
        self.xy_feed = self._slider(
            500.0, 6000.0, 100.0, "xy_feed_mm_min", 4500.0, " mm/min"
        )
        self.z_feed = self._slider(
            500.0, 3000.0, 100.0, "z_feed_mm_min", 1500.0, " mm/min"
        )
        self.rotation_rpm = self._slider(
            1.0, 30.0, 1.0, "rotation_rpm", 29.0, " RPM"
        )
        self.magnet_pickup_dwell = self._slider(
            0.1, 2.0, 0.1, "magnet_pickup_dwell_seconds", 0.5, " s"
        )
        self.magnet_release_dwell = self._slider(
            0.1, 2.0, 0.1, "magnet_release_dwell_seconds", 0.5, " s"
        )
        execution_form.addRow("XY 进给", self.xy_feed)
        execution_form.addRow("Z 进给", self.z_feed)
        execution_form.addRow("F32C 转速", self.rotation_rpm)
        execution_form.addRow("吸合等待", self.magnet_pickup_dwell)
        execution_form.addRow("释放等待", self.magnet_release_dwell)
        side_layout.addWidget(execution_box)

        log_box = QGroupBox("实时状态 / 日志")
        log_layout = QVBoxLayout(log_box)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(800)
        log_layout.addWidget(self.log)
        side_layout.addWidget(log_box, 1)
        sidebar.setWidget(side)
        # 侧栏不参与与预览区的比例抢占，确保状态文本始终完整可读。
        layout.addWidget(sidebar, 0)

    def _apply_adaptive_layout(self, available_width: int) -> None:
        """按当前显示器宽度调整最小尺寸，不区分 VNC 与实体显示器。"""

        compact = available_width <= COMPACT_LAYOUT_MAX_WIDTH
        if compact == self._compact_layout:
            return
        self._compact_layout = compact
        if compact:
            # 1280 宽 VNC/小屏：三列仍同时可见，侧栏内部继续由原有滚动区承载。
            preview_size = (480, 320)
            solution_size = (220, 180)
            sidebar_min, sidebar_max = 360, 440
            task_button_height = 80
        else:
            # 大屏：保持原先为现场操作优化的显示尺度。
            preview_size = (760, 500)
            solution_size = (340, 260)
            sidebar_min, sidebar_max = 580, 680
            task_button_height = 96
        self.preview_label.setMinimumSize(*preview_size)
        self.solution_label.setMinimumSize(*solution_size)
        self._sidebar.setMinimumWidth(sidebar_min)
        self._sidebar.setMaximumWidth(sidebar_max)
        for button in self._task_buttons.values():
            button.setMinimumHeight(task_button_height)

    def fit_to_current_screen(self) -> None:
        """即使 VNC 未运行窗口管理器，也先让窗口落在可见区域内。"""

        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        self._apply_adaptive_layout(available.width())
        self.setGeometry(available)

    def _slider(
        self,
        low: float,
        high: float,
        step: float,
        key: str,
        default: float,
        suffix: str,
    ) -> ParameterSlider:
        control = ParameterSlider(
            low, high, step, float(self._settings.get(key, default)), suffix=suffix
        )
        control.valueChanged.connect(self._save_settings)
        return control

    def _load_settings(self) -> dict[str, Any]:
        try:
            raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _save_settings(self) -> None:
        if not hasattr(self, "magnet_release_dwell"):
            return
        values = {
            "params": self.params_preset.currentText(),
            "preview_a4_profile": str(self.preview_a4_profile.currentData()),
            "process_fps": self.process_fps.value(),
            "acquisition_timeout": self.acquisition_timeout.value(),
            "travel_lift": self.travel_lift.value(),
            "pickup_lift": self.pickup_lift.value(),
            "place_lift": self.place_lift.value(),
            "rest_lift": self.rest_lift.value(),
            "continuous": self.continuous.isChecked(),
            "xy_feed_mm_min": self.xy_feed.value(),
            "z_feed_mm_min": self.z_feed.value(),
            "rotation_rpm": int(self.rotation_rpm.value()),
            "magnet_pickup_dwell_seconds": self.magnet_pickup_dwell.value(),
            "magnet_release_dwell_seconds": self.magnet_release_dwell.value(),
            "max_moves": self.max_moves.value(),
            "plan_only": self.plan_only.isChecked(),
            "camera_path": self._camera,
            "writer_port": self._writer_port,
        }
        SETTINGS_PATH.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _plan_only_toggled(self, enabled: bool) -> None:
        """更新纯检测解算模式的安全提示，不触碰硬件。"""

        self._save_settings()
        if self._runner is not None:
            return
        self.park_button.setEnabled(
            self._hardware and not enabled
        )
        detail = (
            "纯检测与解算模式：任务不会连接写字机、F32C 或电磁铁。"
            if enabled
            else "硬件执行模式：开始任务前必须先建立原点并前往安全停靠点。"
        )
        self._set_status("空闲", detail)

    @staticmethod
    def _device_label(path: str) -> str:
        """把稳定路径与真实节点一起显示，方便现场人工确认。"""

        real_path = os.path.realpath(path)
        name = Path(path).name
        real_name = Path(real_path).name
        return f"{name}  →  {real_name}" if real_path != path else name

    @staticmethod
    def _combo_selected_path(combo: QComboBox) -> str | None:
        value = combo.currentData()
        return value if isinstance(value, str) and value else None

    def _populate_device_combo(
        self,
        combo: QComboBox,
        paths: list[str],
        preferred: str,
        *,
        empty_text: str,
        choose_text: str,
    ) -> str | None:
        """刷新设备下拉框；多候选时不擅自改变用户的明确选择。"""

        combo.blockSignals(True)
        combo.clear()
        selected: str | None = None
        if not paths:
            combo.addItem(empty_text, None)
        else:
            for path in paths:
                combo.addItem(self._device_label(path), path)
                combo.setItemData(combo.count() - 1, path, Qt.ItemDataRole.ToolTipRole)
            if preferred in paths:
                selected = preferred
            elif len(paths) == 1:
                selected = paths[0]
            else:
                combo.insertItem(0, choose_text, None)
            index = combo.findData(selected) if selected is not None else 0
            combo.setCurrentIndex(max(0, index))
        combo.blockSignals(False)
        return selected

    def _refresh_hardware_devices(
        self,
        *,
        restart_preview: bool,
        announce: bool = True,
    ) -> None:
        """扫描可换 USB 设备，让自动发现与人工选择使用同一套来源。"""

        if not self._hardware:
            self.device_status.setText("摄像头：未接入（无外设模式）｜写字机：未接入（无外设模式）")
            return
        handover_dir = str(ROOT / "handover-puzzle")
        if handover_dir not in sys.path:
            sys.path.insert(0, handover_dir)
        try:
            from hardware_discovery import discover_camera_paths, discover_writer_ports

            camera_paths = discover_camera_paths()
            writer_paths = discover_writer_ports()
        except Exception as error:
            self.device_status.setText(f"设备扫描失败：{error}")
            if announce:
                self._append_log(f"设备扫描失败：{error}")
            return

        previous_camera = self._camera
        previous_writer = self._writer_port
        self._refreshing_devices = True
        selected_camera = self._populate_device_combo(
            self.camera_device,
            camera_paths,
            previous_camera,
            empty_text="未发现摄像头（请检查 USB/供电后刷新）",
            choose_text="请选择要使用的摄像头…",
        )
        selected_writer = self._populate_device_combo(
            self.writer_device,
            writer_paths,
            previous_writer,
            empty_text="未发现 USB 串口（请检查写字机/线缆后刷新）",
            choose_text="请选择写字机 USB 串口…",
        )
        self._refreshing_devices = False
        if selected_camera is not None:
            self._camera = selected_camera
        if selected_writer is not None:
            self._writer_port = selected_writer
        self._save_settings()

        camera_detail = (
            self._device_label(selected_camera)
            if selected_camera is not None
            else ("未发现" if not camera_paths else "等待人工选择")
        )
        writer_detail = (
            self._device_label(selected_writer)
            if selected_writer is not None
            else ("未发现" if not writer_paths else "等待人工选择")
        )
        self.device_status.setText(
            f"摄像头：{camera_detail}｜写字机：{writer_detail}\n"
            "正式执行前会再次检查；写字机连接后还会进行 GRBL 通信确认。"
        )
        if announce:
            self._append_log(
                f"设备已刷新：摄像头 {len(camera_paths)} 个，USB 串口 {len(writer_paths)} 个；"
                f"当前摄像头={camera_detail}，写字机={writer_detail}。"
            )
        if restart_preview and hasattr(self, "_preview"):
            # 热插拔后 by-id 路径通常保持不变，但旧 VideoCapture 已经失效。
            # 因而“刷新”必须无条件释放并重开，不能只在路径文字变化时重启。
            if self._stop_preview():
                if selected_camera is not None:
                    self._append_log("已按刷新请求重新打开所选摄像头预览。")
                self._start_preview()

    def _refresh_hardware_devices_from_ui(self) -> None:
        """屏幕按钮入口：热插拔后立即重新枚举并重启所选相机预览。"""

        self._refresh_hardware_devices(restart_preview=True)

    def _camera_device_changed(self, _index: int) -> None:
        if self._refreshing_devices:
            return
        selected = self._combo_selected_path(self.camera_device)
        if selected is None:
            return
        changed = selected != self._camera
        self._camera = selected
        self._save_settings()
        self._append_log(f"已选择摄像头：{selected}")
        if changed and hasattr(self, "_preview") and self._stop_preview():
            self._start_preview()

    def _writer_device_changed(self, _index: int) -> None:
        if self._refreshing_devices:
            return
        selected = self._combo_selected_path(self.writer_device)
        if selected is None:
            return
        self._writer_port = selected
        self._save_settings()
        self._append_log(f"已选择写字机 USB 串口：{selected}")

    def _set_status(self, state: str, detail: str) -> None:
        self.state_label.setText(f"状态：{state}\n{detail}")
        print(f"[状态] {state}：{detail}", flush=True)

    @staticmethod
    def _format_timer(seconds: float) -> str:
        total_tenths = max(0, round(seconds * 10.0))
        minutes, tenths_remainder = divmod(total_tenths, 600)
        whole_seconds, tenths = divmod(tenths_remainder, 10)
        return f"{minutes:02d}:{whole_seconds:02d}.{tenths}"

    def _render_competition_timer(self, hint: str) -> None:
        remaining = max(
            0.0, COMPETITION_TIME_LIMIT_SECONDS - self._competition_elapsed_seconds
        )
        self.competition_timer_label.setText(
            f"剩余 {self._format_timer(remaining)}"
        )
        self.competition_timer_hint.setText(
            f"{hint}\n已用 {self._format_timer(self._competition_elapsed_seconds)} / 02:00.0"
        )

    def _start_competition_timer(self, title: str) -> None:
        """每次真正启动一问时清零并开始该问的两分钟计时。"""

        self._competition_started_at = time.monotonic()
        self._competition_elapsed_seconds = 0.0
        self._competition_timer.start()
        self._render_competition_timer(f"计时中：{title}")

    def _update_competition_timer(self) -> None:
        if self._competition_started_at is None:
            return
        self._competition_elapsed_seconds = time.monotonic() - self._competition_started_at
        if self._competition_elapsed_seconds >= COMPETITION_TIME_LIMIT_SECONDS:
            self._competition_elapsed_seconds = COMPETITION_TIME_LIMIT_SECONDS
            self._competition_timer.stop()
            self._render_competition_timer("已到 2 分钟上限；任务仍可继续完成")
            return
        self._render_competition_timer("计时中：本问正在执行")

    def _stop_competition_timer(self, hint: str) -> None:
        if self._competition_started_at is not None:
            self._competition_elapsed_seconds = min(
                COMPETITION_TIME_LIMIT_SECONDS,
                time.monotonic() - self._competition_started_at,
            )
        self._competition_timer.stop()
        self._competition_started_at = None
        self._render_competition_timer(hint)

    def _set_completion_indicator(self, completed: bool) -> None:
        """比赛计时用的完成灯：仅拼图任务成功结束才点亮绿色。"""

        color = "#1b8f2e" if completed else "#c00000"
        self.completion_light.setStyleSheet(
            f"background:{color}; border:3px solid #000; border-radius:22px;"
        )
        self.completion_light.setToolTip("任务完成" if completed else "未完成或运行中")

    def _announce_piece_completion(self) -> None:
        """最后一片确认释放后更新完成状态。"""

        if self._completion_signaled:
            return
        self._completion_signaled = True
        self._set_completion_indicator(True)
        self._set_status("完成", "最后一片已放下；正在回安全停靠点")
        self._stop_competition_timer("完成：最后一片已释放")
        self._append_log("最后一片已释放：完成灯已变绿；正在回安全停靠点。")

    def _append_log(self, text: str) -> None:
        message = text.rstrip()
        if not message:
            return
        self.log.appendPlainText(message)
        # 屏幕界面和从终端启动时的信息必须一致，便于现场排障。
        print(f"[上位机] {message}", flush=True)

    def _update_origin_status(self) -> None:
        if self._machine_origin_established:
            self.origin_status.setText(
                "本次上电坐标：已建立并保持。后续任务将直接复用，"
                "串口重连后会以安全停靠点恢复该坐标。"
            )
            self.park_button.setText("回安全停靠点")
            self.park_button.setToolTip("将当前位置重锚为安全点坐标后，释放机构并回安全停靠点。")
            self.park_button.setEnabled(
                self._hardware and self._runner is None and not self.plan_only.isChecked()
            )
        else:
            self.origin_status.setText(
                "本次上电坐标：待首次建立。请在上电前将机构放在写字机"
                "机械限位角、Z 自然下垂，然后点击下方初始化按钮。"
            )
            self.park_button.setText("建立原点并前往安全停靠点")
            self.park_button.setToolTip("把当前位置设为机械原点，再移动到固定安全停靠点。")
            self.park_button.setEnabled(
                self._hardware and self._runner is None and not self.plan_only.isChecked()
            )

    def _update_vision_readiness(self) -> None:
        """在预览阶段明确告知实体/屏幕按键是否可触发任务。"""

        if not self._hardware or self._preview is None:
            self.vision_readiness.setText("Q6A HTP：本地图片后端已接入｜相机及运动任务：待验收")
            return
        if self._preview_vision_ready:
            message = (
                f"视觉预检：已稳定识别 {self._preview_stable_pieces} 片"
                "｜遮挡流程下仍可直接按任务按钮"
            )
            style = "background:#fff; color:#000; border:3px solid #000;"
        else:
            message = (
                "视觉采集：请等待｜"
                f"稳定={self._preview_stable_pieces} 片，单帧={self._preview_single_pieces} 片｜"
                f"{self._preview_vision_detail}｜遮挡流程下可直接启动"
            )
            style = "background:#fff; color:#000; border:2px dashed #000;"
        self.vision_readiness.setText(message)
        self.vision_readiness.setStyleSheet(
            "font-size:17px; font-weight:700; padding:8px; " + style
        )

    def _preview_vision_state(
        self,
        stable_pieces: int,
        single_pieces: int,
        ready: bool,
        detail: str,
    ) -> None:
        self._preview_stable_pieces = stable_pieces
        self._preview_single_pieces = single_pieces
        self._preview_vision_ready = ready
        self._preview_vision_detail = detail
        self._update_vision_readiness()

    def _show_task_capture_status(self, text: str) -> None:
        """预览释放后，继续在同一区域显示正式任务的采集进度。"""

        self.vision_readiness.setText(
            "视觉采集：正式任务正在独立验证连续五帧\n" + text
        )
        self.vision_readiness.setStyleSheet(
            "font-size:17px; font-weight:700; padding:8px; "
            "background:#fff; color:#000; border:2px dashed #000;"
        )

    def _restart_preview_for_params(self, _params_name: str) -> None:
        """切换视觉预设后立即重启低优先级预览，使画面实时反映该预设。"""

        if not hasattr(self, "_preview") or self._runner is not None:
            return
        self._preview_vision_state(0, 0, False, "视觉预设已切换，正在重新建立稳定结果")
        if self._stop_preview():
            self._start_preview()

    def _restart_preview_for_preview_profile(self, _index: int) -> None:
        """切换仅预览的蓝纸参数，绝不修改正式任务参数。"""

        if not hasattr(self, "_preview") or self._runner is not None:
            return
        self._preview_vision_state(0, 0, False, "预览蓝纸参数已切换，正在重新检测")
        if self._stop_preview():
            self._start_preview()

    def _start_preview(
        self,
        *,
        preview_mode: str = "live_yolo",
        frozen_overlay: dict[str, Any] | None = None,
        _from_retry: bool = False,
    ) -> None:
        """Do not start capture until a Q6A puzzle backend is available."""
        self._preview_retry_timer.stop()
        self._preview_vision_detail = VISION_UNAVAILABLE
        self.preview_caption.setText("无外设模式" if not self._hardware else "相机任务待验收")
        self._append_log(VISION_UNAVAILABLE)

    def _stop_preview(self) -> bool:
        """Stop preview completely before a replacement can open the camera."""
        self._preview_retry_timer.stop()
        preview = self._preview
        if preview is not None and preview.isRunning():
            preview.stop()
            if not preview.wait(4000):
                self._append_log("预览线程尚未安全退出；保留当前线程，不启动新的相机预览。")
                return False
        if self._preview is preview:
            self._preview = None
        self._preview_vision_state(0, 0, False, "预览已暂停，任务将重新验证连续五帧")
        return True

    def _retry_preview(self) -> None:
        """设备热插拔后重新尝试预览；任务运行中绝不抢占正式相机。"""

        if self._runner is not None or self._terminal_interrupt_requested:
            return
        if self._preview is not None and self._preview.isRunning():
            return
        self._preview_retry_attempt += 1
        self._append_log(
            "相机仍未就绪，正在重新发现"
            f"（第 {self._preview_retry_attempt} 次）…"
        )
        self._start_preview(
            preview_mode=self._preview_retry_mode,
            frozen_overlay=self._preview_retry_overlay,
            _from_retry=True,
        )

    def _load_frozen_overlay(self, path_text: str) -> dict[str, Any] | None:
        """读取正式任务一次性写出的稳定视觉叠加层。"""

        path = Path(path_text.strip())
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("schema_version") != "1.0":
                raise ValueError("冻结结果版本不匹配")
            if len(raw.get("source_image_size", ())) != 2:
                raise ValueError("缺少原始图像尺寸")
            if not isinstance(raw.get("accepted_pieces"), list):
                raise ValueError("缺少冻结拼图轮廓")
            return raw
        except Exception as error:
            self._append_log(f"无法读取冻结视觉叠加结果：{path}；{error}")
            return None

    def _show_frame(self, image: QImage) -> None:
        self._image = image
        self._render_preview_image()

    def _render_preview_image(self) -> None:
        if self._image is None:
            return
        self.preview_label.setPixmap(QPixmap.fromImage(self._image).scaled(
            self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        ))

    def _render_solution_image(self) -> None:
        if self._solution_image is None:
            return
        self.solution_label.setPixmap(QPixmap.fromImage(self._solution_image).scaled(
            self.solution_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_sidebar"):
            self._apply_adaptive_layout(event.size().width())
        self._render_preview_image()
        self._render_solution_image()

    def _preview_failed(self, reason: str) -> None:
        message = reason + f"\n将在 {CAMERA_DISCOVERY_RETRY_MS // 1000} 秒后自动重试。"
        self.preview_label.setText(message)
        self._append_log(message)
        if self._runner is None and not self._terminal_interrupt_requested:
            self._preview_retry_timer.start()

    def _show_solution_image(self, path_text: str, *, flip_layout: bool = True) -> None:
        path = Path(path_text.strip())
        if flip_layout:
            try:
                display_path = render_content_flipped_solution(path)
            except Exception as error:
                display_path = path
                self._append_log(f"仅翻转拼图内容的结果图生成失败，暂显示原图：{error}")
        else:
            # 第 3 问的图片本身就是从冻结相机画面裁出的真实拼图块，并已按
            # 最终 MovePlan 的方向与间隙拼合，不能再被几何图渲染器覆盖。
            display_path = path
        image = QImage(str(display_path))
        if image.isNull():
            self._append_log(f"无法加载求解结果图：{display_path}")
            return
        self._solution_image = image
        self.solution_caption.setText(
            f"结果图：仅拼图内容上下翻转｜{path.name}"
            if flip_layout
            else f"结果图：实拍花纹裁片按最终运动计划复原｜{path.name}"
        )
        self._render_solution_image()
        # 相机应已在“视觉已冻结”事件后恢复为轻量预览。若冻结预览文件
        # 不可用才在此兜底恢复原始画面，避免在任务期间重新启动 YOLO。
        if self._preview is None or not self._preview.isRunning():
            self._start_preview(preview_mode="raw")
        kind = "实拍花纹复原图" if not flip_layout else "求解结果图"
        self._append_log(f"已在上位机结果区显示{kind}：{display_path}")

    def _clear_solution_image(self, message: str) -> None:
        """清空结果内容但保留独立结果区域，避免遮挡实时检测。"""

        self._solution_image = None
        self.solution_label.setPixmap(QPixmap())
        self.solution_label.setText(message)
        self.solution_caption.setText("结果区：等待本次求解")

    def _set_running_widgets(self, running: bool) -> None:
        for button in self._task_buttons.values():
            button.setEnabled(False)  # Q6A 拼图模型尚未迁移
        for widget in (
            self.preview_a4_profile, self.params_preset, self.process_fps, self.travel_lift,
            self.acquisition_timeout, self.pickup_lift, self.place_lift, self.rest_lift,
            self.continuous, self.xy_feed, self.z_feed, self.rotation_rpm,
            self.magnet_pickup_dwell, self.magnet_release_dwell, self.max_moves,
            self.plan_only, self.camera_device, self.writer_device,
            self.refresh_devices_button,
        ):
            widget.setEnabled(not running)
        if not self._hardware:
            for widget in (self.plan_only, self.camera_device, self.writer_device,
                           self.refresh_devices_button):
                widget.setEnabled(False)
        self.stop_button.setEnabled(running)
        self.park_button.setEnabled(
            self._hardware
            and not running
            and not self.plan_only.isChecked()
        )

    def _start_task(self, number: int) -> None:
        self._append_log("已拒绝启动：" + ("无外设模式；" if not self._hardware else "") + VISION_UNAVAILABLE)

    def _park_safely(self) -> None:
        """在已建立坐标的同一次上电中，主动回到视觉安全停靠点。"""

        if not self._hardware:
            self._append_log("无外设模式：已拒绝安全停靠。")
            return
        try:
            validate_motor_port(self._q6a_config["motor_port"])
        except ValueError as error:
            self._append_log(str(error))
            return
        if self._runner is not None:
            self._append_log("已有任务正在运行；不能同时发起安全停靠。")
            return
        if self.plan_only.isChecked():
            self._append_log("当前为硬件隔离模式；已拒绝安全停靠，未连接任何电机。")
            return
        self._refresh_hardware_devices(restart_preview=False, announce=False)
        selected_writer = self._combo_selected_path(self.writer_device)
        if selected_writer is None:
            self._set_status("等待设备", "未选择写字机 USB 串口；请接好后刷新并选择。")
            self._append_log("已拒绝安全停靠：当前没有可用写字机 USB 串口选择。")
            return
        initializing = not self._machine_origin_established
        self.task_label.setText("当前任务：初始化并前往安全停靠点" if initializing else "当前任务：安全停靠")
        self._set_status(
            "初始化安全停靠点" if initializing else "回安全停靠点",
            (
                "以当前位置建立机械原点后移动；请确认机构已在机械限位角。"
                if initializing
                else "释放预览相机后复用当前坐标移动"
            ),
        )
        self._stop_preview()
        self._run_sent = False
        self._stopping = False
        process = QProcess(self)
        process.setWorkingDirectory(str(ROOT / "handover-puzzle"))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_runner_output)
        process.errorOccurred.connect(self._runner_error)
        process.finished.connect(self._runner_finished)
        arguments = [
            str(RUNNER), "--hardware", "--motor-port", self._q6a_config["motor_port"], "--safe-parking-config", str(SAFE_PARKING),
            "--travel-lift-mm", f"{self.travel_lift.value():g}",
            "--pickup-lift-mm", f"{self.pickup_lift.value():g}",
            "--place-lift-mm", f"{self.place_lift.value():g}",
            "--rest-lift-mm", f"{self.rest_lift.value():g}",
            "--xy-feed-mm-min", f"{self.xy_feed.value():g}",
            "--z-feed-mm-min", f"{self.z_feed.value():g}",
            "--rotation-rpm", str(int(self.rotation_rpm.value())),
            "--magnet-pickup-dwell-seconds", f"{self.magnet_pickup_dwell.value():g}",
            "--magnet-release-dwell-seconds", f"{self.magnet_release_dwell.value():g}",
            "--writer-port", selected_writer,
            "--execute",
        ]
        if initializing:
            arguments.append("--initialize-and-park")
        else:
            arguments += ["--reuse-machine-origin", "--park-only"]
        self._append_log(
            ("启动初始化安全停靠：" if initializing else "启动安全停靠：")
            + " ".join(arguments[1:])
        )
        self._runner = process
        self._runner_kind = "initialize" if initializing else "park"
        self._set_running_widgets(True)
        process.start(str(VENV_PYTHON), arguments)

    def _runner_error(self, _error: Any) -> None:
        if self._runner is not None:
            self._append_log(f"任务进程启动或通信异常：{self._runner.errorString()}")

    def _read_runner_output(self) -> None:
        if self._runner is None:
            return
        text = bytes(self._runner.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self._append_log(text)
            capture_updates = re.findall(r"视觉采集中：[^\r\n]+", text)
            if capture_updates:
                self._show_task_capture_status(capture_updates[-1])
            frozen_match = re.search(r"视觉冻结预览已就绪：\s*([^\r\n]+)", text)
            if frozen_match is not None:
                self._frozen_overlay = self._load_frozen_overlay(
                    frozen_match.group(1)
                )
            solution_match = re.search(r"求解结果图已保存：\s*([^\r\n]+)", text)
            if solution_match is not None:
                self._show_solution_image(solution_match.group(1))
            pattern_match = re.search(r"花纹复原图已保存：\s*([^\r\n]+)", text)
            if pattern_match is not None:
                self._show_solution_image(pattern_match.group(1), flip_layout=False)
            if "硬件坐标基准状态：" in text:
                self._machine_origin_established = True
                self._update_origin_status()
                self._append_log(
                    "已记录本次上电机械坐标；后续任务会从安全停靠点恢复该坐标。"
                )
            if self.LAST_PIECE_RELEASED_SIGNAL in text:
                self._announce_piece_completion()
            if "视觉已冻结" in text:
                self._set_status("规划中", "已获得稳定视觉结果")
                self._start_preview(
                    preview_mode=(
                        "frozen_overlay"
                        if self._frozen_overlay is not None else "raw"
                    ),
                    frozen_overlay=self._frozen_overlay,
                )
                self.preview_caption.setText(
                    "实时检测：原始实时画面 + 本次冻结 YOLO 结果（YOLO 已关闭）"
                    if self._frozen_overlay is not None
                    else "实时检测：原始实时画面（冻结叠加结果不可用；YOLO 已关闭）"
                )
                self.vision_readiness.setText(
                    "视觉采集：正式任务已冻结稳定结果；相机已恢复为轻量原始预览，正在求解"
                )
                self.vision_readiness.setStyleSheet(
                    "font-size:17px; font-weight:700; padding:8px; "
                    "background:#fff; color:#000; border:3px solid #000;"
                )
            elif "安全停靠完成" in text:
                self._set_status("完成", "已释放机构并回到安全停靠点")
            elif "任务完成" in text:
                self._set_status("完成", "拼图流程已结束并回安全停靠点")
            elif "输入 RUN" in text and not self._run_sent:
                self._run_sent = True
                self._runner.write(b"RUN\n")
                if self._preview is not None and self._preview.isRunning():
                    self._preview.clear_frozen_overlay()
                self._frozen_overlay = None
                self.preview_caption.setText(
                    "实时检测：机械执行中，仅显示原始实时画面（冻结框已隐藏）"
                )
                self.vision_readiness.setText(
                    "视觉采集：冻结结果已送入求解并开始执行；当前仅显示原始相机画面"
                )
                self._set_status("执行中", "已确认计划，正在连接并执行硬件")

    def _safe_stop(self) -> None:
        if self._runner is None:
            return
        self._stopping = True
        self.stop_button.setEnabled(False)
        self._set_status(
            "终止请求中",
            (
                "正在停止纯检测与解算进程；未连接任何硬件。"
                if self._runner_kind == "plan_only"
                else "等待任务进程抬 Z、释放磁铁并回安全停靠点"
            ),
        )
        try:
            os.kill(int(self._runner.processId()), signal.SIGINT)
            self._append_log("已向任务进程发送 SIGINT 安全终止请求。")
        except OSError as error:
            self._append_log(f"无法发送安全终止请求：{error}")

    def _runner_finished(self, exit_code: int, _status: Any) -> None:
        # finished 信号有时先于最后一段 readyRead 到达；先取尽输出，确保
        # 最后一片释放信号不会被遗漏或晚于“任务完成”处理。
        self._read_runner_output()
        finished_kind = self._runner_kind
        completed_task = finished_kind == "task" and self._completion_signaled
        plan_only_finished = finished_kind == "plan_only"
        if self._stopping:
            if completed_task:
                self._set_status("完成", "最后一片已放下；安全停靠流程被终止，请确认机构位置")
            else:
                self._set_completion_indicator(False)
                self._set_status("已安全终止", "请确认机构位于安全停靠点后再重新开始")
        elif exit_code == 0:
            if plan_only_finished:
                self._set_status(
                    "解算完成（未连接硬件）",
                    "已完成相机检测、稳定识别、解算和运动计划预检；"
                    "未打开写字机、F32C 或电磁铁。",
                )
            else:
                detail = (
                    "已回安全停靠点；本次上电坐标仍可复用"
                    if finished_kind == "park"
                    else (
                        "已从机械原点到达安全停靠点；现在可以执行拼图任务"
                        if finished_kind == "initialize"
                        else "任务完成；机构应位于安全停靠点"
                    )
                )
                self._set_status("完成", detail)
            if finished_kind == "task":
                if not self._completion_signaled:
                    # 兼容旧任务脚本未输出释放信号的情况；新版通常已在最后
                    # 一片释放时完成声光提示，这里不能重复鸣叫。
                    self._announce_piece_completion()
        else:
            if completed_task:
                self._set_status(
                    "完成（停靠异常）",
                    f"最后一片已放下；回安全停靠点失败，退出码：{exit_code}",
                )
            else:
                self._set_completion_indicator(False)
                self._set_status("失败", f"任务进程退出码：{exit_code}；请查看日志")
        if finished_kind in {"task", "plan_only"} and not self._completion_signaled:
            self._stop_competition_timer(
                "纯检测解算完成：计时已停止"
                if plan_only_finished
                else "任务未完成：计时已停止"
            )
        self._append_log(f"任务进程已结束，exit={exit_code}")
        self._runner.deleteLater() if self._runner is not None else None
        self._runner = None
        self._runner_kind = ""
        self._terminal_interrupt_requested = False
        self._set_running_widgets(False)
        self._update_origin_status()
        if (
            finished_kind == "task"
            and (completed_task or exit_code == 0)
        ):
            self._clear_solution_image("本次任务已结束\n下次求解结果将显示在这里")
        self._frozen_overlay = None
        self._stop_preview()
        self.preview_caption.setText("实时检测：相机预览")
        self._start_preview()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._runner is not None:
            QMessageBox.warning(self, "任务仍在执行", "请先使用“暂停 / 安全终止”，等待任务结束后再关闭界面。")
            event.ignore()
            return
        if self._preview is not None:
            self._stop_preview()
        self._competition_timer.stop()
        event.accept()

    def handle_terminal_interrupt(self) -> None:
        """将终端 Ctrl+C 转为 Qt 主线程中的受控操作，不抛 KeyboardInterrupt。"""

        if self._terminal_interrupt_requested:
            return
        self._terminal_interrupt_requested = True
        if self._runner is not None:
            self._append_log("收到终端 Ctrl+C；改为请求安全终止，请等待任务结束。")
            self._safe_stop()
            return
        self._append_log("收到终端 Ctrl+C；正在正常关闭上位机。")
        self.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Q6A 拼图上位机（默认无外设模式）")
    add_mode_arguments(parser)
    args = parser.parse_args(argv)
    if not VENV_PYTHON.is_file() or not RUNNER.is_file() or not SAFE_PARKING.is_file():
        print("上位机文件不完整；请从 Tangram-26 仓库根目录启动。", file=sys.stderr)
        return 2
    app = QApplication(sys.argv)
    app.setApplicationName("拼图装置上位机")
    # 现场光照变化大，界面只使用黑白高对比度，避免灰蓝色文字难以辨认。
    app.setStyleSheet(
        """
        QWidget { background: #ffffff; color: #000000; font-size: 16px; }
        QGroupBox {
            border: 2px solid #000000; margin-top: 14px; padding: 9px;
            font-weight: 700;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        QPushButton {
            background: #ffffff; color: #000000; border: 2px solid #000000;
            border-radius: 0; padding: 8px; font-size: 17px; font-weight: 700;
        }
        QPushButton:pressed { background: #000000; color: #ffffff; }
        QPushButton:disabled {
            background: #ffffff; color: #000000; border: 2px dashed #000000;
            font-style: italic;
        }
        QCheckBox { color: #000000; spacing: 8px; }
        QCheckBox::indicator { width: 22px; height: 22px; border: 2px solid #000000; }
        QCheckBox::indicator:checked { background: #000000; }
        QPlainTextEdit, QComboBox, QDoubleSpinBox, QSpinBox {
            background: #ffffff; color: #000000; border: 2px solid #000000;
            selection-background-color: #000000; selection-color: #ffffff;
        }
        QSlider::groove:horizontal {
            height: 8px; background: #ffffff; border: 2px solid #000000;
        }
        QSlider::handle:horizontal {
            width: 18px; margin: -7px 0; background: #000000;
            border: 2px solid #000000;
        }
        QSlider:disabled::handle:horizontal { background: #888888; }
        QScrollArea { border: 2px solid #000000; }
        """
    )
    window = MainWindow(hardware=args.hardware)
    window.fit_to_current_screen()
    window.showMaximized()

    # Python 默认会把 Ctrl+C 注入正在运行的 Qt 槽函数，造成无意义的
    # KeyboardInterrupt 回溯。转为事件循环中的关闭/安全终止请求。
    def _handle_sigint(_signal_number: int, _frame: Any) -> None:
        QTimer.singleShot(0, window.handle_terminal_interrupt)

    signal.signal(signal.SIGINT, _handle_sigint)
    # Qt 的 exec() 长时间停留在 C++ 事件循环内时，Python 可能不会及时分发
    # SIGINT。周期性回到 Python 一次，确保终端 Ctrl+C 能触发上面的受控关闭。
    signal_pump = QTimer(app)
    signal_pump.setInterval(200)
    signal_pump.timeout.connect(lambda: None)
    signal_pump.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
