#!/usr/bin/env python3
"""实时识别蓝色 A4 上的一片白色拼图，并经确认后执行抓取、旋转和放件。

坐标约定已经和本装置对齐：蓝色 A4 的右上角为写字机 (0,0)，X+ 向下，
Y+ 向左。参考视觉算法输出横向 A4 左上角相对坐标 (u, v)，本脚本只作固定
轴向换算 ``X=206.5*v, Y=292*(1-u)``；不重新拟合相机/写字机标定。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

# OpenCV wheel 的 Qt 运行时未随包携带字体；优先使用系统已有字体，避免每帧
# 重复输出 QFontDatabase 警告。必须在 import cv2 前设置。
if os.path.isdir("/usr/share/fonts/truetype/dejavu"):
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
REFERENCE_VISION_DIR = PROJECT_DIR / "reference-vision" / "ec_26_e"
REFERENCE_VISION_SRC = REFERENCE_VISION_DIR / "src"
PREFERRED_CAMERA = "/dev/v4l/by-id/usb-Generic_USB_Camera3_200901010001-video-index0"
# 现场蓝纸实测尺寸（不是标准 A4）：右上角向下的短边、向左的长边。
A4_X_MM = 206.5
A4_Y_MM = 292.0


@dataclass(frozen=True)
class VisionPiece:
    """一片已转换为写字机 A4 毫米坐标的视觉结果。"""

    group_id: int
    pickup_x_mm: float
    pickup_y_mm: float
    orientation_deg: float
    confidence: float
    source_corners_px: np.ndarray
    source_center_px: tuple[int, int]


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的有限数字")
    return value


def _nonnegative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("必须是大于等于 0 的有限数字")
    return value


def _ratio(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("必须处于 [0, 1]")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", default=PREFERRED_CAMERA, help="V4L2 摄像头设备路径")
    parser.add_argument("--camera-width", type=int, default=1280, help="相机采集宽度，px")
    parser.add_argument("--camera-height", type=int, default=720, help="相机采集高度，px")
    parser.add_argument("--camera-fps", type=int, default=30, help="相机请求帧率")
    parser.add_argument("--process-fps", type=_positive_float, default=8.0, help="蓝纸/拼图检测频率，Hz")
    parser.add_argument(
        "--workspace-mode",
        choices=("global", "reference"),
        default="global",
        help="global=全图寻找蓝色 A4 后再精拟合；reference=使用同事工位的固定参考位置",
    )
    parser.add_argument("--writer-port", default="/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0", help="写字机 USB 串口")
    parser.add_argument("--motor-port", default="/dev/ttyAS3", help="F32C 无刷 UART")
    parser.add_argument("--motor-address", type=int, default=2, help="F32C 地址")
    parser.add_argument("--place-x", type=_nonnegative_float, default=100.0, help="放件 X，mm")
    parser.add_argument("--place-y", type=_nonnegative_float, default=100.0, help="放件 Y，mm")
    parser.add_argument("--rotation-deg", type=float, default=180.0, help="搬运途中相对旋转角度，°")
    parser.add_argument("--safe-z", type=float, default=-8.0, help="搬运时 Z 抬起位置，mm")
    parser.add_argument("--xy-feed", type=_positive_float, default=2500.0, help="XY 速度，mm/min")
    parser.add_argument("--z-feed", type=_positive_float, default=1500.0, help="Z 速度，mm/min")
    parser.add_argument("--rotation-rpm", type=int, default=15, help="无刷位置速度，RPM")
    parser.add_argument("--min-confidence", type=_ratio, default=0.70, help="允许抓取的最低视觉置信度")
    parser.add_argument("--preview-only", action="store_true", help="只拍照检测，不执行任何运动")
    parser.add_argument("--windowed", action="store_true", help="不全屏显示，便于桌面调试")
    parser.add_argument("--debug", action="store_true", help="显示每帧失败原因")
    return parser


def _load_reference_vision() -> tuple[Any, Any, Any, Any]:
    """加载同事工程中的校正和检测实现，不复制算法。"""
    if not REFERENCE_VISION_SRC.is_dir():
        raise FileNotFoundError(f"视觉参考源码不存在：{REFERENCE_VISION_SRC}")
    source_text = str(REFERENCE_VISION_SRC)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        from ec_26_e.vision import Detect1, Detect1Config, Resize, ResizeConfig
    except ImportError as exc:
        raise RuntimeError(
            "无法加载视觉参考工程；请安装 handover-puzzle/requirements.txt"
        ) from exc
    return Resize, ResizeConfig, Detect1, Detect1Config


def _build_detectors() -> tuple[Any, Any, Any]:
    Resize, ResizeConfig, Detect1, Detect1Config = _load_reference_vision()
    config_path = REFERENCE_VISION_DIR / "configs" / "tasks" / "task1.json"
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    resize_data = dict(config["resize"])
    detect_data = dict(config["detect1"])
    # 这次测试目标明确是白色拼图；参考工程 mode=1 使用白色核心筛选。
    detect_data["piece_mode"] = 1
    return Resize, ResizeConfig(**resize_data), Detect1(Detect1Config(**detect_data))


def _order_clockwise_corners(points: np.ndarray) -> np.ndarray:
    """将四个图像坐标固定为 TL、TR、BR、BL（图像坐标系顺时针）。"""
    if points.shape != (4, 2):
        raise ValueError("蓝色 A4 候选四角数量不是 4")
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    if cv2.contourArea(ordered.reshape(-1, 1, 2).astype(np.float32), oriented=True) < 0:
        ordered = ordered[::-1]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -start, axis=0)


def _global_blue_workspace_seed(frame_bgr: np.ndarray) -> np.ndarray:
    """不依赖固定相机位置，从全图最大青蓝四边形产生 A4 初始四角。

    这一步只给参考工程 Resize 提供初始位置；后续四边亚像素拟合、蓝纸
    外观校验和透视校正仍全部由参考工程实现。
    """
    source_height, source_width = frame_bgr.shape[:2]
    detect_size = (640, 360)
    image = cv2.resize(frame_bgr, detect_size, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 青蓝纸在当前与预期现场均明显高饱和；范围避开黑色底板、白色拼图和木桌。
    mask = cv2.inRange(
        hsv,
        np.asarray((80, 60, 40), dtype=np.uint8),
        np.asarray((110, 255, 255), dtype=np.uint8),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("全图未找到青蓝 A4 候选")
    contour = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(contour) / float(detect_size[0] * detect_size[1])
    if area_ratio < 0.25:
        raise ValueError(f"最大青蓝候选过小：{area_ratio:.3f} < 0.250")
    corners = cv2.approxPolyDP(
        contour,
        0.015 * cv2.arcLength(contour, True),
        True,
    ).reshape(-1, 2).astype(np.float32)
    if len(corners) != 4 or not cv2.isContourConvex(corners.reshape(-1, 1, 2)):
        raise ValueError(f"最大青蓝候选不是完整凸四边形（角点数={len(corners)}）")
    ordered = _order_clockwise_corners(corners)
    scale = np.asarray(
        [
            (source_width - 1) / (detect_size[0] - 1),
            (source_height - 1) / (detect_size[1] - 1),
        ],
        dtype=np.float32,
    )
    return ordered * scale


def _resize_with_workspace_mode(
    frame_bgr: np.ndarray,
    resize_class: Any,
    base_config: Any,
    workspace_mode: str,
) -> tuple[Any, np.ndarray | None]:
    """以固定参考或全图蓝纸初定位运行同事工程的 Resize。"""
    if workspace_mode == "reference":
        return resize_class(base_config).resize(frame_bgr), None
    seed_px = _global_blue_workspace_seed(frame_bgr)
    source_height, source_width = frame_bgr.shape[:2]
    relative = seed_px / np.asarray(
        [max(source_width - 1, 1), max(source_height - 1, 1)],
        dtype=np.float32,
    )
    config = replace(
        base_config,
        reference_corners_relative=tuple(
            (float(point[0]), float(point[1])) for point in relative
        ),
    )
    return resize_class(config).resize(frame_bgr), np.rint(seed_px).astype(np.int32)


def _relative_to_writer_mm(point: tuple[float, float]) -> tuple[float, float]:
    """横向 A4 左上角 (u,v) 转为本机右上角 X/Y 毫米坐标。"""
    u, v = point
    return A4_X_MM * float(v), A4_Y_MM * (1.0 - float(u))


def _orientation_deg(corners_relative: tuple[tuple[float, float], ...]) -> float:
    """用最长轮廓边给出显示用朝向；抓取测试仍固定相对旋转 180°。"""
    corners = [_relative_to_writer_mm(point) for point in corners_relative]
    edges = [
        (corners[(index + 1) % len(corners)][0] - corner[0], corners[(index + 1) % len(corners)][1] - corner[1])
        for index, corner in enumerate(corners)
    ]
    dx, dy = max(edges, key=lambda edge: edge[0] ** 2 + edge[1] ** 2)
    return (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0


def _vision_pieces(detect_result: Any, resize_result: Any) -> tuple[VisionPiece, ...]:
    inverse_homography = np.linalg.inv(resize_result.homography)
    result: list[VisionPiece] = []
    for piece in detect_result.pieces:
        pickup_x, pickup_y = _relative_to_writer_mm(piece.center_relative)
        source_corners = cv2.perspectiveTransform(
            np.asarray(piece.corners_px, dtype=np.float32).reshape(1, -1, 2),
            inverse_homography,
        )[0]
        source_center = cv2.perspectiveTransform(
            np.asarray(piece.center_px, dtype=np.float32).reshape(1, 1, 2),
            inverse_homography,
        )[0, 0]
        result.append(
            VisionPiece(
                group_id=piece.group_id,
                pickup_x_mm=pickup_x,
                pickup_y_mm=pickup_y,
                orientation_deg=_orientation_deg(piece.corners_relative),
                confidence=float(piece.confidence),
                source_corners_px=np.rint(source_corners).astype(np.int32),
                source_center_px=tuple(np.rint(source_center).astype(int)),
            )
        )
    return tuple(result)


def _draw_preview(
    frame_bgr: np.ndarray,
    resize_result: Any | None,
    detect_result: Any | None,
    pieces: tuple[VisionPiece, ...],
    status: str,
    failure: str | None,
    workspace_seed_px: np.ndarray | None,
) -> np.ndarray:
    """生成 1920×1080 屏幕画面：原始相机、校正 A4、控制坐标和说明。"""
    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    canvas[:] = (24, 24, 24)
    raw = frame_bgr.copy()
    if workspace_seed_px is not None:
        cv2.polylines(raw, [workspace_seed_px.reshape(-1, 1, 2)], True, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(raw, "GLOBAL BLUE SEED", tuple(workspace_seed_px[0] + (5, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
    if resize_result is not None:
        corners = np.rint(np.asarray(resize_result.source_corners_px)).astype(np.int32)
        cv2.polylines(raw, [corners.reshape(-1, 1, 2)], True, (255, 255, 0), 2, cv2.LINE_AA)
        for label, point in zip(("TL", "TR", "BR", "BL"), corners):
            cv2.putText(raw, label, tuple(point + (5, -5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
    for piece in pieces:
        cv2.polylines(raw, [piece.source_corners_px.reshape(-1, 1, 2)], True, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.circle(raw, piece.source_center_px, 6, (0, 0, 255), -1, cv2.LINE_AA)
        label = f"#{piece.group_id} X={piece.pickup_x_mm:.1f} Y={piece.pickup_y_mm:.1f} a={piece.orientation_deg:.0f} c={piece.confidence:.2f}"
        cv2.putText(raw, label, (piece.source_center_px[0] + 8, piece.source_center_px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)

    raw_display = cv2.resize(raw, (1200, 675), interpolation=cv2.INTER_AREA)
    canvas[185:860, 0:1200] = raw_display
    cv2.putText(canvas, "LIVE CAMERA / BLUE A4 + WHITE PIECE", (30, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2, cv2.LINE_AA)

    if resize_result is not None and detect_result is not None:
        rectified = resize_result.image_bgr.copy()
        for piece in detect_result.pieces:
            corners = np.rint(np.asarray(piece.corners_px)).astype(np.int32)
            center = tuple(np.rint(np.asarray(piece.center_px)).astype(int))
            cv2.polylines(rectified, [corners.reshape(-1, 1, 2)], True, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(rectified, center, 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(rectified, f"group {piece.group_id}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        rectified_display = cv2.resize(rectified, (690, 488), interpolation=cv2.INTER_AREA)
        canvas[235:723, 1210:1900] = rectified_display
        cv2.putText(canvas, "RECTIFIED A4 / DETECT1", (1220, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (230, 230, 230), 2, cv2.LINE_AA)

    status_color = (0, 255, 0) if failure is None else (0, 80, 255)
    cv2.putText(canvas, status[:115], (30, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.82, status_color, 2, cv2.LINE_AA)
    coordinate_text = "Coordinate: A4 right-top = writer (0,0); X+ down, Y+ left"
    cv2.putText(canvas, coordinate_text, (30, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    if failure:
        for line_index, line in enumerate((failure[:78], failure[78:156])):
            if line:
                cv2.putText(canvas, line, (30, 137 + 28 * line_index), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "SPACE: freeze exactly one valid piece   S: save snapshot   ESC/Q: quit", (30, 1025), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)
    return canvas


def _open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    # FFmpeg 可以直接使用 /dev/v4l/by-id 稳定链接；本机 OpenCV V4L2
    # 后端则只接受其解析后的 /dev/videoN 字符设备。
    requested_path = Path(args.camera)
    device_path = requested_path.resolve() if requested_path.exists() else requested_path
    capture = cv2.VideoCapture(str(device_path), cv2.CAP_V4L2)
    if not capture.isOpened():
        raise RuntimeError(
            f"无法打开摄像头：{args.camera}（实际设备：{device_path}）"
        )
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    capture.set(cv2.CAP_PROP_FPS, args.camera_fps)
    return capture


def _save_snapshot(image_bgr: np.ndarray) -> Path:
    output_dir = PACKAGE_DIR / "outputs" / "vision"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"vision_pick_{datetime.now():%Y%m%d_%H%M%S}.jpg"
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"无法保存截图：{path}")
    return path


def _preview_and_select(args: argparse.Namespace) -> tuple[VisionPiece, Path] | None:
    resize_class, resize_config, detector = _build_detectors()
    capture = _open_camera(args)
    window_name = "Tangram Puzzle - Vision Pick Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if not args.windowed:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    latest_display: np.ndarray | None = None
    latest_candidates: tuple[VisionPiece, ...] = ()
    latest_valid: tuple[VisionPiece, ...] = ()
    last_console_report = ""
    last_console_time = 0.0
    last_process_time = -math.inf
    cached_resize: Any | None = None
    cached_detect: Any | None = None
    cached_failure: str | None = "等待第一帧视觉检测..."
    cached_status = "Starting vision pipeline..."
    cached_workspace_seed: np.ndarray | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("摄像头未返回有效帧")
            now = time.monotonic()
            if now - last_process_time >= 1.0 / args.process_fps:
                last_process_time = now
                cached_resize = None
                cached_detect = None
                cached_workspace_seed = None
                try:
                    cached_resize, cached_workspace_seed = _resize_with_workspace_mode(
                        frame,
                        resize_class,
                        resize_config,
                        args.workspace_mode,
                    )
                    cached_detect = detector.detect(cached_resize.image_bgr)
                    latest_candidates = _vision_pieces(cached_detect, cached_resize)
                    latest_valid = tuple(piece for piece in latest_candidates if piece.confidence >= args.min_confidence)
                    cached_failure = None
                    cached_status = f"Detected {len(latest_candidates)} piece(s), valid {len(latest_valid)}; SPACE requires exactly one valid piece"
                except Exception as exc:
                    latest_candidates = ()
                    latest_valid = ()
                    cached_failure = f"参考 Resize/Detect1 未通过：{exc}"
                    cached_status = "Blue A4 not ready; no motion is possible"
            report = cached_status if cached_failure is None else f"{cached_status} | {cached_failure}"
            if report != last_console_report and time.monotonic() - last_console_time >= 1.0:
                print(f"[VISION] {report}")
                last_console_report = report
                last_console_time = time.monotonic()
            latest_display = _draw_preview(frame, cached_resize, cached_detect, latest_candidates, cached_status, cached_failure, cached_workspace_seed)
            cv2.imshow(window_name, latest_display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                return None
            if key in (ord("s"), ord("S")):
                path = _save_snapshot(latest_display)
                print(f"已保存视觉截图：{path}")
            if key == ord(" "):
                if len(latest_valid) != 1:
                    print(f"⛔ 当前有效拼图数为 {len(latest_valid)}，必须恰好为 1；不会运动。")
                    continue
                path = _save_snapshot(latest_display)
                return latest_valid[0], path
    finally:
        capture.release()
        try:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass


def _run_motion(args: argparse.Namespace, piece: VisionPiece) -> int:
    """复用已经真实验证过的机电脚本，避免重新实现抓取时序。"""
    try:
        from .test_pick_rotate_place import main as motion_main
    except ImportError:
        from test_pick_rotate_place import main as motion_main
    command = [
        "--writer-port", args.writer_port,
        "--motor-port", args.motor_port,
        "--motor-address", str(args.motor_address),
        "--pickup-x", f"{piece.pickup_x_mm:.3f}",
        "--pickup-y", f"{piece.pickup_y_mm:.3f}",
        "--place-x", f"{args.place_x:.3f}",
        "--place-y", f"{args.place_y:.3f}",
        "--rotation-deg", f"{args.rotation_deg:.3f}",
        "--safe-z", f"{args.safe_z:.3f}",
        "--xy-feed", f"{args.xy_feed:.3f}",
        "--z-feed", f"{args.z_feed:.3f}",
        "--rotation-rpm", str(args.rotation_rpm),
        "--yes",
    ]
    return motion_main(command)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not -8.0 <= args.safe_z < 0.0:
        print("⛔ --safe-z 只允许 -8mm 到 0mm；Z 负方向为抬起。", file=sys.stderr)
        return 2
    if not 1 <= args.rotation_rpm <= 30:
        print("⛔ --rotation-rpm 必须处于 1～30。", file=sys.stderr)
        return 2
    if not 0 <= args.motor_address <= 255:
        print("⛔ --motor-address 必须处于 0～255。", file=sys.stderr)
        return 2

    print("实时预览启动：请只在蓝色 A4 内放置一片白色拼图。")
    print("屏幕操作：SPACE 冻结唯一有效拼图；S 保存截图；Esc/Q 取消。")
    try:
        selected = _preview_and_select(args)
    except KeyboardInterrupt:
        print("\n已通过 Ctrl+C 退出实时预览；未连接任何运动硬件。")
        return 130
    except (FileNotFoundError, RuntimeError, cv2.error) as exc:
        print(f"⛔ 实时预览启动失败：{exc}", file=sys.stderr)
        return 1
    if selected is None:
        print("已取消；未连接写字机、无刷或电磁铁。")
        return 0
    piece, snapshot = selected
    print()
    print("=" * 72)
    print("视觉抓取候选已冻结（尚未运动）")
    print("=" * 72)
    print(f"截图：{snapshot}")
    print(f"拼图 group={piece.group_id}，置信度={piece.confidence:.3f}，视觉朝向={piece.orientation_deg:.1f}°")
    print(f"抓取：X={piece.pickup_x_mm:.3f} mm，Y={piece.pickup_y_mm:.3f} mm")
    print(f"放件：X={args.place_x:.3f} mm，Y={args.place_y:.3f} mm；途中相对旋转 {args.rotation_deg:+.1f}°")
    print("请确认机构已人工置于右上角原点、Z 自然下垂、路径无人手。")
    if args.preview_only:
        print("预览模式完成；未连接任何运动硬件。")
        return 0
    if input("确认视觉抓取坐标无误后输入 RUN（其他输入取消）: ").strip().upper() != "RUN":
        print("已取消；未连接任何运动硬件。")
        return 0
    return _run_motion(args, piece)


if __name__ == "__main__":
    raise SystemExit(main())
