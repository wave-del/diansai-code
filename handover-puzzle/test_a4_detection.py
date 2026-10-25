#!/usr/bin/env python3
"""实时检查蓝色 A4 与白色拼图检测；只读取摄像头，不控制任何硬件。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

if os.path.isdir("/usr/share/fonts/truetype/dejavu"):
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
REFERENCE_DIR = PROJECT_DIR / "reference-vision" / "ec_26_e"
REFERENCE_SRC = REFERENCE_DIR / "src"
PREFERRED_CAMERA = "/dev/v4l/by-id/usb-Generic_USB_Camera3_200901010001-video-index0"

# 参数预设：current=现场现有参数（夜间调好的固定门槛，默认行为不变）；
# new=稳健参数（蓝纸 seed 降门槛 + 白片 seed 自适应相对亮度，日夜通用）。
# 显式传入的 --hue-min 等 CLI 参数始终覆盖预设。
PARAM_PRESETS: dict[str, dict[str, float | int]] = {
    "current": {
        "hue_min": 92,
        "hue_max": 112,
        "sat_min": 150,
        "value_min": 100,
        "white_seed_min_brightness": 180,
        "white_seed_max_saturation": 90,
        "white_seed_brightness_ratio": 0.0,
    },
    "new": {
        "hue_min": 92,
        "hue_max": 112,
        "sat_min": 145,
        "value_min": 90,
        "white_seed_min_brightness": 60,
        "white_seed_max_saturation": 90,
        "white_seed_brightness_ratio": 0.7,
    },
}


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("必须为正的有限数字")
    return value


def _ratio(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("必须在 [0, 1] 范围内")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", default=PREFERRED_CAMERA, help="V4L2 摄像头路径；可使用 /dev/v4l/by-id 稳定链接")
    parser.add_argument("--width", type=int, default=1280, help="采集宽度，px")
    parser.add_argument("--height", type=int, default=720, help="采集高度，px")
    parser.add_argument("--fps", type=int, default=30, help="相机请求帧率")
    parser.add_argument("--process-fps", type=_positive_float, default=6.0, help="A4/拼图检测频率，Hz")
    parser.add_argument(
        "--params",
        choices=sorted(PARAM_PRESETS),
        default="current",
        help="参数预设：current=现有固定门槛；new=稳健自适应门槛（夜间/白天通用）",
    )
    parser.add_argument("--hue-min", type=int, default=None, help="高置信蓝纸 HSV Hue 下限；省略时取 --params 预设")
    parser.add_argument("--hue-max", type=int, default=None, help="高置信蓝纸 HSV Hue 上限；省略时取 --params 预设")
    parser.add_argument("--sat-min", type=int, default=None, help="高置信蓝纸 HSV Saturation 下限；省略时取 --params 预设")
    parser.add_argument("--value-min", type=int, default=None, help="高置信蓝纸 HSV Value 下限；省略时取 --params 预设")
    parser.add_argument("--min-a4-area", type=_ratio, default=0.25, help="蓝纸候选最小画面面积比例")
    parser.add_argument("--max-refine-shift", type=_positive_float, default=35.0, help="参考工程精拟合角点相对蓝纸候选允许的最大偏移，px")
    parser.add_argument("--windowed", action="store_true", help="窗口模式；默认全屏")
    return parser


def _load_reference() -> tuple[Any, Any, Any, Any]:
    raise RuntimeError("旧传统视觉参考工程未包含在本包中；Q6A 拼图视觉后端待迁移。")


def _resolve_params(args: argparse.Namespace) -> dict[str, float | int]:
    """合并 --params 预设与显式 CLI 覆盖，返回完整检测参数。"""
    preset = dict(PARAM_PRESETS[args.params])
    for key in ("hue_min", "hue_max", "sat_min", "value_min"):
        value = getattr(args, key, None)
        if value is not None:
            preset[key] = value
    return preset


def _build_reference_detectors(params: dict[str, float | int] | None = None) -> tuple[Any, Any, Any]:
    Resize, ResizeConfig, Detect1, Detect1Config = _load_reference()
    with (REFERENCE_DIR / "configs" / "tasks" / "task1.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    resize_config = ResizeConfig(**dict(config["resize"]))
    detect_config = dict(config["detect1"])
    detect_config["piece_mode"] = 1
    if params is not None:
        for key in ("white_seed_min_brightness", "white_seed_max_saturation", "white_seed_brightness_ratio"):
            detect_config[key] = params[key]
    return Resize, resize_config, Detect1(Detect1Config(**detect_config))


def _order_corners(points: np.ndarray) -> np.ndarray:
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    if cv2.contourArea(ordered.reshape(-1, 1, 2).astype(np.float32), oriented=True) < 0:
        ordered = ordered[::-1]
    return np.roll(ordered, -int(np.argmin(ordered[:, 0] + ordered[:, 1])), axis=0)


def _blue_a4_seed(frame_bgr: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, float, np.ndarray]:
    """以足够亮、饱和的青蓝区域定位纸张粗四角，排除暗色机架与接线。"""
    source_height, source_width = frame_bgr.shape[:2]
    detect_width, detect_height = 640, 360
    image = cv2.resize(frame_bgr, (detect_width, detect_height), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.asarray((args.hue_min, args.sat_min, args.value_min), dtype=np.uint8)
    upper = np.asarray((args.hue_max, 255, 255), dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area_ratio = cv2.contourArea(contour) / float(detect_width * detect_height)
        if area_ratio < args.min_a4_area:
            continue
        polygon = cv2.approxPolyDP(contour, 0.015 * cv2.arcLength(contour, True), True).reshape(-1, 2)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon.reshape(-1, 1, 2)):
            continue
        if np.any(polygon[:, 0] <= 1) or np.any(polygon[:, 0] >= detect_width - 2) or np.any(polygon[:, 1] <= 1) or np.any(polygon[:, 1] >= detect_height - 2):
            continue
        candidates.append((area_ratio, polygon.astype(np.float32)))
    if not candidates:
        raise ValueError("未找到完整四边形蓝纸候选；请检查蓝纸颜色、完整可见性或 HSV 参数")
    area_ratio, polygon = max(candidates, key=lambda item: item[0])
    scale = np.asarray([(source_width - 1) / (detect_width - 1), (source_height - 1) / (detect_height - 1)], dtype=np.float32)
    return _order_corners(polygon) * scale, area_ratio, mask


def _detect(frame_bgr: np.ndarray, resize_class: Any, base_resize_config: Any, detector: Any, args: argparse.Namespace) -> tuple[Any, Any, np.ndarray, float, np.ndarray]:
    seed_px, area_ratio, mask = _blue_a4_seed(frame_bgr, args)
    height, width = frame_bgr.shape[:2]
    relative = seed_px / np.asarray((width - 1, height - 1), dtype=np.float32)
    config = replace(base_resize_config, reference_corners_relative=tuple((float(x), float(y)) for x, y in relative))
    resize_result = resize_class(config).resize(frame_bgr)
    refined = np.asarray(resize_result.source_corners_px, dtype=np.float32)
    shifts = np.linalg.norm(refined - seed_px, axis=1)
    if float(np.max(shifts)) > args.max_refine_shift:
        raise ValueError(f"精拟合偏离蓝纸候选过大：{float(np.max(shifts)):.1f}px > {args.max_refine_shift:.1f}px")
    detect_result = detector.detect(resize_result.image_bgr)
    return resize_result, detect_result, np.rint(seed_px).astype(np.int32), area_ratio, mask


def _draw(frame: np.ndarray, resize_result: Any | None, detect_result: Any | None, seed: np.ndarray | None, area_ratio: float | None, status: str) -> np.ndarray:
    canvas = np.full((1080, 1920, 3), (24, 24, 24), dtype=np.uint8)
    raw = frame.copy()
    if seed is not None:
        cv2.polylines(raw, [seed.reshape(-1, 1, 2)], True, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(raw, "BLUE A4 SEED", tuple(seed[0] + (6, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
    if resize_result is not None:
        corners = np.rint(np.asarray(resize_result.source_corners_px)).astype(np.int32)
        cv2.polylines(raw, [corners.reshape(-1, 1, 2)], True, (255, 255, 0), 3, cv2.LINE_AA)
        for name, point in zip(("TL", "TR", "BR", "BL"), corners):
            cv2.putText(raw, name, tuple(point + (6, -6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
    if resize_result is not None and detect_result is not None:
        inverse = np.linalg.inv(resize_result.homography)
        for piece in detect_result.pieces:
            points = np.asarray(piece.corners_px, dtype=np.float32).reshape(1, -1, 2)
            source = cv2.perspectiveTransform(points, inverse)[0]
            center = cv2.perspectiveTransform(np.asarray(piece.center_px, dtype=np.float32).reshape(1, 1, 2), inverse)[0, 0]
            cv2.polylines(raw, [np.rint(source).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 3, cv2.LINE_AA)
            center_i = tuple(np.rint(center).astype(int))
            cv2.circle(raw, center_i, 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(raw, f"piece {piece.group_id}  confidence={piece.confidence:.2f}", (center_i[0] + 8, center_i[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2, cv2.LINE_AA)
    canvas[175:850, 0:1200] = cv2.resize(raw, (1200, 675), interpolation=cv2.INTER_AREA)
    cv2.putText(canvas, "LIVE: BLUE A4 + WHITE PUZZLE", (28, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 235, 235), 2, cv2.LINE_AA)
    if resize_result is not None and detect_result is not None:
        rectified = resize_result.image_bgr.copy()
        for piece in detect_result.pieces:
            corners = np.rint(np.asarray(piece.corners_px)).astype(np.int32)
            cv2.polylines(rectified, [corners.reshape(-1, 1, 2)], True, (0, 255, 0), 2, cv2.LINE_AA)
        canvas[235:723, 1210:1900] = cv2.resize(rectified, (690, 488), interpolation=cv2.INTER_AREA)
        cv2.putText(canvas, "REFERENCE RESIZE + DETECT1", (1220, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (235, 235, 235), 2, cv2.LINE_AA)
    color = (0, 255, 0) if status.startswith("OK") else (0, 80, 255)
    cv2.putText(canvas, status[:120], (28, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    if area_ratio is not None:
        cv2.putText(canvas, f"blue A4 candidate area={area_ratio:.3f}; purple=seed, cyan=reference edge fit, green=piece", (28, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, "S: save screenshot   ESC/Q: quit   This script never controls hardware", (28, 1030), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 2, cv2.LINE_AA)
    return canvas


def _open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    # 配置 by-id 路径优先；换卡或换摄像头导致该链接失效时，只有现场唯一的
    # V4L2 摄像头才会被自动选中，避免多个设备时误用错误画面。
    from hardware_discovery import open_camera_with_fallback

    capture, _device = open_camera_with_fallback(args.camera, cv2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    return capture


def _listen_terminal_quit(stop_event: threading.Event) -> None:
    """允许预览窗口失焦时在终端输入 q 后回车退出。"""
    try:
        command = input()
    except (EOFError, KeyboardInterrupt):
        return
    if command.strip().lower() in {"q", "quit", "exit"}:
        stop_event.set()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    params = _resolve_params(args)
    args.hue_min = int(params["hue_min"])
    args.hue_max = int(params["hue_max"])
    args.sat_min = int(params["sat_min"])
    args.value_min = int(params["value_min"])
    if not (0 <= args.hue_min <= args.hue_max <= 179 and 0 <= args.sat_min <= 255 and 0 <= args.value_min <= 255):
        print("⛔ HSV 参数不在 OpenCV 合法范围内。", file=sys.stderr)
        return 2
    resize_class, resize_config, detector = _build_reference_detectors(params)
    capture: cv2.VideoCapture | None = None
    window = "Tangram Puzzle - A4 Detection Only"
    last_process = -math.inf
    latest: tuple[Any, Any, np.ndarray, float] | None = None
    latest_status = "Waiting for camera frame..."
    terminal_stop = threading.Event()
    try:
        capture = _open_camera(args)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        if args.windowed:
            cv2.resizeWindow(window, 1280, 720)
        else:
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        print("实时 A4/拼图诊断已启动：只读取摄像头。")
        print(f"参数预设：{args.params}（HSV={args.hue_min}/{args.hue_max}/{args.sat_min}/{args.value_min}，"
              f"白片seed 亮度={params['white_seed_min_brightness']}，相对比例={params['white_seed_brightness_ratio']}）")
        print("退出：预览窗口获得焦点后按 Esc/Q；或终端输入 q 后按 Enter；Ctrl+C 也可安全退出。")
        threading.Thread(target=_listen_terminal_quit, args=(terminal_stop,), name="a4-terminal-quit", daemon=True).start()
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("摄像头未返回有效画面")
            now = time.monotonic()
            if now - last_process >= 1.0 / args.process_fps:
                last_process = now
                try:
                    resize_result, detect_result, seed, area_ratio, _ = _detect(frame, resize_class, resize_config, detector, args)
                    latest = (resize_result, detect_result, seed, area_ratio)
                    latest_status = f"OK: A4 fitted, piece_count={len(detect_result.pieces)}, line_residual={resize_result.line_residual:.3f}px"
                except Exception as exc:
                    latest = None
                    latest_status = f"FAIL: {exc}"
            if latest is None:
                display = _draw(frame, None, None, None, None, latest_status)
            else:
                display = _draw(frame, *latest, latest_status)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if terminal_stop.is_set() or key in (27, ord("q"), ord("Q")):
                return 0
            if key in (ord("s"), ord("S")):
                output_dir = PACKAGE_DIR / "outputs" / "a4_detection"
                output_dir.mkdir(parents=True, exist_ok=True)
                output = output_dir / f"a4_detection_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(output), display)
                print(f"已保存：{output}")
    except KeyboardInterrupt:
        print("\n已通过 Ctrl+C 退出；未控制任何硬件。")
        return 130
    except (OSError, RuntimeError, cv2.error, ValueError) as exc:
        print(f"⛔ A4 诊断失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if capture is not None:
            capture.release()
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except cv2.error:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
