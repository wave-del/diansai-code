#!/usr/bin/env python3
"""实时组合检测诊断：遮挡鲁棒 seed + 同事 Resize + YOLO 拼图检测。

原始画面全屏显示：A4 四角（紫=seed、青=精修、橙=估计角）、
拼图边界（绿）、状态栏（seed 模式、实测/估计角、稳定/单帧、耗时）。
只读取摄像头，不控制任何硬件。

用法：
  .venv/bin/python tests/a4_robust/run_live_yolo.py [--params current|new] [--windowed]
按 Esc/Q 或终端 q 退出；S 保存截图。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_DIR = _PACKAGE_DIR.parent.parent
MODEL_LIB = _REPO_DIR / "model-lib"

# Q6A 推理器显式加载 model-lib/q6a；无需注入 A733 运行库。

if os.path.isdir("/usr/share/fonts/truetype/dejavu"):
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

try:
    from . import _paths
except ImportError:
    import _paths
_paths.ensure()

import test_a4_detection as a4d  # noqa: E402

try:
    from .yolo_pipeline import YoloRobustDetector
except ImportError:
    from yolo_pipeline import YoloRobustDetector
try:
    from .source_region_filter import DEFAULT_SOURCE_MAX_X_RATIO
except ImportError:
    from source_region_filter import DEFAULT_SOURCE_MAX_X_RATIO


def _draw(frame, result, status: str) -> np.ndarray:
    canvas = np.full((1080, 1920, 3), (24, 24, 24), dtype=np.uint8)
    raw = frame.copy()
    if result is not None:
        seed = result.seed
        # seed 角（紫）+ 估计角（橙）
        for index, corner in enumerate(seed.corners):
            color = (0, 165, 255) if not seed.reliability[index] else (255, 0, 255)
            cv2.circle(raw, tuple(np.rint(corner).astype(int)), 7, color, -1, cv2.LINE_AA)
        cv2.polylines(
            raw,
            [np.rint(seed.corners).astype(np.int32).reshape(-1, 1, 2)],
            True,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
        # 精修后四角（青）
        refined = np.asarray(result.resize_result.source_corners_px, dtype=np.float32)
        cv2.polylines(
            raw,
            [np.rint(refined).astype(np.int32).reshape(-1, 1, 2)],
            True,
            (255, 255, 0),
            3,
            cv2.LINE_AA,
        )
        inverse = np.linalg.inv(result.resize_result.homography)
        width, height = result.detect_result.image_size
        divider = np.asarray(
            [[width * DEFAULT_SOURCE_MAX_X_RATIO, 0.0],
             [width * DEFAULT_SOURCE_MAX_X_RATIO, height - 1.0]],
            dtype=np.float32,
        ).reshape(1, -1, 2)
        divider_source = cv2.perspectiveTransform(divider, inverse)[0]
        cv2.line(
            raw,
            tuple(np.rint(divider_source[0]).astype(int)),
            tuple(np.rint(divider_source[1]).astype(int)),
            (0, 255, 255), 2, cv2.LINE_AA,
        )
        # 拼图：稳定结果优先，未稳定显示单帧
        display = (
            result.detect_result
            if result.detect_result.pieces
            else result.single_result
        )
        if display is not None and display.pieces:
            for piece in display.pieces:
                points = np.asarray(piece.corners_px, dtype=np.float32).reshape(1, -1, 2)
                source = cv2.perspectiveTransform(points, inverse)[0]
                center = cv2.perspectiveTransform(
                    np.asarray(piece.center_px, dtype=np.float32).reshape(1, 1, 2),
                    inverse,
                )[0, 0]
                cv2.polylines(
                    raw,
                    [np.rint(source).astype(np.int32).reshape(-1, 1, 2)],
                    True,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.circle(raw, tuple(np.rint(center).astype(int)), 6, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(raw, "LEFT PUZZLE", tuple(np.rint(center + (8, -8)).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        for piece in result.rejected_pieces:
            points = np.asarray(piece.corners_px, dtype=np.float32).reshape(1, -1, 2)
            source = cv2.perspectiveTransform(points, inverse)[0]
            center = cv2.perspectiveTransform(
                np.asarray(piece.center_px, dtype=np.float32).reshape(1, 1, 2), inverse,
            )[0, 0]
            cv2.polylines(
                raw,
                [np.rint(source).astype(np.int32).reshape(-1, 1, 2)],
                True, (0, 0, 255), 3, cv2.LINE_AA,
            )
            cv2.putText(raw, "IGNORED RIGHT", tuple(np.rint(center + (8, -8)).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    canvas[175:850, 0:1200] = cv2.resize(raw, (1200, 675), interpolation=cv2.INTER_AREA)
    cv2.putText(canvas, "green=LEFT PUZZLE | red=IGNORED RIGHT | yellow=52% source boundary", (28, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (235, 235, 235), 2, cv2.LINE_AA)
    color = (0, 255, 0) if status.startswith("OK") else (0, 80, 255)
    cv2.putText(canvas, status[:130], (28, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "S: 保存截图   ESC/Q: 退出   本脚本不控制任何硬件", (28, 1030),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 2, cv2.LINE_AA)
    return canvas


def _listen_terminal_quit(stop_event: threading.Event) -> None:
    try:
        command = input()
    except (EOFError, KeyboardInterrupt):
        return
    if command.strip().lower() in {"q", "quit", "exit"}:
        stop_event.set()


def main(argv: list[str] | None = None) -> int:
    parser = a4d._build_parser()
    parser.description = "实时组合检测诊断：遮挡鲁棒 seed + 同事 Resize + YOLO 拼图。只读摄像头。"
    parser.add_argument("--no-stable", action="store_true", help="关闭七帧稳定器，只显示单帧结果")
    args = parser.parse_args(argv)
    params = a4d._resolve_params(args)
    args.hue_min = int(params["hue_min"])
    args.hue_max = int(params["hue_max"])
    args.sat_min = int(params["sat_min"])
    args.value_min = int(params["value_min"])
    a4_args = argparse.Namespace(
        hue_min=args.hue_min, hue_max=args.hue_max,
        sat_min=args.sat_min, value_min=args.value_min,
        min_a4_area=0.25,
    )
    detector = YoloRobustDetector(params, a4_args, stable=not args.no_stable)
    capture = a4d._open_camera(
        argparse.Namespace(camera=args.camera, width=args.width, height=args.height, fps=args.fps)
    )
    window = "Tangram - YOLO Robust Detection"
    last_process = -math.inf
    latest = None
    status = "等待摄像头画面..."
    terminal_stop = threading.Event()
    output_dir = _PACKAGE_DIR / "outputs"
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        if args.windowed:
            cv2.resizeWindow(window, 1280, 720)
        else:
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        print("实时组合检测诊断已启动：只读取摄像头。")
        print(f"参数预设：{args.params}（HSV={args.hue_min}/{args.hue_max}/{args.sat_min}/{args.value_min}）"
              + ("，七帧稳定器开启" if not args.no_stable else "，单帧模式"))
        print("退出：预览窗口获得焦点后按 Esc/Q；或终端输入 q 后按 Enter；Ctrl+C 也可安全退出。")
        threading.Thread(target=_listen_terminal_quit, args=(terminal_stop,), name="yolo-quit", daemon=True).start()
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("摄像头未返回有效画面")
            now = time.monotonic()
            if now - last_process >= 1.0 / args.process_fps:
                last_process = now
                try:
                    latest = detector.detect(frame)
                    measured = sum(latest.seed.reliability)
                    stable_pieces = len(latest.detect_result.pieces)
                    single_pieces = (
                        len(latest.single_result.pieces)
                        if latest.single_result is not None
                        else 0
                    )
                    t = latest.timing_ms
                    diag = detector.diagnostics
                    status = (
                        f"OK: seed={latest.seed.mode} 实测角={measured}/4 稳定片数={stable_pieces} "
                        f"单帧片数={single_pieces} "
                        f"seed={t['seed']:.0f}ms resize={t['resize']:.0f}ms yolo+精修={t['yolo_refine']:.0f}ms "
                        f"总={t['total']:.0f}ms"
                    )
                except Exception as exc:
                    latest = None
                    status = f"FAIL: {exc}"
            display = _draw(frame, latest, status)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if terminal_stop.is_set() or key in (27, ord("q"), ord("Q")):
                return 0
            if key in (ord("s"), ord("S")):
                output_dir.mkdir(parents=True, exist_ok=True)
                output = output_dir / f"yolo_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(output), display)
                print(f"已保存：{output}")
    except KeyboardInterrupt:
        print("\n已通过 Ctrl+C 退出；未控制任何硬件。")
        return 130
    finally:
        capture.release()
        detector.close()
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except cv2.error:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
