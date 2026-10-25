#!/usr/bin/env python3
"""实时遮挡鲁棒 A4/拼图检测诊断：只读取摄像头，不控制任何硬件。

与 handover-puzzle/test_a4_detection.py 相同的界面，但使用
遮挡鲁棒管线：蓝纸被遮挡一角/一边时，用 3 边 + 宽高比补全第 4 角。
估计角以橙色标注；状态栏显示实测/估计角数量。

用法：
  .venv/bin/python tests/a4_robust/run_live.py [--params current|new] [--windowed]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path

if os.path.isdir("/usr/share/fonts/truetype/dejavu"):
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np

try:
    from . import _paths
except ImportError:
    import _paths
_paths.ensure()

import test_a4_detection as a4d

try:
    from .pipeline import detect_robust
except ImportError:
    from pipeline import detect_robust


def _draw(frame: np.ndarray, result, status: str) -> np.ndarray:
    canvas = np.full((1080, 1920, 3), (24, 24, 24), dtype=np.uint8)
    raw = frame.copy()
    if result is not None and result.resize_result is not None:
        corners = np.rint(np.asarray(result.resize_result.source_corners_px)).astype(np.int32)
        for index, corner in enumerate(corners):
            color = (0, 165, 255) if not result.reliability[index] else (255, 0, 255)
            cv2.circle(raw, tuple(corner), 8, color, -1, cv2.LINE_AA)
            cv2.putText(
                raw,
                f"{['TL','TR','BR','BL'][index]}{'*' if not result.reliability[index] else ''}",
                tuple(corner + (8, -8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.polylines(raw, [corners.reshape(-1, 1, 2)], True, (255, 0, 255), 2, cv2.LINE_AA)
        detect = result.detect_result
        if detect is not None:
            inverse = np.linalg.inv(result.resize_result.homography)
            for piece in detect.pieces:
                points = np.asarray(piece.corners_px, dtype=np.float32).reshape(1, -1, 2)
                source = cv2.perspectiveTransform(points, inverse)[0]
                center = cv2.perspectiveTransform(
                    np.asarray(piece.center_px, dtype=np.float32).reshape(1, 1, 2), inverse
                )[0, 0]
                cv2.polylines(raw, [np.rint(source).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.circle(raw, tuple(np.rint(center).astype(int)), 6, (0, 0, 255), -1, cv2.LINE_AA)
    canvas[175:850, 0:1200] = cv2.resize(raw, (1200, 675), interpolation=cv2.INTER_AREA)
    cv2.putText(canvas, "ROBUST A4 + PIECE DETECTION (orange = estimated corner)", (28, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2, cv2.LINE_AA)
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
    parser.description = "实时遮挡鲁棒 A4/拼图检测诊断：只读取摄像头。"
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
    capture = a4d._open_camera(
        argparse.Namespace(camera=args.camera, width=args.width, height=args.height, fps=args.fps)
    )
    window = "Tangram - Robust A4 Detection"
    last_process = -math.inf
    latest = None
    status = "等待摄像头画面..."
    terminal_stop = threading.Event()
    output_dir = Path(__file__).resolve().parent / "outputs"
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        if args.windowed:
            cv2.resizeWindow(window, 1280, 720)
        else:
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        print("实时遮挡鲁棒 A4/拼图诊断已启动：只读取摄像头。")
        print(f"参数预设：{args.params}（HSV={args.hue_min}/{args.hue_max}/{args.sat_min}/{args.value_min}）")
        print("退出：预览窗口获得焦点后按 Esc/Q；或终端输入 q 后按 Enter；Ctrl+C 也可安全退出。")
        threading.Thread(target=_listen_terminal_quit, args=(terminal_stop,), name="robust-quit", daemon=True).start()
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("摄像头未返回有效画面")
            now = time.monotonic()
            if now - last_process >= 1.0 / args.process_fps:
                last_process = now
                try:
                    latest = detect_robust(frame, params, a4_args)
                    measured = sum(latest.reliability)
                    mode = latest.seed.mode
                    status = (
                        f"OK: seed={mode} 实测角={measured}/4，估计角={4 - measured}/4，"
                        f"pieces={len(latest.detect_result.pieces)}"
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
                output = output_dir / f"robust_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(output), display)
                print(f"已保存：{output}")
    except KeyboardInterrupt:
        print("\n已通过 Ctrl+C 退出；未控制任何硬件。")
        return 130
    finally:
        capture.release()
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except cv2.error:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
