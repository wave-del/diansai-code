#!/usr/bin/env python3
"""录制 USB 摄像头视频；默认 10 秒，只读取摄像头，不控制机械硬件。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

import cv2


DEFAULT_CAMERA = "/dev/v4l/by-id/usb-Generic_USB_Camera3_200901010001-video-index0"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "camera"


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default=DEFAULT_CAMERA, help="V4L2 摄像头路径")
    parser.add_argument("--seconds", type=_positive_float, default=10.0, help="录制时长，秒")
    parser.add_argument("--width", type=int, default=1280, help="采集宽度，px")
    parser.add_argument("--height", type=int, default=720, help="采集高度，px")
    parser.add_argument("--fps", type=float, default=30.0, help="目标帧率")
    parser.add_argument("--output", type=Path, help="输出 AVI 路径；省略则自动按时间命名")
    args = parser.parse_args()

    requested = Path(args.camera)
    device = requested.resolve() if requested.exists() else requested
    capture = cv2.VideoCapture(str(device), cv2.CAP_V4L2)
    if not capture.isOpened():
        print(f"⛔ 无法打开摄像头：{args.camera}（实际设备：{device}）")
        return 1

    writer: cv2.VideoWriter | None = None
    try:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or args.fps
        if width <= 0 or height <= 0:
            raise RuntimeError("摄像头未返回有效分辨率")

        output = args.output
        if output is None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output = OUTPUT_DIR / f"camera_{datetime.now():%Y%m%d_%H%M%S}.avi"
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("无法创建 MJPEG AVI 输出文件")

        print(f"▶ 开始录制：{args.seconds:.1f}s，{width}×{height} @ {fps:.1f} FPS")
        print(f"  输出：{output}")
        started = time.monotonic()
        frames = 0
        while time.monotonic() - started < args.seconds:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("摄像头中途未返回有效帧")
            writer.write(frame)
            frames += 1
        elapsed = time.monotonic() - started
        print(f"✅ 录制完成：{elapsed:.2f}s，{frames} 帧，实际约 {frames / elapsed:.1f} FPS")
        return 0
    except KeyboardInterrupt:
        print("\n⚠ 已中断；已尽力保存已写入的视频片段。")
        return 130
    except Exception as exc:
        print(f"⛔ 录制失败：{exc}")
        return 1
    finally:
        if writer is not None:
            writer.release()
        capture.release()


if __name__ == "__main__":
    raise SystemExit(main())
