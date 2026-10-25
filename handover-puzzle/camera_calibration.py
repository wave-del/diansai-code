#!/usr/bin/env python3
"""交互式 USB 摄像头棋盘格标定；只读取摄像头，不连接任何机械硬件。

当前接入的棋盘格为 6×4 个内部角点，即 7×5 个黑白方格。运行时
按 Space 保存当前有效画面，至少采集 15 张；按 C 计算并保存标定参数。
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

if os.path.isdir("/usr/share/fonts/truetype/dejavu"):
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
PREFERRED_CAMERA = "/dev/v4l/by-id/usb-Generic_USB_Camera3_200901010001-video-index0"
DEFAULT_OUTPUT = PACKAGE_DIR / "calibration" / "camera_1280x720.json"


@dataclass(frozen=True)
class CalibrationSample:
    """一张成功检测到完整棋盘格的标定样本。"""

    corners: np.ndarray
    center_px: tuple[float, float]


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("必须是正的有限数字")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", default=PREFERRED_CAMERA, help="V4L2 摄像头路径；可填写 /dev/v4l/by-id 稳定链接")
    parser.add_argument("--width", type=_positive_int, default=1280, help="采集宽度，px；必须与后续视觉运行时一致")
    parser.add_argument("--height", type=_positive_int, default=720, help="采集高度，px；必须与后续视觉运行时一致")
    parser.add_argument("--fps", type=_positive_int, default=30, help="相机请求帧率")
    parser.add_argument("--detect-fps", type=_positive_float, default=3.0, help="预览期间后台棋盘格检测频率，Hz；Space 始终使用原图精确检测")
    parser.add_argument("--corners-x", type=_positive_int, default=6, help="棋盘格横向内部角点数")
    parser.add_argument("--corners-y", type=_positive_int, default=4, help="棋盘格纵向内部角点数")
    parser.add_argument("--square-mm", type=_positive_float, default=25.0, help="实际测得的单个方格边长，mm")
    parser.add_argument("--min-samples", type=_positive_int, default=15, help="允许计算标定所需的最少有效样本数")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 JSON 标定文件")
    parser.add_argument("--windowed", action="store_true", help="窗口模式；默认全屏显示")
    return parser


def _resolve_camera_path(camera: str) -> Path:
    requested = Path(camera)
    return requested.resolve() if requested.exists() else requested


def _open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    path = _resolve_camera_path(args.camera)
    capture = cv2.VideoCapture(str(path), cv2.CAP_V4L2)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开摄像头：{args.camera}（实际设备：{path}）")
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    return capture


def _find_corners(
    gray: np.ndarray,
    pattern_size: tuple[int, int],
    *,
    precise: bool,
) -> np.ndarray | None:
    """在均衡灰度图中寻找角点；精确模式只在采样时调用。"""
    image = cv2.equalizeHist(gray)
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        if precise:
            flags |= cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(image, pattern_size, flags)
        if found and corners is not None:
            return corners.astype(np.float32).reshape(-1, 1, 2)
    found, corners = cv2.findChessboardCorners(
        image,
        pattern_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if found and corners is not None:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            40,
            0.001,
        )
        return cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria).reshape(-1, 1, 2)
    return None


def _object_points(pattern_size: tuple[int, int], square_mm: float) -> np.ndarray:
    columns, rows = pattern_size
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    return points * square_mm


def _draw_overlay(
    frame_bgr: np.ndarray,
    corners: np.ndarray | None,
    samples: list[CalibrationSample],
    args: argparse.Namespace,
    message: str,
) -> np.ndarray:
    image = frame_bgr.copy()
    height, width = image.shape[:2]
    if corners is not None:
        cv2.drawChessboardCorners(image, (args.corners_x, args.corners_y), corners, True)
        state, color = "棋盘格完整：可按 SPACE 采集", (0, 255, 0)
    else:
        state, color = "未检测到完整棋盘格：调整距离、角度或光照", (0, 80, 255)
    cv2.rectangle(image, (0, 0), (width, 115), (20, 20, 20), -1)
    cv2.putText(image, state, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
    cv2.putText(
        image,
        f"samples={len(samples)}/{args.min_samples}  board={args.corners_x}x{args.corners_y} inner corners  square={args.square_mm:g} mm",
        (24, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(image, "SPACE capture   C calibrate/save   R reset   ESC/Q quit", (24, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    if message:
        cv2.putText(image, message[:105], (24, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    for sample in samples:
        cv2.circle(image, tuple(np.rint(sample.center_px).astype(int)), 5, (255, 0, 255), -1, cv2.LINE_AA)
    return image


def _per_view_errors(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> list[float]:
    errors: list[float] = []
    for object_set, image_set, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(object_set, rvec, tvec, camera_matrix, distortion)
        errors.append(float(cv2.norm(image_set, projected, cv2.NORM_L2) / len(projected)))
    return errors


def _calibrate(samples: list[CalibrationSample], image_size: tuple[int, int], pattern_size: tuple[int, int], square_mm: float) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    object_template = _object_points(pattern_size, square_mm)
    object_sets = [object_template.copy() for _ in samples]
    image_sets = [sample.corners for sample in samples]
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(object_sets, image_sets, image_size, None, None)
    errors = _per_view_errors(object_sets, image_sets, rvecs, tvecs, matrix, distortion)
    new_matrix, valid_roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, image_size, 1.0, image_size)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "opencv_pinhole_rational",
        "image_size_px": [int(image_size[0]), int(image_size[1])],
        "chessboard": {
            "internal_corners": [int(pattern_size[0]), int(pattern_size[1])],
            "square_size_mm": float(square_mm),
            "sample_count": len(samples),
        },
        "rms_reprojection_error_px": float(rms),
        "per_view_mean_error_px": [float(error) for error in errors],
        "camera_matrix": matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "optimal_new_camera_matrix_alpha_1": new_matrix.tolist(),
        "valid_roi_xywh": [int(value) for value in valid_roi],
    }
    return payload, matrix, distortion


def _save_calibration(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.corners_x < 3 or args.corners_y < 3:
        print("⛔ --corners-x 和 --corners-y 均至少为 3。", file=sys.stderr)
        return 2
    if args.min_samples < 10:
        print("⛔ 至少需要 10 张样本；建议 20～30 张。", file=sys.stderr)
        return 2
    pattern_size = (args.corners_x, args.corners_y)
    samples: list[CalibrationSample] = []
    capture: cv2.VideoCapture | None = None
    detector_executor: ThreadPoolExecutor | None = None
    window = "Tangram Puzzle - Camera Calibration"
    print("=" * 72)
    print("相机棋盘格标定：只读取摄像头，不会连接任何机械硬件。")
    print(f"棋盘格：{args.corners_x}×{args.corners_y} 内部角点；单格={args.square_mm:g} mm。")
    print("请在中心、四角、四边采集 20～30 张；每张必须完整可见，额外采几张轻微倾斜画面。")
    print("=" * 72)
    try:
        capture = _open_camera(args)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        if args.windowed:
            cv2.resizeWindow(window, args.width, args.height)
        else:
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        detector_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chessboard-preview")
        pending_detection: Future[np.ndarray | None] | None = None
        latest_corners: np.ndarray | None = None
        last_detection_submit = -math.inf
        message = "移动棋盘格；完整角点变绿后按 SPACE 采集。"
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("摄像头未返回有效画面")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            now = time.monotonic()
            if pending_detection is not None and pending_detection.done():
                latest_corners = pending_detection.result()
                pending_detection = None
            if pending_detection is None and now - last_detection_submit >= 1.0 / args.detect_fps:
                pending_detection = detector_executor.submit(
                    _find_corners,
                    gray.copy(),
                    pattern_size,
                    precise=False,
                )
                last_detection_submit = now
            preview = _draw_overlay(frame, latest_corners, samples, args, message)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                print("已退出；未保存标定参数。")
                return 0
            if key in (ord("r"), ord("R")):
                samples.clear()
                message = "已清空内存中的全部样本。"
            elif key == ord(" "):
                precise_corners = _find_corners(gray, pattern_size, precise=True)
                if precise_corners is None:
                    message = "当前未检测到完整棋盘格；本次未采集。"
                else:
                    center = tuple(np.mean(precise_corners.reshape(-1, 2), axis=0).astype(float))
                    samples.append(CalibrationSample(corners=precise_corners.copy(), center_px=center))
                    latest_corners = precise_corners
                    message = f"已采集第 {len(samples)} 张；请移动到新的位置或倾角。"
                    print(message)
            elif key in (ord("c"), ord("C")):
                if len(samples) < args.min_samples:
                    message = f"样本不足：{len(samples)}/{args.min_samples}；建议继续采集到 20～30 张。"
                    print(f"⛔ {message}")
                    continue
                height, width = frame.shape[:2]
                payload, matrix, distortion = _calibrate(samples, (width, height), pattern_size, args.square_mm)
                _save_calibration(args.output, payload)
                maximum = max(payload["per_view_mean_error_px"])
                print()
                print(f"✅ 标定已保存：{args.output}")
                print(f"RMS 重投影误差：{payload['rms_reprojection_error_px']:.3f} px；单帧最大均值误差：{maximum:.3f} px")
                print("经验判断：≤0.5 px 很好；0.5～1.0 px 可用；>1.0 px 请增加边缘/倾斜样本后重新标定。")
                undistorted = cv2.undistort(frame, matrix, distortion)
                output = args.output.with_name(f"{args.output.stem}_undistorted_preview.jpg")
                cv2.imwrite(str(output), undistorted)
                print(f"已保存最后一帧去畸变预览：{output}")
                return 0
    except KeyboardInterrupt:
        print("\n已通过 Ctrl+C 退出；未保存新的标定参数。")
        return 130
    except (OSError, RuntimeError, cv2.error) as exc:
        print(f"⛔ 标定失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if capture is not None:
            capture.release()
        if detector_executor is not None:
            detector_executor.shutdown(wait=False, cancel_futures=True)
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except cv2.error:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
