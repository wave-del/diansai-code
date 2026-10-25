#!/usr/bin/env python3
"""视觉锁定拼图质心后，抓取、途中相对旋转 90°、放到 (10,10) mm。

预览阶段只读取摄像头。仅在用户输入 RUN 后，才复用已验证的真实取放流程。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import select
import signal
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

# Q6A QNN 显式加载 model-lib/q6a，不注入历史 A733 运行库。

# tests/a4_robust（YOLO 组合管线所在）
_TESTS_A4_ROBUST = Path(__file__).resolve().parent.parent / "tests" / "a4_robust"
if str(_TESTS_A4_ROBUST) not in sys.path:
    sys.path.insert(0, str(_TESTS_A4_ROBUST))

import cv2
import numpy as np

try:
    from . import test_a4_detection as a4_detection
    from .test_pick_rotate_place import main as motion_main
except ImportError:
    import test_a4_detection as a4_detection
    from test_pick_rotate_place import main as motion_main


# 现场蓝纸实测尺寸（不是标准 A4）：右上角为原点时，短边沿 X 向下，
# 长边沿 Y 向左。两值必须与物理蓝纸一致，不能使用标准 A4 的 210×297。
A4_X_MM = 206.5
A4_Y_MM = 292.0
# 现场任务约定：透视校正后蓝纸的左半边为待抓取区，右半边为已拼好区域。
# u=0 是校正画面的左边缘，u=1 是右边缘。
SOURCE_LEFT_MAX_U = 0.5


@dataclass(frozen=True)
class CentroidTarget:
    """冻结的一片拼图及其以写字机原点表示的质心位置。"""

    pickup_x_mm: float
    pickup_y_mm: float
    confidence: float
    piece_index: int


def _nonnegative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("必须为大于等于 0 的有限数字")
    return value


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


def _rpm(text: str) -> int:
    value = int(text)
    if not 1 <= value <= 30:
        raise argparse.ArgumentTypeError("RPM 必须是 1～30 的整数")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", default=a4_detection.PREFERRED_CAMERA, help="V4L2 摄像头路径")
    parser.add_argument("--camera-width", type=int, default=1280, help="相机宽度，px")
    parser.add_argument("--camera-height", type=int, default=720, help="相机高度，px")
    parser.add_argument("--camera-fps", type=int, default=30, help="相机请求帧率")
    parser.add_argument("--process-fps", type=_positive_float, default=6.0, help="视觉检测频率，Hz")
    parser.add_argument(
        "--params",
        choices=sorted(a4_detection.PARAM_PRESETS),
        default="current",
        help="参数预设：current=现有固定门槛；new=稳健自适应门槛（夜间/白天通用）",
    )
    parser.add_argument(
        "--yolo/--traditional",
        dest="detection",
        default="yolo",
        help="检测引擎：yolo=遮挡鲁棒seed+同事Resize+YOLO+七帧稳定（默认）；traditional=原传统管线",
    )
    parser.add_argument("--min-confidence", type=_ratio, default=0.85, help="允许抓取的最低拼图置信度")
    parser.add_argument("--place-x", type=_nonnegative_float, default=10.0, help="放件 X，mm")
    parser.add_argument("--place-y", type=_nonnegative_float, default=10.0, help="放件 Y，mm")
    parser.add_argument("--rotation-deg", type=float, default=90.0, help="搬运途中相对旋转角度，°")
    parser.add_argument("--safe-z", type=float, default=-8.0, help="搬运时抬笔 Z，mm；负数向上")
    parser.add_argument("--xy-feed", type=_positive_float, default=2500.0, help="XY 速度，mm/min")
    parser.add_argument("--z-feed", type=_positive_float, default=1500.0, help="Z 速度，mm/min")
    parser.add_argument("--rotation-rpm", type=_rpm, default=15, help="无刷速度，RPM")
    parser.add_argument("--writer-port", default="/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0", help="写字机串口")
    parser.add_argument("--motor-port", default="/dev/ttyAS3", help="F32C UART")
    parser.add_argument("--motor-address", type=int, default=2, help="F32C 地址")
    parser.add_argument("--preview-only", action="store_true", help="冻结并打印质心坐标，不连接任何机械")
    parser.add_argument("--windowed", action="store_true", help="窗口模式；默认全屏")
    return parser


def _a4_args(args: argparse.Namespace) -> argparse.Namespace:
    """构造与实时诊断脚本一致的蓝纸初定位参数（HSV 来自 --params 预设）。"""
    params = a4_detection._resolve_params(args)
    return argparse.Namespace(
        hue_min=int(params["hue_min"]),
        hue_max=int(params["hue_max"]),
        sat_min=int(params["sat_min"]),
        value_min=int(params["value_min"]),
        min_a4_area=0.25,
        max_refine_shift=35.0,
    )


class _VisionBackend:
    """统一视觉后端接口：detect/reset/close + mode 标签。

    detect(frame) 返回统一结果对象，字段：
    seed(SimpleNamespace: corners/reliability/mode)、resize_result、
    detect_result（稳定器输出，可能为空）、single_result（单帧）。
    """

    mode = "?"

    def detect(self, frame: Any) -> Any:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


class _TraditionalVision(_VisionBackend):
    """传统管线（原 _detect 链路），接口对齐。"""

    mode = "traditional"

    def __init__(self, params: dict[str, float | int], a4_args: Any) -> None:
        self._params = params
        self._a4_args = a4_args
        self._resize_class, self._resize_config, self._detector = (
            a4_detection._build_reference_detectors(params)
        )

    def detect(self, frame: Any) -> Any:
        resize_result, detect_result, seed_px, area_ratio, mask = (
            a4_detection._detect(
                frame,
                self._resize_class,
                self._resize_config,
                self._detector,
                self._a4_args,
            )
        )
        return SimpleNamespace(
            seed=SimpleNamespace(
                corners=seed_px,
                reliability=(True, True, True, True),
                mode="full",
                area_ratio=area_ratio,
                mask=mask,
            ),
            resize_result=resize_result,
            detect_result=detect_result,
            single_result=detect_result,
            timing_ms={},
        )


class _YoloVision(_VisionBackend):
    """YOLO 组合管线：遮挡鲁棒 seed + 同事 Resize + YOLO 门控精修 + 七帧稳定。

    仅在 yolo 模式下导入 yolo_pipeline（同事的 ec_26_e 包与
    传统管线的 reference-vision 同名，两者不能同进程混用）。
    """

    mode = "yolo"

    def __init__(self, params: dict[str, float | int], a4_args: Any) -> None:
        from yolo_pipeline import YoloRobustDetector

        self._detector = YoloRobustDetector(params, a4_args, stable=True)

    def detect(self, frame: Any) -> Any:
        return self._detector.detect(frame)

    def reset(self) -> None:
        self._detector.reset()

    def close(self) -> None:
        self._detector.close()


def _build_vision(args: argparse.Namespace) -> _VisionBackend:
    """按 --yolo/--traditional 构造统一视觉后端。"""
    params = a4_detection._resolve_params(args)
    a4_args = _a4_args(args)
    if args.detection == "yolo":
        return _YoloVision(params, a4_args)
    return _TraditionalVision(params, a4_args)


def _writer_centroid(piece: Any) -> tuple[float, float]:
    """校正 A4 左上角相对坐标转换为本机右上角写字机毫米坐标。"""
    u, v = piece.center_relative
    return A4_X_MM * float(v), A4_Y_MM * (1.0 - float(u))


def _is_pickup_candidate(piece: Any) -> bool:
    """只有校正蓝纸左半边的碎片可被抓取；右半边是目标拼图区。"""
    return float(piece.center_relative[0]) < SOURCE_LEFT_MAX_U


def _draw_selection(frame: np.ndarray, latest: Any | None, target: CentroidTarget | None, status: str) -> np.ndarray:
    """绘制完整原始画面和完整校正 A4，状态文字放在两者之外。"""
    canvas = np.full((1080, 1920, 3), (24, 24, 24), dtype=np.uint8)
    image = frame.copy()
    result = latest.resize_result if latest is not None else None
    detected = None
    if latest is not None:
        detected = (
            latest.detect_result
            if latest.detect_result.pieces
            else latest.single_result
        )
    if latest is not None and latest.seed is not None:
        seed = latest.seed
        for index, corner in enumerate(seed.corners):
            color = (0, 165, 255) if not seed.reliability[index] else (255, 0, 255)
            cv2.circle(image, tuple(np.rint(corner).astype(int)), 6, color, -1, cv2.LINE_AA)
    if result is not None:
        corners = np.rint(np.asarray(result.source_corners_px)).astype(np.int32)
        cv2.polylines(image, [corners.reshape(-1, 1, 2)], True, (255, 255, 0), 3, cv2.LINE_AA)
    if result is not None and detected is not None:
        inverse = np.linalg.inv(result.homography)
        for piece in detected.pieces:
            points = cv2.perspectiveTransform(np.asarray(piece.corners_px, dtype=np.float32).reshape(1, -1, 2), inverse)[0]
            center = cv2.perspectiveTransform(np.asarray(piece.center_px, dtype=np.float32).reshape(1, 1, 2), inverse)[0, 0]
            in_source = _is_pickup_candidate(piece)
            color = (0, 255, 0) if in_source else (90, 90, 90)
            if target is not None and piece.group_id == target.piece_index:
                color = (0, 255, 0)
            cv2.polylines(image, [np.rint(points).astype(np.int32).reshape(-1, 1, 2)], True, color, 3, cv2.LINE_AA)
            center_i = tuple(np.rint(center).astype(int))
            cv2.circle(image, center_i, 7, (0, 0, 255), -1, cv2.LINE_AA)
            x, y = _writer_centroid(piece)
            area = "SOURCE" if in_source else "IGNORED"
            cv2.putText(image, f"#{piece.group_id} {area}: X={x:.1f} Y={y:.1f} c={piece.confidence:.2f}", (center_i[0] + 8, center_i[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
    canvas[185:860, 0:1200] = cv2.resize(image, (1200, 675), interpolation=cv2.INTER_AREA)
    cv2.putText(canvas, "LIVE CAMERA / FULL ORIGINAL IMAGE", (28, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (235, 235, 235), 2, cv2.LINE_AA)

    if result is not None and detected is not None:
        rectified = result.image_bgr.copy()
        divider_x = int(round(rectified.shape[1] * SOURCE_LEFT_MAX_U))
        cv2.line(rectified, (divider_x, 0), (divider_x, rectified.shape[0] - 1), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(rectified, "PICKUP", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(rectified, "IGNORE", (divider_x + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (130, 130, 130), 2, cv2.LINE_AA)
        for piece in detected.pieces:
            corners = np.rint(np.asarray(piece.corners_px)).astype(np.int32)
            center = tuple(np.rint(np.asarray(piece.center_px)).astype(int))
            color = (0, 255, 0) if _is_pickup_candidate(piece) else (90, 90, 90)
            cv2.polylines(rectified, [corners.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
            cv2.circle(rectified, center, 5, (0, 0, 255), -1, cv2.LINE_AA)
        canvas[235:723, 1210:1900] = cv2.resize(rectified, (690, 488), interpolation=cv2.INTER_AREA)
        cv2.putText(canvas, "RECTIFIED / FULL A4 / DETECT1", (1220, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (235, 235, 235), 2, cv2.LINE_AA)

    state_color = (0, 255, 0) if target is not None else (0, 80, 255)
    cv2.putText(canvas, status[:120], (28, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.74, state_color, 2, cv2.LINE_AA)
    if target is not None:
        summary = f"Selected centroid: X={target.pickup_x_mm:.2f} mm  Y={target.pickup_y_mm:.2f} mm  confidence={target.confidence:.3f}"
    else:
        summary = "Red dot = detected polygon centroid; terminal RUN freezes one valid piece."
    cv2.putText(canvas, summary, (28, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.59, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Terminal: RUN freeze / q cancel   SPACE optional   This preview never moves hardware", (28, 1030), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (235, 235, 235), 2, cv2.LINE_AA)
    return canvas


def _select_target(args: argparse.Namespace, vision: _VisionBackend) -> CentroidTarget | None:
    capture = a4_detection._open_camera(
        argparse.Namespace(camera=args.camera, width=args.camera_width, height=args.camera_height, fps=args.camera_fps)
    )
    window = "Tangram Puzzle - Vision Centroid Pick"
    latest: Any | None = None
    latest_valid: list[Any] = []
    status = "等待蓝色 A4 与白色拼图..."
    frozen: CentroidTarget | None = None
    last_detection = -math.inf
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        if args.windowed:
            cv2.resizeWindow(window, args.camera_width, args.camera_height)
        else:
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        print("终端控制：输入 RUN 并回车冻结唯一有效拼图；输入 q 并回车取消。GUI Space 仅作备用。")
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("摄像头未返回有效帧")
            now = time.monotonic()
            if now - last_detection >= 1.0 / args.process_fps:
                last_detection = now
                try:
                    latest = vision.detect(frame)
                    stable_pieces = latest.detect_result.pieces
                    shown_pieces = (
                        stable_pieces if stable_pieces else latest.single_result.pieces
                    )
                    latest_valid = [
                        piece for piece in shown_pieces
                        if piece.confidence >= args.min_confidence and _is_pickup_candidate(piece)
                    ]
                    ignored_count = len(shown_pieces) - len(latest_valid)
                    stabilizing = (
                        "（七帧稳定中，单帧可见）"
                        if not stable_pieces and latest.single_result is not None and latest.single_result.pieces
                        else ""
                    )
                    status = (
                        f"{vision.mode.upper()} A4 OK；左半待抓={len(latest_valid)}，"
                        f"忽略={ignored_count}{stabilizing}；RUN 需要恰好 1 个"
                    )
                except Exception as exc:
                    latest = None
                    latest_valid = []
                    status = f"A4/拼图检测失败：{exc}"
            selected: CentroidTarget | None = None
            if len(latest_valid) == 1:
                piece = latest_valid[0]
                x, y = _writer_centroid(piece)
                selected = CentroidTarget(x, y, float(piece.confidence), int(piece.group_id))
            shown_target = frozen or selected
            cv2.imshow(window, _draw_selection(frame, latest, shown_target, status))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                return None
            # 不启动后台 input 线程，避免预览结束后线程抢走第二次 RUN 确认。
            # Linux PTY 可用 select 非阻塞地轮询终端输入，GUI 同时保持刷新。
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
            if readable:
                terminal_command = sys.stdin.readline().strip().upper()
                if not terminal_command or terminal_command == "Q":
                    return None
                if terminal_command != "RUN":
                    print("⛔ 预览阶段只接受 RUN（冻结）或 q（取消）。")
                elif frozen is None and selected is None:
                    print(f"⛔ 当前有效拼图数={len(latest_valid)}；必须恰好为 1，未冻结。")
                elif frozen is None:
                    frozen = selected
                    status = "质心已冻结；请确认机构在写字机自身原点，然后在终端再次输入 RUN"
                    print(
                        f"✅ 已冻结：X={frozen.pickup_x_mm:.3f} mm，Y={frozen.pickup_y_mm:.3f} mm。"
                        "预览持续显示；确认写字机自身原点后再次输入 RUN。"
                    )
                else:
                    return frozen
            if key == ord(" "):
                if frozen is None and selected is None:
                    print(f"⛔ 当前有效拼图数={len(latest_valid)}；必须恰好为 1。")
                elif frozen is None:
                    frozen = selected
                    status = "质心已冻结；请确认机构在写字机自身原点，终端再次 RUN 开始真实运动"
                else:
                    return frozen
    finally:
        capture.release()
        try:
            cv2.destroyWindow(window)
            cv2.waitKey(1)
        except cv2.error:
            pass


def _motion_command(args: argparse.Namespace, target: CentroidTarget, *, reuse_machine_origin: bool = False) -> list[str]:
    return [
        "--writer-port", args.writer_port,
        "--motor-port", args.motor_port,
        "--motor-address", str(args.motor_address),
        "--pickup-x", f"{target.pickup_x_mm:.3f}",
        "--pickup-y", f"{target.pickup_y_mm:.3f}",
        "--place-x", f"{args.place_x:.3f}",
        "--place-y", f"{args.place_y:.3f}",
        "--rotation-deg", f"{args.rotation_deg:.3f}",
        "--safe-z", f"{args.safe_z:.3f}",
        "--xy-feed", f"{args.xy_feed:.3f}",
        "--z-feed", f"{args.z_feed:.3f}",
        "--rotation-rpm", str(args.rotation_rpm),
        "--yes",
    ] + (["--reuse-machine-origin"] if reuse_machine_origin else [])


def _run_motion(args: argparse.Namespace, target: CentroidTarget, *, reuse_machine_origin: bool = False) -> int:
    return motion_main(_motion_command(args, target, reuse_machine_origin=reuse_machine_origin))


def _persistent_motion_worker(args: argparse.Namespace, command_queue: Any, result_queue: Any) -> None:
    """整个批量任务只打开一次串口，避免 CH340 DTR 脉冲复位 GRBL。"""
    try:
        from electromagnet import ElectromagnetController
        from f32c_controller import F32CMotor
        from writer_controller import WriterController

        writer = WriterController()
        motor = F32CMotor(port=args.motor_port, address=args.motor_address)
        magnet = ElectromagnetController()
        writer_open = motor_open = magnet_open = False
        origin_ready = False
        z_safe = False
        try:
            magnet.open(); magnet_open = True
            writer.connect(args.writer_port, baud=115200); writer_open = True
            writer.set_motor_hold(True)
            motor.connect(verify=False); motor_open = True
            motor.enable(rpm=args.rotation_rpm)
            result_queue.put(("ready", "硬件已连接；本次运行不会再次重开写字机串口"))

            while True:
                command = command_queue.get()
                if command[0] == "close":
                    result_queue.put(("closed", "硬件会话已关闭"))
                    return
                if command[0] != "run":
                    raise RuntimeError(f"未知硬件命令：{command[0]}")
                target: CentroidTarget = command[1]
                try:
                    motor.return_to_mechanical_zero(rpm=args.rotation_rpm, wait=True)
                    if not origin_ready:
                        # 仅第一轮：操作者已让机构位于写字机自身机械原点。
                        writer.home()
                        origin_ready = True
                    writer.set_motor_hold(True)
                    writer.configure_pen(up_z=args.safe_z, down_z=0.0, speed=args.z_feed)
                    writer.pen_up(); z_safe = True
                    writer.move_to(target.pickup_x_mm, target.pickup_y_mm, feedrate=args.xy_feed)
                    writer.pen_move_to(0.0, feedrate=args.z_feed); z_safe = False
                    magnet.engage(); time.sleep(0.5)
                    writer.pen_up(); z_safe = True

                    rotation_errors: list[BaseException] = []
                    def rotate() -> None:
                        try:
                            current = motor.get_angle()
                            motor.move_to((current + args.rotation_deg) % 360.0, rpm=args.rotation_rpm, wait=True)
                        except BaseException as exc:
                            rotation_errors.append(exc)
                    rotation = threading.Thread(target=rotate, daemon=True)
                    rotation.start()
                    writer.move_to(args.place_x, args.place_y, feedrate=args.xy_feed)
                    rotation.join()
                    if rotation_errors:
                        raise RuntimeError(f"无刷旋转失败：{rotation_errors[0]}")
                    writer.pen_move_to(0.0, feedrate=args.z_feed); z_safe = False
                    magnet.release(); time.sleep(0.5)
                    writer.pen_up(); z_safe = True
                    # 正常任务回蓝纸业务原点，绝不回机械上电原点。
                    writer.move_to(0.0, 0.0, feedrate=args.xy_feed)
                    writer.set_motor_hold(True)
                    result_queue.put(("done", "本轮完成；已回蓝纸坐标 (0,0)"))
                except BaseException as exc:
                    if writer_open and not z_safe:
                        try: writer.pen_up()
                        except BaseException: pass
                    if magnet_open:
                        try: magnet.release()
                        except BaseException: pass
                    result_queue.put(("error", f"{type(exc).__name__}: {exc}"))
                    return
        finally:
            if magnet_open:
                try: magnet.close(release=True)
                except BaseException: pass
            if motor_open:
                try: motor.close(disable=False)
                except BaseException: pass
            if writer_open:
                try: writer.set_motor_hold(True); writer.disconnect()
                except BaseException: pass
    except BaseException as exc:
        result_queue.put(("error", f"硬件会话启动失败：{type(exc).__name__}: {exc}"))


def _preview_during_motion(args: argparse.Namespace, target: CentroidTarget, process: Any, result_queue: Any, vision: _VisionBackend) -> tuple[str, str]:
    """机械子进程运行时持续显示实时原图、校正 A4 与当前检测结果。"""
    capture = a4_detection._open_camera(
        argparse.Namespace(camera=args.camera, width=args.camera_width, height=args.camera_height, fps=args.camera_fps)
    )
    window = "Tangram Puzzle - Vision Centroid Pick"
    latest: Any | None = None
    last_detection = -math.inf
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        if args.windowed:
            cv2.resizeWindow(window, args.camera_width, args.camera_height)
        else:
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while process.is_alive():
            try:
                result = result_queue.get_nowait()
                if result[0] in {"done", "error"}:
                    return result
            except queue.Empty:
                pass
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("摄像头未返回有效帧")
            now = time.monotonic()
            status = "真实取放执行中：预览持续刷新；请勿触碰机构，q/Esc 不会中断机械动作"
            if now - last_detection >= 1.0 / args.process_fps:
                last_detection = now
                try:
                    latest = vision.detect(frame)
                except Exception as exc:
                    latest = None
                    status = f"真实取放执行中；当前帧检测失败：{exc}"
            cv2.imshow(window, _draw_selection(frame, latest, target, status))
            cv2.waitKey(1)
        process.join()
        try:
            return result_queue.get_nowait()
        except queue.Empty:
            return ("error", f"硬件工作进程意外退出，exit={process.exitcode}")
    finally:
        capture.release()
        try:
            cv2.destroyWindow(window)
            cv2.waitKey(1)
        except cv2.error:
            pass


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not -8.0 <= args.safe_z < 0.0:
        print("⛔ --safe-z 只允许 -8mm 到 0mm；负数为抬起。", file=sys.stderr)
        return 2
    if not 0 <= args.motor_address <= 255:
        print("⛔ --motor-address 必须在 0～255。", file=sys.stderr)
        return 2
    if not 0.0 < abs(args.rotation_deg) <= 360.0:
        print("⛔ --rotation-deg 必须处于 (-360,0) 或 (0,360]。", file=sys.stderr)
        return 2
    print("批量视觉质心抓取测试：每轮在预览中输入两次 RUN；完成后自动回到预览等待下一片。")
    try:
        vision = _build_vision(args)
    except Exception as exc:
        print(f"⛔ 视觉引擎初始化失败：{exc}", file=sys.stderr)
        return 1
    print(f"视觉引擎：{vision.mode}（{'YOLO+七帧稳定' if vision.mode == 'yolo' else '传统管线'}）")
    hardware_process: Any | None = None
    command_queue: Any | None = None
    result_queue: Any | None = None

    def close_hardware_session() -> None:
        nonlocal hardware_process, command_queue
        if hardware_process is not None and hardware_process.is_alive() and command_queue is not None:
            command_queue.put(("close",))
            hardware_process.join(timeout=8.0)
        if hardware_process is not None and hardware_process.is_alive():
            os.kill(hardware_process.pid, signal.SIGINT)
            hardware_process.join(timeout=20.0)

    while True:
        try:
            target = _select_target(args, vision)
        except KeyboardInterrupt:
            close_hardware_session()
            vision.close()
            print("\n已通过 Ctrl+C 退出预览；未连接新的硬件动作。")
            return 130
        except (OSError, RuntimeError, cv2.error, ValueError) as exc:
            close_hardware_session()
            vision.close()
            print(f"⛔ 视觉预览失败：{exc}", file=sys.stderr)
            return 1
        if target is None:
            close_hardware_session()
            vision.close()
            print("已取消；未连接新的硬件动作。")
            return 0
        print()
        print("=" * 72)
        print("已确认视觉质心；现在启动真实取放，屏幕预览将持续刷新")
        print("=" * 72)
        print(f"拼图 #{target.piece_index}：质心 X={target.pickup_x_mm:.3f} mm，Y={target.pickup_y_mm:.3f} mm，置信度={target.confidence:.3f}")
        print(f"动作：吸取质心 → 途中相对旋转 {args.rotation_deg:+.1f}° → 放到 X={args.place_x:.3f}, Y={args.place_y:.3f} mm → 回蓝纸 (0,0)")
        if hardware_process is None:
            print("第一轮开始前：确认机构已在写字机自身原点，Z 自然下垂到下限，确认路径无人手。")
        else:
            print("本次硬件会话保持串口和机器坐标；请放入下一片，确认路径无人手。")
        if args.preview_only:
            vision.close()
            print("预览模式完成；未连接任何硬件。")
            return 0
        if hardware_process is None:
            context = mp.get_context("spawn")
            command_queue = context.Queue()
            result_queue = context.Queue()
            hardware_process = context.Process(
                target=_persistent_motion_worker,
                args=(args, command_queue, result_queue),
                name="tangram-persistent-motion",
            )
            hardware_process.start()
            try:
                kind, payload = result_queue.get(timeout=15.0)
            except queue.Empty:
                close_hardware_session()
                print("⛔ 硬件会话启动超时", file=sys.stderr)
                return 1
            if kind != "ready":
                close_hardware_session()
                print(f"⛔ 硬件会话启动失败：{payload}", file=sys.stderr)
                return 1
            print(f"▶ {payload}")
        assert hardware_process is not None and command_queue is not None and result_queue is not None
        command_queue.put(("run", target))
        try:
            kind, payload = _preview_during_motion(args, target, hardware_process, result_queue, vision)
        except KeyboardInterrupt:
            if hardware_process.is_alive():
                print("\n⛔ 正在向机械子进程转发 Ctrl+C，等待安全恢复...", file=sys.stderr)
                os.kill(hardware_process.pid, signal.SIGINT)
                hardware_process.join(timeout=20.0)
            vision.close()
            return 130
        except Exception as exc:
            close_hardware_session()
            vision.close()
            print(f"⛔ 运动期间预览失败：{exc}", file=sys.stderr)
            return 1
        if kind == "error":
            close_hardware_session()
            vision.close()
            print(f"⛔ 机械子进程异常：{payload}", file=sys.stderr)
            return 1
        print(f"\n✅ {payload}。请放入下一片待抓取拼图；预览即将重新开始。")
        # 场景已变化（拼图被抓走）：清空稳定器轨迹
        vision.reset()


if __name__ == "__main__":
    raise SystemExit(main())
