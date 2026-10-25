"""集中实现视觉结果绘制、OpenCV 窗口和 Task1 调试观察器。"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Protocol

import cv2
import numpy as np

from ec_26_e.cam import CameraFrame
from ec_26_e.vision import Detect1Result, ResizeResult


DEFAULT_WINDOW_NAME = "EC 26 E - Live Detection"
DEFAULT_DISPLAY_NAME = ":0"
_WORKSPACE_COLOR = (255, 255, 0)
_WORKSPACE_CORNER_COLOR = (255, 0, 255)
_PIECE_COLOR = (0, 255, 0)
_PIECE_CORNER_COLOR = (0, 0, 255)
_LABEL_COLOR = (0, 255, 255)
_TEXT_COLOR = (255, 255, 255)
_FAILURE_COLOR = (0, 0, 255)
_TASK1_STATUS_SCALE_MULTIPLIER = 2.5
_EXIT_BUTTON_COLOR = (32, 32, 210)
_EXIT_BUTTON_ACTIVE_COLOR = (20, 20, 245)
_EXIT_BUTTON_TEXT_COLOR = (255, 255, 255)


class DrawError(RuntimeError):
    """表示绘制计算或本机 GUI 显示无法继续。"""


class FramePreview(Protocol):
    """约束绘制后 BGR 帧的非阻塞屏幕显示能力。"""

    def start(self) -> None:
        """创建窗口并准备接收帧。"""

    def show_and_should_stop(self, image_bgr: np.ndarray) -> bool:
        """显示最新帧，并返回用户是否请求中止任务。"""

    def close(self) -> None:
        """幂等关闭窗口及 GUI 资源。"""


def _validate_display_session(
    display_name: str | None,
) -> str:
    """在进入 Qt 前验证 X11 会话，避免后端直接终止进程。"""

    effective_name = (
        os.environ.get("DISPLAY")
        if display_name is None
        else display_name
    )
    if not effective_name:
        raise DrawError(
            "未设置可用的 DISPLAY；板端本地预览默认应连接 :0"
        )

    xdpyinfo = shutil.which("xdpyinfo")
    if xdpyinfo is None:
        raise DrawError(
            "找不到 xdpyinfo，无法在启动 OpenCV/Qt 前安全验证显示会话"
        )
    try:
        result = subprocess.run(
            [xdpyinfo, "-display", effective_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise DrawError(
            f"验证显示会话 {effective_name} 失败：{error}"
        ) from error
    if result.returncode != 0:
        reason = result.stderr.strip() or "X11 连接失败"
        raise DrawError(
            f"无法连接显示会话 {effective_name}：{reason}"
        )
    return effective_name


def _validate_image(image_bgr: np.ndarray) -> None:
    """校验绘制和显示使用的非空 uint8 BGR 图像。"""

    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr 必须是 NumPy 数组")
    if (
        image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or image_bgr.size == 0
    ):
        raise ValueError("image_bgr 必须是非空 H×W×3 图像")
    if image_bgr.dtype != np.uint8:
        raise ValueError("image_bgr 必须使用 uint8 像素类型")


def _draw_preview_exit_button(
    image_bgr: np.ndarray,
    *,
    active: bool = False,
) -> tuple[int, int, int, int]:
    """在显示副本右上角绘制退出按钮并返回可点击边界。"""

    height, width = image_bgr.shape[:2]
    margin = max(2, min(18, min(width, height) // 40))
    available_width = max(1, width - margin * 2)
    available_height = max(1, height - margin * 2)
    button_width = min(max(96, width // 7), available_width)
    button_height = min(max(44, height // 10), available_height)
    x2 = width - margin
    y1 = margin
    x1 = max(0, x2 - button_width)
    y2 = min(height, y1 + button_height)
    color = _EXIT_BUTTON_ACTIVE_COLOR if active else _EXIT_BUTTON_COLOR
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), _EXIT_BUTTON_TEXT_COLOR, 2)

    label = "EXIT"
    scale = max(0.45, min(1.2, button_height / 58.0))
    thickness = max(1, int(round(scale * 2)))
    (text_width, text_height), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )
    text_x = max(x1 + 2, x1 + (button_width - text_width) // 2)
    text_y = min(y2 - 4, y1 + (button_height + text_height) // 2)
    cv2.putText(
        image_bgr,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        _EXIT_BUTTON_TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )
    return x1, y1, x2, y2


def _drawing_style(
    image_bgr: np.ndarray,
) -> tuple[float, int, int]:
    """按画面短边生成有上下限的字号、线宽和点半径。"""

    short_side = min(image_bgr.shape[:2])
    scale = float(np.clip(short_side / 1080.0 * 0.7, 0.45, 0.9))
    thickness = int(np.clip(round(short_side / 540.0), 1, 2))
    radius = int(np.clip(round(short_side / 240.0), 2, 4))
    return scale, thickness, radius


def _workspace_corners(resize_result: ResizeResult) -> np.ndarray:
    """把 Resize 浮点源图四角转换为 OpenCV 整数多边形。"""

    if not isinstance(resize_result, ResizeResult):
        raise TypeError("resize_result 必须是 ResizeResult")
    return np.rint(
        np.asarray(
            resize_result.source_corners_px,
            dtype=np.float32,
        )
    ).astype(np.int32)


def _draw_workspace(
    image_bgr: np.ndarray,
    resize_result: ResizeResult,
) -> None:
    """在可写帧上绘制 A4 工作区边界、角点和方向标签。"""

    corners = _workspace_corners(resize_result)
    scale, thickness, radius = _drawing_style(image_bgr)
    cv2.polylines(
        image_bgr,
        [corners.reshape(-1, 1, 2)],
        True,
        _WORKSPACE_COLOR,
        thickness,
        cv2.LINE_AA,
    )
    for label, point in zip(("TL", "TR", "BR", "BL"), corners):
        coordinate = int(point[0]), int(point[1])
        cv2.circle(
            image_bgr,
            coordinate,
            radius,
            _WORKSPACE_CORNER_COLOR,
            -1,
        )
        cv2.putText(
            image_bgr,
            label,
            (coordinate[0] + radius + 2, coordinate[1] - radius - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            _WORKSPACE_COLOR,
            thickness,
            cv2.LINE_AA,
        )


def _draw_live_detection(
    image_bgr: np.ndarray,
    resize_result: ResizeResult,
    detect1_result: Detect1Result,
    *,
    sequence_id: int,
    status_scale_multiplier: float,
) -> np.ndarray:
    """按指定状态字号绘制工作区和逆映射后的 Detect1 结果。"""

    _validate_image(image_bgr)
    if not isinstance(detect1_result, Detect1Result):
        raise TypeError("detect1_result 必须是 Detect1Result")
    if isinstance(sequence_id, bool) or not isinstance(sequence_id, int):
        raise TypeError("sequence_id 必须是整数")

    annotated = image_bgr.copy()
    _draw_workspace(annotated, resize_result)
    _, thickness, radius = _drawing_style(annotated)
    inverse_homography = np.linalg.inv(resize_result.homography)
    for piece in detect1_result.pieces:
        rectified_corners = np.asarray(
            piece.corners_px,
            dtype=np.float32,
        ).reshape(1, -1, 2)
        source_corners = cv2.perspectiveTransform(
            rectified_corners,
            inverse_homography,
        )[0]
        source_polygon = np.rint(source_corners).astype(np.int32)
        cv2.polylines(
            annotated,
            [source_polygon.reshape(-1, 1, 2)],
            True,
            _PIECE_COLOR,
            thickness,
            cv2.LINE_AA,
        )
        for point in source_polygon:
            cv2.circle(
                annotated,
                (int(point[0]), int(point[1])),
                radius,
                _PIECE_CORNER_COLOR,
                -1,
            )
        rectified_center = np.asarray(
            piece.center_px,
            dtype=np.float32,
        ).reshape(1, 1, 2)
        source_center = cv2.perspectiveTransform(
            rectified_center,
            inverse_homography,
        )[0, 0]
        scale, _, _ = _drawing_style(annotated)
        cv2.putText(
            annotated,
            f"group {piece.group_id}",
            tuple(np.rint(source_center).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            _LABEL_COLOR,
            thickness,
            cv2.LINE_AA,
        )

    scale, thickness, _ = _drawing_style(annotated)
    scale *= status_scale_multiplier
    if status_scale_multiplier > 1.0:
        thickness = max(thickness, 2)
    status_origin = (20, max(35, int(round(30 * scale))))
    cv2.putText(
        annotated,
        f"frame={sequence_id} pieces={len(detect1_result.pieces)}",
        status_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        _PIECE_COLOR,
        thickness,
        cv2.LINE_AA,
    )
    return annotated


def draw_live_detection(
    image_bgr: np.ndarray,
    resize_result: ResizeResult,
    detect1_result: Detect1Result,
    *,
    sequence_id: int,
) -> np.ndarray:
    """使用通用字号绘制实时 Detect1 结果。"""

    return _draw_live_detection(
        image_bgr,
        resize_result,
        detect1_result,
        sequence_id=sequence_id,
        status_scale_multiplier=1.0,
    )


def _draw_detection_failure(
    image_bgr: np.ndarray,
    *,
    sequence_id: int,
    stage: str,
    resize_result: ResizeResult | None = None,
    status_scale_multiplier: float,
) -> np.ndarray:
    """按指定状态字号绘制失败阶段和可用的工作区边界。"""

    _validate_image(image_bgr)
    if isinstance(sequence_id, bool) or not isinstance(sequence_id, int):
        raise TypeError("sequence_id 必须是整数")
    if not isinstance(stage, str):
        raise TypeError("stage 必须是字符串")
    if not stage.strip():
        raise ValueError("stage 不能为空")

    annotated = image_bgr.copy()
    if resize_result is not None:
        _draw_workspace(annotated, resize_result)
    scale, thickness, _ = _drawing_style(annotated)
    scale *= status_scale_multiplier
    if status_scale_multiplier > 1.0:
        thickness = max(thickness, 2)
    status_origin = (20, max(35, int(round(30 * scale))))
    cv2.putText(
        annotated,
        f"frame={sequence_id} {stage}_failed",
        status_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        _FAILURE_COLOR,
        thickness,
        cv2.LINE_AA,
    )
    return annotated


def draw_detection_failure(
    image_bgr: np.ndarray,
    *,
    sequence_id: int,
    stage: str,
    resize_result: ResizeResult | None = None,
) -> np.ndarray:
    """使用通用字号绘制失败阶段和可用的工作区边界。"""

    return _draw_detection_failure(
        image_bgr,
        sequence_id=sequence_id,
        stage=stage,
        resize_result=resize_result,
        status_scale_multiplier=1.0,
    )


def draw_debug_result(
    image_bgr: np.ndarray,
    result: Detect1Result,
) -> np.ndarray:
    """在校正 A4 图副本上绘制 Detect1 分组、轮廓和角点序号。"""

    _validate_image(image_bgr)
    if not isinstance(result, Detect1Result):
        raise TypeError("result 必须是 Detect1Result")
    annotated = image_bgr.copy()
    scale, thickness, radius = _drawing_style(annotated)
    for piece in result.pieces:
        corners = np.rint(piece.corners_px).astype(np.int32)
        cv2.polylines(
            annotated,
            [corners.reshape(-1, 1, 2)],
            True,
            _PIECE_COLOR,
            thickness,
            cv2.LINE_AA,
        )
        for corner_index, (x, y) in enumerate(corners):
            point = int(x), int(y)
            cv2.circle(
                annotated,
                point,
                radius,
                _PIECE_CORNER_COLOR,
                -1,
            )
            cv2.putText(
                annotated,
                str(corner_index),
                (point[0] + radius + 2, point[1] - radius - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                _TEXT_COLOR,
                thickness,
                cv2.LINE_AA,
            )
        cv2.putText(
            annotated,
            f"group {piece.group_id}",
            tuple(np.rint(piece.center_px).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            _LABEL_COLOR,
            thickness,
            cv2.LINE_AA,
        )
    return annotated


def draw_video_failure(
    image_bgr: np.ndarray,
    *,
    frame_index: int,
    stage: str,
) -> np.ndarray:
    """在校正视频帧副本上绘制固定尺寸失败状态。"""

    return draw_detection_failure(
        image_bgr,
        sequence_id=frame_index,
        stage=stage,
    )


def draw_video_result(
    image_bgr: np.ndarray,
    result: Detect1Result,
    *,
    frame_index: int,
) -> np.ndarray:
    """在校正视频帧副本上绘制检测结果和帧序号。"""

    annotated = draw_debug_result(image_bgr, result)
    scale, thickness, _ = _drawing_style(annotated)
    cv2.putText(
        annotated,
        f"frame={frame_index} pieces={len(result.pieces)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        _PIECE_COLOR,
        thickness,
        cv2.LINE_AA,
    )
    return annotated


class OpenCVPreview:
    """通过板端 X11/Qt 普通窗口非阻塞显示最新绘制帧。"""

    def __init__(
        self,
        *,
        window_name: str = DEFAULT_WINDOW_NAME,
        display_name: str | None = DEFAULT_DISPLAY_NAME,
        fullscreen: bool = False,
        window_size: tuple[int, int] | None = None,
    ) -> None:
        """保存可选诊断显示参数；构造阶段不创建窗口。"""

        if not isinstance(window_name, str):
            raise TypeError("window_name 必须是字符串")
        if not window_name.strip():
            raise ValueError("window_name 不能为空")
        if display_name is not None and not isinstance(display_name, str):
            raise TypeError("display_name 必须是字符串或 None")
        if display_name is not None and not display_name.strip():
            raise ValueError("display_name 不能为空")
        if not isinstance(fullscreen, bool):
            raise TypeError("fullscreen 必须是布尔值")
        if window_size is not None:
            if (
                not isinstance(window_size, tuple)
                or len(window_size) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    for value in window_size
                )
            ):
                raise ValueError("window_size 必须是两个正整数的元组")
        self.window_name = window_name
        self.display_name = display_name
        self.fullscreen = fullscreen
        self.window_size = window_size
        self._started = False
        self._closed = False
        self._exit_requested = False
        self._exit_button_bounds: tuple[int, int, int, int] | None = None

    def start(self) -> None:
        """创建普通、全屏或指定尺寸窗口。"""

        if self._closed:
            raise DrawError("预览窗口已关闭，不能重新启动")
        if self._started:
            raise DrawError("预览窗口已经启动")
        effective_display = _validate_display_session(self.display_name)
        os.environ["DISPLAY"] = effective_display
        window_created = False
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            window_created = True
            cv2.setMouseCallback(self.window_name, self._handle_mouse_event)
            if self.fullscreen:
                cv2.setWindowProperty(
                    self.window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN,
                )
            elif self.window_size is not None:
                cv2.resizeWindow(self.window_name, *self.window_size)
                cv2.moveWindow(self.window_name, 0, 0)
        except cv2.error as error:
            if window_created:
                try:
                    cv2.destroyWindow(self.window_name)
                except cv2.error:
                    pass
            raise DrawError(
                f"无法创建预览窗口：{error}"
            ) from error
        self._started = True

    def _handle_mouse_event(
        self,
        event: int,
        x: int,
        y: int,
        _flags: int,
        _parameter: object,
    ) -> None:
        """在鼠标或触摸释放于退出按钮内时记录安全停止请求。"""

        if event != cv2.EVENT_LBUTTONUP or self._exit_button_bounds is None:
            return
        x1, y1, x2, y2 = self._exit_button_bounds
        if x1 <= x <= x2 and y1 <= y <= y2:
            self._exit_requested = True

    def show_and_should_stop(self, image_bgr: np.ndarray) -> bool:
        """显示最新帧；点击 EXIT、按 Esc 或关闭窗口时请求中止。"""

        if not self._started or self._closed:
            raise DrawError("预览窗口尚未启动或已经关闭")
        _validate_image(image_bgr)
        try:
            display_image = image_bgr.copy()
            self._exit_button_bounds = _draw_preview_exit_button(
                display_image,
                active=self._exit_requested,
            )
            cv2.imshow(self.window_name, display_image)
            key = cv2.waitKey(1) & 0xFF
            visible = cv2.getWindowProperty(
                self.window_name,
                cv2.WND_PROP_VISIBLE,
            )
        except cv2.error as error:
            raise DrawError(f"实时预览显示失败：{error}") from error
        return self._exit_requested or key == 27 or visible < 1

    def close(self) -> None:
        """幂等关闭窗口并处理最后一轮 GUI 事件。"""

        if self._closed:
            return
        self._closed = True
        if not self._started:
            return
        try:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass


class Task1DrawObserver:
    """把 Task1 强类型视觉事件绘制后发送到屏幕窗口。"""

    def __init__(self, preview: FramePreview | None = None) -> None:
        """保存可替换预览实现；默认连接板端 X11 :0 普通窗口。"""

        self.preview = (
            preview if preview is not None else OpenCVPreview()
        )

    def start(self) -> None:
        """启动底层预览窗口。"""

        self.preview.start()

    def show_detection(
        self,
        frame: CameraFrame,
        resize_result: ResizeResult,
        detect_result: Detect1Result,
    ) -> bool:
        """绘制并显示一张成功视觉检测帧。"""

        if not isinstance(frame, CameraFrame):
            raise TypeError("frame 必须是 CameraFrame")
        try:
            annotated = _draw_live_detection(
                frame.image_bgr,
                resize_result,
                detect_result,
                sequence_id=frame.sequence_id,
                status_scale_multiplier=_TASK1_STATUS_SCALE_MULTIPLIER,
            )
        except (cv2.error, np.linalg.LinAlgError) as error:
            raise DrawError(f"实时检测结果绘制失败：{error}") from error
        return self.preview.show_and_should_stop(annotated)

    def show_failure(
        self,
        frame: CameraFrame,
        *,
        stage: str,
        resize_result: ResizeResult | None = None,
    ) -> bool:
        """绘制并显示一张视觉处理失败帧。"""

        if not isinstance(frame, CameraFrame):
            raise TypeError("frame 必须是 CameraFrame")
        try:
            annotated = _draw_detection_failure(
                frame.image_bgr,
                sequence_id=frame.sequence_id,
                stage=stage,
                resize_result=resize_result,
                status_scale_multiplier=_TASK1_STATUS_SCALE_MULTIPLIER,
            )
        except cv2.error as error:
            raise DrawError(f"实时失败状态绘制失败：{error}") from error
        return self.preview.show_and_should_stop(annotated)

    def close(self) -> None:
        """关闭底层预览窗口。"""

        self.preview.close()
