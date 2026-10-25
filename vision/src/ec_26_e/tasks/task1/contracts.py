"""定义 Task1 的运行选项、可替换依赖协议和失败契约。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Literal, Protocol

import numpy as np

from ec_26_e.actuation import ActuationResult
from ec_26_e.cam import CameraFrame
from solver2 import (
    MovePlan,
    PlanningOptions,
    PuzzlePiece,
)
from ec_26_e.vision import Detect1Result, ResizeResult


LandscapeRotation = Literal["clockwise", "counterclockwise"]
_A4_DIAGONAL_TOLERANCE_LIMIT_MM = 357.93


def _validate_positive_number(name: str, value: float) -> None:
    """校验配置项为有限正数，并拒绝布尔值冒充数字。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是实数")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须是有限正数")


def _validate_positive_integer(name: str, value: int) -> None:
    """校验配置项为正整数。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须是整数")
    if value <= 0:
        raise ValueError(f"{name} 必须是正整数")


def _validate_nonnegative_integer(name: str, value: int) -> None:
    """校验跳变帧容忍数量为非负整数。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须是整数")
    if value < 0:
        raise ValueError(f"{name} 不能小于 0")


@dataclass(frozen=True, slots=True)
class Task1Options:
    """描述共享稳定审查、坐标方向、采集超时和控制求解选项。

    Task1 与 Task2 共用这组采集参数；各自的强类型配置会在装配时
    限制允许的拼图模式，避免把任务名称耦合进通用审查流程。
    """

    minimum_stable_frames: int = 5
    piece_vertex_tolerance_mm: float = 2.0
    workspace_corner_x_tolerance_mm: float = 2.0
    workspace_corner_y_tolerance_mm: float = 2.0
    workspace_corner_distance_tolerance_mm: float = 2.5
    review_cycle_seconds: float = 2.0
    allowed_jump_frames: int = 5
    acquisition_timeout_seconds: float = 5.0
    frame_read_timeout_seconds: float = 2.0
    landscape_rotation: LandscapeRotation = "counterclockwise"
    planning_options: PlanningOptions = PlanningOptions()

    def __post_init__(self) -> None:
        """在启动相机前校验全部独立值和跨字段约束。"""

        _validate_positive_integer(
            "minimum_stable_frames",
            self.minimum_stable_frames,
        )
        if self.minimum_stable_frames < 2:
            raise ValueError("minimum_stable_frames 不能小于 2")
        _validate_positive_number(
            "piece_vertex_tolerance_mm",
            self.piece_vertex_tolerance_mm,
        )
        if (
            self.piece_vertex_tolerance_mm
            > _A4_DIAGONAL_TOLERANCE_LIMIT_MM
        ):
            raise ValueError("piece_vertex_tolerance_mm 不能超过 A4 对角线")
        _validate_positive_number(
            "workspace_corner_x_tolerance_mm",
            self.workspace_corner_x_tolerance_mm,
        )
        if self.workspace_corner_x_tolerance_mm > 292.0:
            raise ValueError(
                "workspace_corner_x_tolerance_mm 必须位于 (0, 292]"
            )
        _validate_positive_number(
            "workspace_corner_y_tolerance_mm",
            self.workspace_corner_y_tolerance_mm,
        )
        if self.workspace_corner_y_tolerance_mm > 207.0:
            raise ValueError(
                "workspace_corner_y_tolerance_mm 必须位于 (0, 207]"
            )
        _validate_positive_number(
            "workspace_corner_distance_tolerance_mm",
            self.workspace_corner_distance_tolerance_mm,
        )
        if (
            self.workspace_corner_distance_tolerance_mm
            > _A4_DIAGONAL_TOLERANCE_LIMIT_MM
        ):
            raise ValueError(
                "workspace_corner_distance_tolerance_mm 必须位于 "
                "(0, 357.93]"
            )
        _validate_positive_number(
            "review_cycle_seconds",
            self.review_cycle_seconds,
        )
        _validate_nonnegative_integer(
            "allowed_jump_frames",
            self.allowed_jump_frames,
        )
        _validate_positive_number(
            "acquisition_timeout_seconds",
            self.acquisition_timeout_seconds,
        )
        _validate_positive_number(
            "frame_read_timeout_seconds",
            self.frame_read_timeout_seconds,
        )
        if (
            self.acquisition_timeout_seconds
            < self.review_cycle_seconds
        ):
            raise ValueError(
                "acquisition_timeout_seconds 不能小于审查周期"
            )
        if self.landscape_rotation not in {"clockwise", "counterclockwise"}:
            raise ValueError(
                "landscape_rotation 必须为 clockwise 或 counterclockwise"
            )
        if not isinstance(self.planning_options, PlanningOptions):
            raise TypeError("planning_options 必须是 PlanningOptions")


class Task1Error(RuntimeError):
    """表示 Task1 编排或依赖契约无法继续执行。"""


class Task1TimeoutError(Task1Error, TimeoutError):
    """表示 Task1 在采集期限内没有得到稳定可规划结果。"""


class ResizeProcessor(Protocol):
    """约束 Task1 所需的无状态 A4 透视校正能力。"""

    def resize(self, image_bgr: np.ndarray) -> ResizeResult:
        """把一张相机 BGR 帧校正为横向 A4 图像。"""


class Detect1Processor(Protocol):
    """约束 Task1 所需的横向 A4 纸片检测能力。"""

    def detect(self, image_bgr: np.ndarray) -> Detect1Result:
        """检测校正图中的纸片并返回冻结结果对象。"""


class Task1FrameObserver(Protocol):
    """约束 Task1 可选实时视觉事件观察器及其生命周期。"""

    def start(self) -> None:
        """准备接收视觉事件。"""

    def show_detection(
        self,
        frame: CameraFrame,
        resize_result: ResizeResult,
        detect_result: Detect1Result,
    ) -> bool:
        """显示成功检测帧，并返回用户是否请求中止。"""

    def show_failure(
        self,
        frame: CameraFrame,
        *,
        stage: str,
        resize_result: ResizeResult | None = None,
    ) -> bool:
        """显示失败帧，并返回用户是否请求中止。"""

    def close(self) -> None:
        """释放观察器资源。"""


class PuzzlePlanner(Protocol):
    """约束 Task1 所需的同步、无副作用拼图规划能力。"""

    def __call__(
        self,
        pieces: Sequence[PuzzlePiece],
        options: PlanningOptions,
    ) -> MovePlan:
        """把稳定毫米轮廓转换为移动方案。"""


class PlanActuator(Protocol):
    """约束 Task1 所需的同步 MovePlan 执行能力。"""

    def execute(self, plan: MovePlan) -> ActuationResult | None:
        """执行内存计划并在 FINISH 后完成安全退出。"""
