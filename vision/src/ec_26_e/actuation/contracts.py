"""定义 MovePlan 装配、无反馈执行和磁铁输出的稳定契约。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Protocol, runtime_checkable

from ec_26_e.motion import (
    FocRotationConfig,
    MotionSerialConfig,
    ServoReply,
)


def _finite(name: str, value: float) -> float:
    """校验一个有限实数并返回浮点值。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是实数")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数")
    return number


def _positive(name: str, value: float) -> float:
    """校验一个有限正数。"""

    number = _finite(name, value)
    if number <= 0:
        raise ValueError(f"{name} 必须大于 0")
    return number


def _nonnegative(name: str, value: float) -> float:
    """校验一个有限非负数。"""

    number = _finite(name, value)
    if number < 0:
        raise ValueError(f"{name} 不能小于 0")
    return number


@dataclass(frozen=True, slots=True)
class MagnetConfig:
    """描述 Linux GPIO 字符设备中的磁铁高电平输出。"""

    chip_path: Path = Path("/dev/gpiochip0")
    line_offset: int = -1

    def __post_init__(self) -> None:
        """在访问 GPIO 前校验芯片路径和 line offset。"""

        if not isinstance(self.chip_path, Path):
            raise TypeError("chip_path 必须是 pathlib.Path")
        if not str(self.chip_path):
            raise ValueError("chip_path 不能为空")
        if isinstance(self.line_offset, bool) or not isinstance(
            self.line_offset,
            int,
        ):
            raise TypeError("line_offset 必须是整数")
        if self.line_offset < 0:
            raise ValueError("line_offset 必须通过 MOTION_MAGNET_GPIO_LINE 配置")


@dataclass(frozen=True, slots=True)
class ActuationConfig:
    """聚合 Task1 运动适配所需的机器部署和时序参数。"""

    serial: MotionSerialConfig
    magnet: MagnetConfig
    rotation: FocRotationConfig = FocRotationConfig()
    x_min_mm: float = 0.0
    x_max_mm: float = 230.0
    y_min_mm: float = -10.0
    y_max_mm: float = 360.0
    # 标定约定：电机组合零点向 X/Y 正方向分别移动 14.5/6.5 mm，
    # 即可到达 A4 零点。
    a4_origin_x_mm: float = 14.5
    a4_origin_y_mm: float = 6.5
    xy_speed_mm_per_second: float = 80.0
    xy_fixed_wait_seconds: float = 1.0
    z_boot_degrees: float = 0.0
    z_up_degrees: float = 90.0
    z_down_degrees: float = 0.0
    z_speed_level: int = 64
    z_enable_wait_seconds: float = 1.0
    z_move_wait_seconds: float = 1.0
    magnet_attach_wait_seconds: float = 0.3
    magnet_release_wait_seconds: float = 0.3
    rotation_zero_wait_seconds: float = 1.0

    def __post_init__(self) -> None:
        """校验全部独立参数及软件限位的组合关系。"""

        if not isinstance(self.serial, MotionSerialConfig):
            raise TypeError("serial 必须是 MotionSerialConfig")
        if not isinstance(self.magnet, MagnetConfig):
            raise TypeError("magnet 必须是 MagnetConfig")
        if not isinstance(self.rotation, FocRotationConfig):
            raise TypeError("rotation 必须是 FocRotationConfig")
        x_min = _finite("x_min_mm", self.x_min_mm)
        x_max = _finite("x_max_mm", self.x_max_mm)
        y_min = _finite("y_min_mm", self.y_min_mm)
        y_max = _finite("y_max_mm", self.y_max_mm)
        if not 0.0 <= x_min < x_max <= 230.0:
            raise ValueError("X 软件限位必须满足 0 <= min < max <= 230")
        if not -10.0 <= y_min < y_max <= 360.0:
            raise ValueError("Y 软件限位必须满足 -10 <= min < max <= 360")
        _finite("a4_origin_x_mm", self.a4_origin_x_mm)
        _finite("a4_origin_y_mm", self.a4_origin_y_mm)
        _positive(
            "xy_speed_mm_per_second",
            self.xy_speed_mm_per_second,
        )
        _nonnegative(
            "xy_fixed_wait_seconds",
            self.xy_fixed_wait_seconds,
        )
        _finite("z_boot_degrees", self.z_boot_degrees)
        _finite("z_up_degrees", self.z_up_degrees)
        _finite("z_down_degrees", self.z_down_degrees)
        if self.z_up_degrees == self.z_down_degrees:
            raise ValueError("Z 抬起角度和落下角度不能相同")
        if isinstance(self.z_speed_level, bool) or not isinstance(
            self.z_speed_level,
            int,
        ):
            raise TypeError("z_speed_level 必须是整数")
        if not 1 <= self.z_speed_level <= 127:
            raise ValueError("z_speed_level 必须位于 [1, 127]")
        _positive(
            "z_enable_wait_seconds",
            self.z_enable_wait_seconds,
        )
        _positive("z_move_wait_seconds", self.z_move_wait_seconds)
        _nonnegative(
            "magnet_attach_wait_seconds",
            self.magnet_attach_wait_seconds,
        )
        _nonnegative(
            "magnet_release_wait_seconds",
            self.magnet_release_wait_seconds,
        )
        _positive(
            "rotation_zero_wait_seconds",
            self.rotation_zero_wait_seconds,
        )


@dataclass(frozen=True, slots=True)
class TransferCycle:
    """描述一片纸的量化拾取和放置机器绝对坐标。"""

    order: int
    piece_id: str
    pick_x_0p01mm: int
    pick_y_0p01mm: int
    place_x_0p01mm: int
    place_y_0p01mm: int
    rotation_degrees_clockwise: float


class MotionLifecycleState(str, Enum):
    """列出一次 MovePlan 从接收到安全退出的顶层状态。"""

    IDLE = "idle"
    VALIDATING = "validating"
    ENABLING = "enabling"
    INITIAL_Z_RAISING = "initial_z_raising"
    EXECUTING = "executing"
    FINISH = "finish"
    SHUTTING_DOWN = "shutting_down"
    EXITED = "exited"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class ActuationResult:
    """描述 MovePlan 完成及随后安全退出的最终软件结果。"""

    state: MotionLifecycleState
    completed_cycles: int
    state_history: tuple[MotionLifecycleState, ...]


class ActuationError(RuntimeError):
    """表示 Task1 运动适配层无法安全完成。"""


class ActuationValidationError(ActuationError, ValueError):
    """表示 MovePlan 或机器配置未通过运动前安全检查。"""


class ActuationExecutionError(ActuationError):
    """表示 MovePlan 尚未 FINISH 时发生运动异常。"""


class ActuationShutdownError(ActuationError):
    """表示 MovePlan 已 FINISH，但电机失能或资源释放失败。"""


@runtime_checkable
class MagnetOutput(Protocol):
    """约束磁铁 GPIO 输出的显式生命周期。"""

    def open(self) -> None:
        """申请 GPIO line 并立即输出低电平。"""

    def set_enabled(self, enabled: bool) -> None:
        """使用高电平上磁、低电平消磁。"""

    def close(self) -> None:
        """在低电平状态释放 GPIO line。"""


@runtime_checkable
class CompletionIndicator(Protocol):
    """约束任务完成提示灯的显式 GPIO 生命周期。"""

    def open(self) -> None:
        """申请提示灯 GPIO 并立即输出低电平。"""

    def set_enabled(self, enabled: bool) -> None:
        """使用高电平点亮、低电平熄灭提示灯。"""

    def close(self) -> None:
        """在低电平状态释放提示灯 GPIO。"""


@runtime_checkable
class NoReplyTransport(Protocol):
    """约束生命周期执行器使用的无回包串口能力。"""

    def open(self) -> None:
        """打开串口。"""

    def send_frame(self, frame: bytes) -> None:
        """完整写入一条报文。"""

    def discard_replies(self) -> None:
        """丢弃设备可能产生但任务不依赖的回包。"""

    def take_reply(self) -> ServoReply | None:
        """兼容 XY 状态机并立即返回可选回包。"""

    def close(self) -> None:
        """关闭串口。"""


@runtime_checkable
class RotationAxis(Protocol):
    """约束 Task1 使用的定向多圈旋转轴能力。"""

    def open(self) -> None:
        """打开串口并使能旋转轴。"""

    def current_angle_deg(self) -> float:
        """返回持久机械单圈角度。"""

    def move_clockwise_from_zero(
        self,
        target_degrees: float,
        *,
        wait: bool = True,
    ) -> float:
        """从机械零点顺时针旋转至目标。"""

    def return_counterclockwise_to_zero(
        self,
        target_degrees: float,
        *,
        wait: bool = True,
    ) -> float:
        """从目标位置逆时针沿原角度回零。"""

    def close(self) -> None:
        """失能旋转轴并关闭串口。"""
