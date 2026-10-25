"""定义实体按键绑定、消抖状态和常驻服务配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ButtonBinding:
    """把一个排针物理脚映射为一个正式任务。"""

    physical_pin: int
    chip_path: Path
    line_offset: int
    task_name: str

    def __post_init__(self) -> None:
        """拒绝非法排针、GPIO和任务映射。"""

        if isinstance(self.physical_pin, bool) or self.physical_pin <= 0:
            raise ValueError("physical_pin 必须是正整数")
        if not isinstance(self.chip_path, Path):
            raise TypeError("chip_path 必须是 pathlib.Path")
        if isinstance(self.line_offset, bool) or self.line_offset < 0:
            raise ValueError("line_offset 必须是非负整数")
        if not isinstance(self.task_name, str) or not self.task_name.strip():
            raise ValueError("task_name 必须是非空字符串")


def create_a7z_button_bindings(
    *,
    task1_chip_path: Path = Path("/dev/gpiochip0"),
    task1_line_offset: int = 38,
    task2_chip_path: Path = Path("/dev/gpiochip1"),
    task2_line_offset: int = 36,
) -> tuple[ButtonBinding, ButtonBinding]:
    """创建本机A7Z物理35、37脚到Task1、Task2的唯一正式映射。"""

    return (
        ButtonBinding(35, task1_chip_path, task1_line_offset, "task1"),
        ButtonBinding(37, task2_chip_path, task2_line_offset, "task2"),
    )


@dataclass(frozen=True, slots=True)
class ButtonServiceConfig:
    """保存按键服务的硬件映射、时间边界和运行路径。"""

    bindings: tuple[ButtonBinding, ...]
    active_link: Path
    status_path: Path
    lock_path: Path
    debounce_seconds: float = 0.03
    poll_interval_seconds: float = 0.01
    release_guard_seconds: float = 0.10
    conflict_window_seconds: float = 0.10
    retry_window_seconds: float = 120.0
    retry_delay_seconds: float = 2.0
    cleanup_timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        """校验唯一任务映射、唯一GPIO和所有正时间参数。"""

        if not isinstance(self.bindings, tuple) or not self.bindings:
            raise ValueError("bindings 必须是非空元组")
        if any(not isinstance(item, ButtonBinding) for item in self.bindings):
            raise TypeError("bindings 只能包含 ButtonBinding")
        tasks = [item.task_name for item in self.bindings]
        lines = [(item.chip_path, item.line_offset) for item in self.bindings]
        if len(set(tasks)) != len(tasks):
            raise ValueError("每个任务只能绑定一个按键")
        if len(set(lines)) != len(lines):
            raise ValueError("每根GPIO线只能绑定一个按键")
        for name in ("active_link", "status_path", "lock_path"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} 必须是 pathlib.Path")
        for name in (
            "debounce_seconds",
            "poll_interval_seconds",
            "release_guard_seconds",
            "conflict_window_seconds",
            "retry_window_seconds",
            "retry_delay_seconds",
            "cleanup_timeout_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} 必须是正数")
            if value <= 0:
                raise ValueError(f"{name} 必须大于0")


class ButtonDebouncer:
    """把原始按压电平转换为稳定且只报告一次的状态变化。"""

    def __init__(self, debounce_seconds: float) -> None:
        """保存消抖窗口并初始化未知状态。"""

        if debounce_seconds <= 0:
            raise ValueError("debounce_seconds 必须大于0")
        self.debounce_seconds = float(debounce_seconds)
        self.stable_pressed: bool | None = None
        self._candidate_pressed: bool | None = None
        self._candidate_since = 0.0

    def update(self, pressed: bool, now: float) -> bool:
        """吸收一次原始采样，稳定状态变化时返回True。"""

        if not isinstance(pressed, bool):
            raise TypeError("pressed 必须是布尔值")
        if self._candidate_pressed is None or pressed != self._candidate_pressed:
            self._candidate_pressed = pressed
            self._candidate_since = now
            return False
        if (
            self.stable_pressed != self._candidate_pressed
            and now - self._candidate_since >= self.debounce_seconds
        ):
            self.stable_pressed = self._candidate_pressed
            return True
        return False
