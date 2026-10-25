"""定义所有注册任务共享的运行上下文和用户取消语义。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TaskContext:
    """描述由项目环境统一控制的任务级可选运行能力。"""

    draw_enabled: bool = False

    def __post_init__(self) -> None:
        """拒绝非布尔绘制开关。"""

        if not isinstance(self.draw_enabled, bool):
            raise TypeError("draw_enabled 必须是布尔值")


class TaskCancelledError(RuntimeError):
    """表示用户通过统一交互界面主动中止当前任务。"""
