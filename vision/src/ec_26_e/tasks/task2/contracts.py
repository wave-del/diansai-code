"""定义 Task2 随机拼图复用的任务运行选项和失败契约。"""

from __future__ import annotations

from dataclasses import dataclass

from ec_26_e.control import PlanningOptions
from ec_26_e.tasks.task1.contracts import (
    Task1Error,
    Task1Options,
)


@dataclass(frozen=True, slots=True)
class Task2Options(Task1Options):
    """约束 Task2 使用随机拼图模式的共享视觉采集选项。"""

    planning_options: PlanningOptions = PlanningOptions(mode="random")

    def __post_init__(self) -> None:
        """复用公共校验并拒绝非随机拼图模式。"""

        Task1Options.__post_init__(self)
        if self.planning_options.mode != "random":
            raise ValueError("Task2 仅支持 random 拼图模式")


class Task2Error(Task1Error):
    """表示 Task2 编排或依赖契约无法继续执行。"""
