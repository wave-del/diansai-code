"""公开 Task1 的强类型配置、可替换依赖协议和纯内存入口。"""

from ec_26_e.tasks.task1.contracts import (
    Detect1Processor,
    PlanActuator,
    PuzzlePlanner,
    ResizeProcessor,
    Task1FrameObserver,
    Task1Error,
    Task1Options,
    Task1TimeoutError,
)
from ec_26_e.tasks.task1.runner import run_task1

__all__ = [
    "Detect1Processor",
    "PlanActuator",
    "PuzzlePlanner",
    "ResizeProcessor",
    "Task1FrameObserver",
    "Task1Error",
    "Task1Options",
    "Task1TimeoutError",
    "run_task1",
]
