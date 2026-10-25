"""以 Task2 名称复用相机、视觉稳定审查和随机控制规划流程。"""

from __future__ import annotations

from ec_26_e.cam import CameraSource
from ec_26_e.control import MovePlan, plan_sheet_moves
from ec_26_e.tasks.task1.contracts import (
    Detect1Processor,
    PuzzlePlanner,
    ResizeProcessor,
    Task1FrameObserver,
)
from ec_26_e.tasks.task1.runner import run_puzzle_task
from ec_26_e.tasks.task2.contracts import Task2Options


def run_task2(
    camera: CameraSource,
    resizer: ResizeProcessor,
    detector: Detect1Processor,
    *,
    options: Task2Options = Task2Options(),
    planner: PuzzlePlanner = plan_sheet_moves,
    observer: Task1FrameObserver | None = None,
) -> MovePlan:
    """运行一次 Task2，并在释放相机后返回随机拼图移动方案。"""

    if not isinstance(options, Task2Options):
        raise TypeError("options 必须是 Task2Options")
    return run_puzzle_task(
        camera,
        resizer,
        detector,
        options=options,
        planner=planner,
        observer=observer,
        task_label="Task2",
    )
