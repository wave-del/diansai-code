"""公开 Task2 随机拼图任务的配置选项与纯内存入口。"""

from ec_26_e.tasks.task2.contracts import (
    Task2Error,
    Task2Options,
)
from ec_26_e.tasks.task2.runner import run_task2

__all__ = [
    "Task2Error",
    "Task2Options",
    "run_task2",
]
