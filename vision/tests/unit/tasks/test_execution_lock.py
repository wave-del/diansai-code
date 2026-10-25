"""验证所有相机和运动入口共用的进程级非阻塞锁。"""

from __future__ import annotations

from pathlib import Path

import pytest

from ec_26_e.tasks.execution_lock import TaskBusyError, TaskExecutionLock


def test_second_owner_is_rejected_until_first_releases(tmp_path: Path) -> None:
    """验证忙时立即拒绝，原持有者释放后可以重新申请。"""

    lock_path = tmp_path / "runtime" / "task.lock"
    first = TaskExecutionLock(lock_path)
    second = TaskExecutionLock(lock_path)

    first.acquire()
    with pytest.raises(TaskBusyError, match="正在运行"):
        second.acquire()
    first.release()

    second.acquire()
    second.release()


def test_context_manager_releases_after_exception(tmp_path: Path) -> None:
    """验证任务异常离开上下文后不会留下永久忙碌状态。"""

    lock_path = tmp_path / "task.lock"
    with pytest.raises(RuntimeError, match="模拟任务失败"):
        with TaskExecutionLock(lock_path):
            raise RuntimeError("模拟任务失败")

    with TaskExecutionLock(lock_path):
        assert lock_path.read_text(encoding="utf-8").startswith("pid=")
