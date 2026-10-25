"""提供跨按钮、桌面预览和手工任务共用的非阻塞执行锁。"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path


DEFAULT_EXECUTION_RUNTIME_DIRECTORY = Path(
    os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
) / "ec26e"
DEFAULT_EXECUTION_LOCK_PATH = DEFAULT_EXECUTION_RUNTIME_DIRECTORY / "task.lock"


class TaskBusyError(RuntimeError):
    """表示相机或运动硬件已被另一个EC26E入口占用。"""


class TaskExecutionLock:
    """持有进程级flock，并在退出上下文时幂等释放。"""

    def __init__(self, path: Path = DEFAULT_EXECUTION_LOCK_PATH) -> None:
        """保存锁文件路径，构造阶段不访问文件系统。"""

        if not isinstance(path, Path):
            raise TypeError("path 必须是 pathlib.Path")
        self.path = path
        self._stream: object | None = None

    def acquire(self) -> None:
        """非阻塞申请全局锁，忙碌时返回明确异常。"""

        if self._stream is not None:
            raise RuntimeError("任务执行锁已经持有")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            stream = self.path.open("a+", encoding="utf-8")
        except OSError as error:
            raise RuntimeError(f"无法打开任务执行锁 {self.path}：{error}") from error
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            stream.seek(0)
            stream.truncate()
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
        except BlockingIOError as error:
            stream.close()
            raise TaskBusyError("已有任务或视觉预览正在运行") from error
        except BaseException:
            stream.close()
            raise
        self._stream = stream

    def release(self) -> None:
        """释放锁并关闭文件；重复调用安全。"""

        stream = self._stream
        self._stream = None
        if stream is None:
            return
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()

    def __enter__(self) -> TaskExecutionLock:
        """申请锁并返回自身。"""

        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """离开上下文时释放锁。"""

        self.release()
