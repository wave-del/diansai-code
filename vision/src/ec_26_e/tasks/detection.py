"""提供正式任务检测器的兼容资源释放边界。"""

from __future__ import annotations


def close_detector(detector: object) -> None:
    """释放支持 close 的检测器，并兼容没有该方法的旧测试注入。"""

    close = getattr(detector, "close", None)
    if callable(close):
        close()
