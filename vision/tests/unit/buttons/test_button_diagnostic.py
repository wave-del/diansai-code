"""验证实体按键诊断脚本的映射、消抖打印和资源释放。

该文件使用假GPIO，不依赖A7Z硬件或systemd，可由默认单元测试运行。
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from ec_26_e.buttons import create_a7z_button_bindings
from tests.external.gpio.test_buttons import monitor_button_presses


class SequenceButtonSource:
    """按顺序提供按键快照，并记录打开和关闭状态。"""

    def __init__(self, states: list[Mapping[str, bool]]) -> None:
        """保存有限的按键状态序列。"""

        self.states = iter(states)
        self.opened = False
        self.closed = False

    def open(self) -> None:
        """记录GPIO已打开。"""

        self.opened = True

    def read_pressed(self) -> Mapping[str, bool]:
        """返回下一份按键快照，耗尽后模拟用户退出。"""

        try:
            return next(self.states)
        except StopIteration as error:
            raise KeyboardInterrupt from error

    def close(self) -> None:
        """记录GPIO已释放。"""

        self.closed = True


def test_prints_only_pressed_physical_pin_and_closes_source() -> None:
    """验证初始松开不打印编号，稳定按下35脚只打印35。"""

    source = SequenceButtonSource(
        [
            {"task1": False, "task2": False},
            {"task1": False, "task2": False},
            {"task1": True, "task2": False},
            {"task1": True, "task2": False},
        ]
    )
    emitted: list[str] = []
    timestamps = iter((0.00, 0.00, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03))

    with pytest.raises(KeyboardInterrupt):
        monitor_button_presses(
            create_a7z_button_bindings(),
            debounce_seconds=0.005,
            poll_interval_seconds=0.001,
            source=source,  # type: ignore[arg-type]
            emit=emitted.append,
            monotonic=lambda: next(timestamps),
            sleep=lambda _seconds: None,
        )

    assert source.opened
    assert source.closed
    assert emitted[-1] == "35"
    assert "37" not in emitted
