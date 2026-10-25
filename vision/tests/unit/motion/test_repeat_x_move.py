"""验证重复 X 入口只通过 Python 复刻 XY 状态机控制。

该文件使用内存串口和模拟时钟，不依赖真实设备。可通过
``python -m unittest tests.unit.motion.test_repeat_x_move`` 独立运行。
"""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from ec_26_e.motion import MotionSerialConfig
from tests.external.motion import repeat_x_move


class FakeClock:
    """提供不真实等待的单调时钟。"""

    def __init__(self) -> None:
        """从零秒开始。"""

        self.now = 0.0

    def monotonic(self) -> float:
        """返回当前秒值。"""

        return self.now

    def sleep(self, seconds: float) -> None:
        """直接推进模拟时间。"""

        self.now += seconds


class FakeTransport:
    """记录状态机发出的报文并模拟上下文关闭。"""

    def __init__(self) -> None:
        """创建测试配置和空发送记录。"""

        self.config = MotionSerialConfig(
            device_path=Path("/dev/fake-motion")
        )
        self.frames: list[bytes] = []
        self.closed = False

    def __enter__(self) -> FakeTransport:
        """返回当前传输对象。"""

        return self

    def __exit__(self, *args: object) -> None:
        """记录串口上下文退出。"""

        self.closed = True

    def send_frame(self, frame: bytes) -> None:
        """记录完整报文。"""

        self.frames.append(frame)

    def take_reply(self) -> None:
        """当前 C 默认路径不要求回包。"""

        return None

    def discard_replies(self) -> None:
        """模拟 C 的回包清空。"""


class RepeatXMoveTest(unittest.TestCase):
    """验证目标序列、状态机发报和保持使能终态。"""

    def test_dry_run_lists_absolute_targets_without_serial(self) -> None:
        """dry-run 应只计算 15/30/45/60 mm 目标。"""

        config = MotionSerialConfig(
            device_path=Path("/dev/fake-motion")
        )
        with patch.object(repeat_x_move, "Servo42CSerial") as serial:
            summary = repeat_x_move.run_repeat_x(
                config,
                execute=False,
            )

        serial.assert_not_called()
        self.assertEqual(
            summary["requested"]["targets_mm"],
            [15.0, 30.0, 45.0, 60.0],
        )

    def test_real_path_initializes_once_and_uses_absolute_state(self) -> None:
        """真实路径应使能一次，随后四次从已提交位置递增 15 mm。"""

        transport = FakeTransport()
        clock = FakeClock()
        with patch.object(
            repeat_x_move,
            "Servo42CSerial",
            return_value=transport,
        ):
            summary = repeat_x_move.run_repeat_x(
                transport.config,
                execute=True,
                sleep=clock.sleep,
                monotonic=clock.monotonic,
            )

        self.assertTrue(summary["executed"])
        self.assertTrue(transport.closed)
        self.assertEqual(
            [frame[1] for frame in transport.frames],
            [0xF3, 0xF3, 0xF3] + [0xFD] * 8,
        )
        self.assertNotIn(0x00, [frame[2] for frame in transport.frames[:3]])


if __name__ == "__main__":
    unittest.main()
