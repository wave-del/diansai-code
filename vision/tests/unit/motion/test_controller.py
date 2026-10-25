"""验证 XYZ 方向换算、顺序执行、软边界和完成回包语义。

该文件使用内存传输替身，不依赖真实串口，可通过
``python -m unittest tests.unit.motion.test_controller`` 独立运行。
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import unittest

from ec_26_e.motion import (
    MotionSerialConfig,
    MksAxisController,
    MotorAddress,
    Servo42CProtocol,
    ServoReply,
)


def reply(address: MotorAddress, status: int) -> ServoReply:
    """创建控制器测试使用的合法结构化回包。"""

    frame = bytes((int(address), status, (int(address) + status) & 0xFF))
    return ServoReply(address=address, status=status, frame=frame)


class FakeTransport:
    """按发送命令自动准备状态 1/2 回包的同步传输替身。"""

    def __init__(self) -> None:
        """初始化空发送记录和分地址回包队列。"""

        self._config = MotionSerialConfig(
            device_path=Path("/dev/fake-motion"),
        )
        self.sent_frames: list[bytes] = []
        self.wait_calls: list[tuple[MotorAddress, int | None]] = []
        self._replies: dict[MotorAddress, deque[ServoReply]] = defaultdict(deque)

    @property
    def config(self) -> MotionSerialConfig:
        """返回控制器所需的回包超时配置。"""

        return self._config

    def send_frame(self, frame: bytes) -> None:
        """记录报文并根据功能码准备成功回包。"""

        self.sent_frames.append(frame)
        address = MotorAddress(frame[0])
        self._replies[address].append(reply(address, 1))
        if frame[1] == Servo42CProtocol.CODE_MOVE:
            self._replies[address].append(reply(address, 2))

    def wait_reply(
        self,
        address: MotorAddress,
        *,
        expected_status: int | None = None,
        timeout_seconds: float | None = None,
    ) -> ServoReply:
        """返回目标地址最早回包并核对控制器要求的状态。"""

        self.wait_calls.append((address, expected_status))
        result = self._replies[address].popleft()
        if expected_status is not None:
            if result.status != expected_status:
                raise AssertionError("控制器等待状态与测试回包不一致")
        return result


class AxisPlanningTest(unittest.TestCase):
    """验证纯命令规划的单位、方向、地址和边界。"""

    def setUp(self) -> None:
        """使每个测试都从尚未使用右移补偿的进程状态开始。"""

        MksAxisController._right_compensation_pending = True

    def test_positive_xyz_matches_calibrated_mechanical_directions(self) -> None:
        """正 X/Y/Z 应分别映射为右、下、上的机构方向。"""

        commands = MksAxisController.plan_relative(
            x_mm=1.5,
            y_mm=2.0,
            z_degrees=90.0,
            speed=64,
        )

        self.assertEqual(
            [command.address for command in commands],
            [
                MotorAddress.X0,
                MotorAddress.X1,
                MotorAddress.Y,
                MotorAddress.Z,
            ],
        )
        self.assertEqual(
            [command.pulses for command in commands],
            [600, 600, 800, 3200],
        )
        self.assertEqual(commands[0].frame[2], 0x40)
        self.assertEqual(commands[1].frame[2], 0xC0)
        self.assertEqual(commands[2].frame[2], 0xC0)
        self.assertEqual(commands[3].frame[2], 0x40)
        later_commands = MksAxisController.plan_relative(x_mm=1.5)
        self.assertEqual(
            [command.pulses for command in later_commands],
            [600, 600],
        )

    def test_negative_directions_are_reversed(self) -> None:
        """负位移应反转每个逻辑轴的协议方向。"""

        commands = MksAxisController.plan_relative(
            x_mm=-1.0,
            y_mm=-1.0,
            z_degrees=-1.0,
        )

        self.assertEqual(commands[0].frame[2], 0xC0)
        self.assertEqual(commands[1].frame[2], 0x40)
        self.assertEqual(commands[2].frame[2], 0x40)
        self.assertEqual(commands[3].frame[2], 0xC0)
        self.assertEqual([commands[0].pulses, commands[1].pulses], [400, 400])

    def test_zero_move_is_no_op_and_boundaries_are_rejected(self) -> None:
        """全零应不产生报文，超出各轴安全范围应立即失败。"""

        self.assertEqual(MksAxisController.plan_relative(), ())
        invalid = (
            {"x_mm": 230.001},
            {"y_mm": -360.001},
            {"z_degrees": 90.001},
            {"speed": 0},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    MksAxisController.plan_relative(**values)

    def test_half_pulse_uses_round_half_up(self) -> None:
        """恰好半个脉冲应按通常四舍五入转换为一个脉冲。"""

        commands = MksAxisController.plan_relative(x_mm=-0.00125)

        self.assertEqual(
            [command.pulses for command in commands],
            [1, 1],
        )

    def test_power_axis_x_expands_to_two_addresses(self) -> None:
        """使能 xz 应按 X0、X1、Z 顺序生成报文。"""

        commands = MksAxisController.plan_power("zx", enabled=True)

        self.assertEqual(
            [command.address for command in commands],
            [MotorAddress.X0, MotorAddress.X1, MotorAddress.Z],
        )


class AxisExecutionTest(unittest.TestCase):
    """验证 X→Y→Z 调度和开始、完成回包等待。"""

    def test_runtime_relative_move_rejects_xy_and_keeps_z(self) -> None:
        """真实 XY 必须进入复刻状态机，旧同步入口只允许独立 Z。"""

        transport = FakeTransport()
        controller = MksAxisController(transport)

        with self.assertRaisesRegex(ValueError, "MksXYController"):
            controller.move_relative(x_mm=1.0)
        result = controller.move_relative(z_degrees=1.0)

        self.assertEqual(
            [frame[0] for frame in transport.sent_frames],
            [0xE3],
        )
        self.assertEqual(
            transport.wait_calls,
            [
                (MotorAddress.Z, 1),
                (MotorAddress.Z, 2),
            ],
        )
        self.assertEqual(len(result.commands), 1)
        self.assertEqual(len(result.replies), 2)

    def test_enable_disable_and_stop_wait_for_each_address(self) -> None:
        """普通轴操作应逐地址发送并要求状态 1。"""

        transport = FakeTransport()
        controller = MksAxisController(transport)

        enabled = controller.enable_axes("y")
        disabled = controller.disable_axes("y")
        stopped = controller.stop_axes("y")

        self.assertEqual(len(enabled.replies), 1)
        self.assertEqual(len(disabled.replies), 1)
        self.assertEqual(len(stopped.replies), 1)
        self.assertEqual(
            [frame[1] for frame in transport.sent_frames],
            [0xF3, 0xF3, 0xF7],
        )

    def test_3200_pulse_timeout_matches_firmware(self) -> None:
        """速度 64、3200 脉冲应得到固件使用的 700 ms 上限。"""

        self.assertAlmostEqual(
            MksAxisController.completion_timeout_seconds(3200, 64),
            0.7,
        )


if __name__ == "__main__":
    unittest.main()
