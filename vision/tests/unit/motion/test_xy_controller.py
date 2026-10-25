"""验证 Python XY 状态机逐项遵循 MKS ``xy_axes.c``。

覆盖绝对坐标、上电使能、并行发报、无回包计时完成、复位、紧急停止和
可选回包分支。该文件不依赖真实串口，可通过
``python -m unittest tests.unit.motion.test_xy_controller`` 独立运行。
"""

from __future__ import annotations

from collections import deque
import unittest

from ec_26_e.motion import (
    MksXYController,
    MotorAddress,
    Servo42CProtocol,
    ServoReply,
    XYFault,
    XYResult,
    XYState,
    mm_to_0p01mm,
)


class FakeClock:
    """提供可精确推进的单调时钟。"""

    def __init__(self) -> None:
        """从零秒开始。"""

        self.now = 0.0

    def __call__(self) -> float:
        """返回当前秒值。"""

        return self.now

    def advance_ms(self, milliseconds: int) -> None:
        """按整数毫秒推进时间。"""

        self.now += milliseconds / 1000.0


class FakeXYTransport:
    """记录发送帧并提供 C 风格非阻塞回包队列。"""

    def __init__(self) -> None:
        """创建空报文、回包和清理计数。"""

        self.sent_frames: list[bytes] = []
        self.replies: deque[ServoReply] = deque()
        self.discard_count = 0

    def send_frame(self, frame: bytes) -> None:
        """记录一条完整报文。"""

        self.sent_frames.append(frame)

    def take_reply(self) -> ServoReply | None:
        """返回当前最早回包，没有时立即返回 None。"""

        return self.replies.popleft() if self.replies else None

    def discard_replies(self) -> None:
        """清空回包并记录一次 C 风格清理。"""

        self.replies.clear()
        self.discard_count += 1

    def feed_reply(self, address: MotorAddress, status: int) -> None:
        """追加一条校验和正确的结构化回包。"""

        frame = bytes(
            (int(address), status, (int(address) + status) & 0xFF)
        )
        self.replies.append(
            ServoReply(address=address, status=status, frame=frame)
        )


def ready_controller(
    *,
    require_motor_reply: bool = False,
) -> tuple[MksXYController, FakeXYTransport, FakeClock]:
    """创建控制器并推进到 C 的 READY 状态。"""

    clock = FakeClock()
    transport = FakeXYTransport()
    controller = MksXYController(
        transport,
        require_motor_reply=require_motor_reply,
        monotonic=clock,
    )
    controller.initialize()
    clock.advance_ms(500)
    controller.process()
    if not require_motor_reply:
        if controller.state != XYState.READY:
            raise AssertionError("无回包控制器未进入 READY")
    return controller, transport, clock


class MksXYNoReplyTest(unittest.TestCase):
    """验证当前 C 宏为 0 时实际执行的 XY 路径。"""

    def test_power_wait_enables_all_xy_without_waiting_replies(self) -> None:
        """500 ms 到期前不发送，到期后按 E0/E1/E2 顺序使能。"""

        clock = FakeClock()
        transport = FakeXYTransport()
        controller = MksXYController(transport, monotonic=clock)
        controller.initialize()

        clock.advance_ms(499)
        controller.process()
        self.assertEqual(transport.sent_frames, [])
        self.assertEqual(controller.state, XYState.WAIT_POWER)

        clock.advance_ms(1)
        controller.process()
        self.assertEqual(
            transport.sent_frames,
            [
                Servo42CProtocol.build_enable(MotorAddress.X0, True),
                Servo42CProtocol.build_enable(MotorAddress.X1, True),
                Servo42CProtocol.build_enable(MotorAddress.Y, True),
            ],
        )
        self.assertEqual(controller.state, XYState.READY)
        self.assertTrue(controller.is_at_origin())

    def test_absolute_xy_sends_x_pair_and_y_back_to_back(self) -> None:
        """绝对目标应换算增量，并按 E0/E1/E2 同批发送。"""

        controller, transport, clock = ready_controller()
        transport.sent_frames.clear()

        result = controller.move_to(100, 200)

        self.assertEqual(result, XYResult.OK)
        self.assertEqual(controller.state, XYState.WAIT_MOVE_COMPLETE)
        self.assertEqual(controller.move_timeout_ms, 550)
        self.assertEqual(
            transport.sent_frames,
            [
                Servo42CProtocol.build_move(MotorAddress.X0, 64, 400),
                Servo42CProtocol.build_move(MotorAddress.X1, -64, 400),
                Servo42CProtocol.build_move(MotorAddress.Y, -64, 800),
            ],
        )

        clock.advance_ms(24)
        controller.process()
        self.assertEqual(controller.state, XYState.WAIT_MOVE_COMPLETE)
        clock.advance_ms(1)
        controller.process()
        self.assertEqual(controller.state, XYState.READY)
        self.assertEqual(controller.x_position_0p01mm, 100)
        self.assertEqual(controller.y_position_0p01mm, 200)

        transport.sent_frames.clear()
        self.assertEqual(controller.move_to(200, 200), XYResult.OK)
        self.assertEqual(
            transport.sent_frames,
            [
                Servo42CProtocol.build_move(MotorAddress.X0, 64, 400),
                Servo42CProtocol.build_move(MotorAddress.X1, -64, 400),
            ],
        )

    def test_expected_time_and_fault_timeout_are_separate(self) -> None:
        """无回包按 80 mm/s 完成，有回包故障超时保留安全余量。"""

        pulses_for_100_mm = 40_000

        self.assertEqual(
            MksXYController.calculate_expected_move_ms(
                pulses_for_100_mm,
                0,
            ),
            1250,
        )
        self.assertEqual(
            MksXYController.calculate_move_timeout_ms(
                pulses_for_100_mm,
                0,
            ),
            3000,
        )

    def test_result_codes_match_not_ready_busy_and_range_rules(self) -> None:
        """调用结果应保持 C 的状态优先级和绝对工作区边界。"""

        clock = FakeClock()
        transport = FakeXYTransport()
        controller = MksXYController(transport, monotonic=clock)
        controller.initialize()
        self.assertEqual(controller.move_to(0, 0), XYResult.NOT_READY)

        clock.advance_ms(500)
        controller.process()
        self.assertEqual(controller.move_to(23_001, 0), XYResult.OUT_OF_RANGE)
        self.assertEqual(controller.move_to(0, 36_001), XYResult.OUT_OF_RANGE)
        self.assertEqual(controller.move_to(0, -1_001), XYResult.OUT_OF_RANGE)
        self.assertEqual(controller.move_to(0, -350), XYResult.OK)
        self.assertEqual(controller.y_target_0p01mm, -350)
        self.assertEqual(controller.move_to(100, 0), XYResult.BUSY)

    def test_reset_request_queues_during_move_and_returns_to_origin(self) -> None:
        """运动期间的复位请求应排队，并在当前移动完成后启动回原点。"""

        controller, transport, clock = ready_controller()
        self.assertEqual(controller.move_to(100, 200), XYResult.OK)
        self.assertEqual(controller.reset_to_origin(), XYResult.OK)
        self.assertTrue(controller.reset_requested)

        clock.advance_ms(controller.move_timeout_ms)
        controller.process()

        self.assertFalse(controller.reset_requested)
        self.assertTrue(controller.reset_in_progress)
        self.assertEqual(controller.state, XYState.WAIT_MOVE_COMPLETE)
        self.assertEqual(controller.x_target_0p01mm, 0)
        self.assertEqual(controller.y_target_0p01mm, 0)
        self.assertEqual(
            [frame[0] for frame in transport.sent_frames[-3:]],
            [0xE0, 0xE1, 0xE2],
        )

        clock.advance_ms(controller.move_timeout_ms)
        controller.process()
        self.assertTrue(controller.reset_completed)
        self.assertTrue(controller.is_at_origin())

    def test_emergency_stop_stops_disables_and_invalidates_position(self) -> None:
        """紧急停止应按 C 顺序处理全部 XY 电机并锁存故障。"""

        controller, transport, _ = ready_controller()
        transport.sent_frames.clear()

        controller.emergency_stop()

        self.assertEqual(controller.state, XYState.FAULT)
        self.assertEqual(controller.fault, XYFault.EMERGENCY_STOP)
        self.assertFalse(controller.position_valid)
        self.assertEqual(
            transport.sent_frames,
            [
                Servo42CProtocol.build_stop(MotorAddress.X0),
                Servo42CProtocol.build_stop(MotorAddress.X1),
                Servo42CProtocol.build_enable(MotorAddress.X0, False),
                Servo42CProtocol.build_enable(MotorAddress.X1, False),
                Servo42CProtocol.build_stop(MotorAddress.Y),
                Servo42CProtocol.build_enable(MotorAddress.Y, False),
            ],
        )
        self.assertEqual(controller.move_to(0, 0), XYResult.POSITION_INVALID)

    def test_assume_origin_clears_fault_and_restarts_power_wait(self) -> None:
        """假定原点应停止失能、清故障并重新等待 500 ms 上电。"""

        controller, transport, clock = ready_controller()
        controller.emergency_stop()
        transport.sent_frames.clear()

        controller.assume_origin()

        self.assertEqual(controller.state, XYState.WAIT_POWER)
        self.assertEqual(controller.fault, XYFault.NONE)
        self.assertTrue(controller.position_valid)
        self.assertTrue(controller.reset_completed)
        self.assertEqual(
            [frame[1] for frame in transport.sent_frames],
            [0xF7, 0xF7, 0xF3, 0xF3, 0xF7, 0xF3],
        )
        clock.advance_ms(500)
        controller.process()
        self.assertTrue(controller.is_at_origin())

    def test_mm_conversion_requires_c_public_api_precision(self) -> None:
        """毫米适配只接受可无损表示为 C 端 0.01 mm 单位的数值。"""

        self.assertEqual(mm_to_0p01mm("x", 1.25), 125)
        with self.assertRaisesRegex(ValueError, "0.01 mm"):
            mm_to_0p01mm("x", 1.001)


class MksXYReplyModeTest(unittest.TestCase):
    """验证 C 中保留的 ``XY_REQUIRE_MOTOR_REPLY=1`` 分支。"""

    def test_enable_sequence_and_direct_complete_reply_match_c(self) -> None:
        """使能逐地址确认，移动状态 02 可同时表示开始和完成。"""

        controller, transport, _ = ready_controller(
            require_motor_reply=True
        )
        self.assertEqual(controller.state, XYState.WAIT_ENABLE_X0)

        transport.feed_reply(MotorAddress.X0, 1)
        controller.process()
        self.assertEqual(controller.state, XYState.WAIT_ENABLE_X1)
        transport.feed_reply(MotorAddress.X1, 1)
        controller.process()
        self.assertEqual(controller.state, XYState.WAIT_ENABLE_Y)
        transport.feed_reply(MotorAddress.Y, 1)
        controller.process()
        self.assertEqual(controller.state, XYState.READY)

        self.assertEqual(controller.move_to(100, 0), XYResult.OK)
        transport.feed_reply(MotorAddress.X0, 2)
        transport.feed_reply(MotorAddress.X1, 2)
        controller.process()
        self.assertEqual(controller.state, XYState.READY)
        self.assertEqual(controller.x_position_0p01mm, 100)


if __name__ == "__main__":
    unittest.main()
