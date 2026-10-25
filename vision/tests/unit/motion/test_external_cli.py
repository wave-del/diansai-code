"""验证两个 motion 外部脚本的默认执行、显式 dry-run 和清理。

该文件不依赖串口或 pyserial，不会执行真实运动，可通过
``python -m unittest tests.unit.motion.test_external_cli`` 独立运行。
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from ec_26_e.motion import (
    MotionSerialConfig,
    MotionTimeoutError,
    MotorAddress,
    SerialWriteEvent,
    Servo42CProtocol,
    ServoReply,
)
from tests.external.motion import control_axes, helpers, replay_current_behavior
from tests.external.motion.control_axes import app as control_app
from tests.external.motion.replay_current_behavior import app as replay_app


def reply(address: MotorAddress, status: int) -> ServoReply:
    """创建外部脚本测试使用的合法回包。"""

    frame = bytes((int(address), status, (int(address) + status) & 0xFF))
    return ServoReply(address=address, status=status, frame=frame)


class ReplyFailureTransport:
    """在首条移动命令失败，并记录后续发送和串口关闭动作。"""

    def __init__(self) -> None:
        """创建空发送记录和测试配置。"""

        self._config = MotionSerialConfig(
            device_path=Path("/dev/fake-motion"),
        )
        self.sent_frames: list[bytes] = []
        self.closed = False
        self._move_failed = False

    @property
    def config(self) -> MotionSerialConfig:
        """返回轴控制器运行时协议要求的配置。"""

        return self._config

    def __enter__(self) -> ReplyFailureTransport:
        """返回当前假传输。"""

        return self

    def __exit__(self, *args: object) -> None:
        """记录外部脚本已退出串口上下文。"""

        self.closed = True

    def send_frame(self, frame: bytes) -> None:
        """记录每条发往假串口的报文。"""

        self.sent_frames.append(frame)

    def wait_reply(
        self,
        address: MotorAddress,
        *,
        expected_status: int | None = None,
        timeout_seconds: float | None = None,
    ) -> ServoReply:
        """首次等待移动回包时超时，其余动作返回成功。"""

        if (
            self.sent_frames[-1][1] == Servo42CProtocol.CODE_MOVE
            and not self._move_failed
        ):
            self._move_failed = True
            raise MotionTimeoutError("模拟移动超时")
        return reply(address, expected_status or 1)


class SuccessTransport(ReplyFailureTransport):
    """模拟全部命令均按预期返回成功状态的串口传输。"""

    def wait_reply(
        self,
        address: MotorAddress,
        *,
        expected_status: int | None = None,
        timeout_seconds: float | None = None,
    ) -> ServoReply:
        """返回调用方要求的状态，用于验证默认真实执行路径。"""

        return reply(address, expected_status or 1)


class XYSuccessTransport(SuccessTransport):
    """提供 C 风格无回包 XY 状态机需要的额外传输方法。"""

    def take_reply(self) -> ServoReply | None:
        """当前 C 默认路径不消费回包，因此始终返回 None。"""

        return None

    def discard_replies(self) -> None:
        """模拟清空回包；无回包路径不需要额外动作。"""


class FakeClock:
    """用 sleep 调用直接推进单调时钟，避免单元测试真实等待。"""

    def __init__(self) -> None:
        """从零时刻开始。"""

        self.now = 0.0

    def monotonic(self) -> float:
        """返回当前模拟秒值。"""

        return self.now

    def sleep(self, seconds: float) -> None:
        """直接推进指定秒数。"""

        self.now += seconds


class MotionExternalCliTest(unittest.TestCase):
    """验证帮助、默认执行、全零 no-op 和 dry-run 输出。"""

    def setUp(self) -> None:
        """为每个测试创建隔离的 Typer 调用器。"""

        self.runner = CliRunner()

    def test_help_does_not_require_serial_device(self) -> None:
        """两个脚本查看帮助均应成功且不打开设备。"""

        replay = self.runner.invoke(replay_app, ["--help"])
        control = self.runner.invoke(control_app, ["--help"])

        self.assertEqual(replay.exit_code, 0, replay.output)
        self.assertEqual(control.exit_code, 0, control.output)

    def test_replay_explicit_dry_run_emits_expected_frames(self) -> None:
        """显式 dry-run 应输出 E3 往返报文且不等待输入。"""

        result = self.runner.invoke(replay_app, ["--dry-run"])

        self.assertEqual(result.exit_code, 0, result.output)
        summary = json.loads(result.stdout)
        self.assertTrue(summary["dry_run"])
        frame_hex = [
            execution["commands"][0]["frame_hex"]
            for execution in summary["executions"]
        ]
        self.assertEqual(
            frame_hex,
            [
                "E3 FD 40 00 00 0C 80 AC",
                "E3 FD C0 00 00 0C 80 2C",
            ],
        )
        self.assertEqual(
            [
                reply["frame_hex"]
                for reply in summary["executions"][0]["commands"][0][
                    "expected_replies"
                ]
            ],
            ["E3 01 E4", "E3 02 E5"],
        )

    def test_control_default_move_is_safe_no_op(self) -> None:
        """move 全部默认距离为零时不能打开串口或发送报文。"""

        result = self.runner.invoke(control_app, ["move"])

        self.assertEqual(result.exit_code, 0, result.output)
        summary = json.loads(result.stdout)
        self.assertFalse(summary["dry_run"])
        self.assertTrue(summary["no_op"])
        self.assertFalse(summary["executed"])
        self.assertEqual(summary["executions"], [])

    def test_control_move_and_enable_support_explicit_dry_run(self) -> None:
        """非零移动和 enable 可显式只输出完整计划。"""

        move = self.runner.invoke(
            control_app,
            ["--dry-run", "move", "--x", "1", "--z", "90"],
        )
        enable = self.runner.invoke(
            control_app,
            ["--dry-run", "enable", "--axes", "z"],
        )

        self.assertEqual(move.exit_code, 0, move.output)
        self.assertEqual(enable.exit_code, 0, enable.output)
        move_summary = json.loads(move.stdout)
        enable_summary = json.loads(enable.stdout)
        self.assertTrue(move_summary["dry_run"])
        self.assertEqual(
            move_summary["requested"]["xy_coordinate_mode"],
            "absolute_0p01mm",
        )
        self.assertEqual(
            move_summary["requested"]["xy_target_0p01mm"],
            {"x": 100, "y": 0},
        )
        self.assertTrue(
            move_summary["requested"]["xy_leave_enabled_after_success"]
        )
        self.assertEqual(
            len(enable_summary["executions"][0]["commands"]),
            1,
        )

    def test_control_xy_real_path_matches_c_lifecycle(self) -> None:
        """XY 应等待 500 ms、同批发 E0/E1/E2 并在完成后保持使能。"""

        transport = XYSuccessTransport()
        clock = FakeClock()
        context = control_axes.MotionCliContext(
            config=transport.config,
            execute=True,
        )
        with patch.object(
            control_axes,
            "Servo42CSerial",
            return_value=transport,
        ):
            summary = control_axes.run_move(
                context,
                x_mm=1.0,
                y_mm=2.0,
                sleep=clock.sleep,
                monotonic=clock.monotonic,
            )

        self.assertTrue(summary["executed"])
        self.assertTrue(transport.closed)
        self.assertAlmostEqual(clock.now, 0.525, places=9)
        self.assertEqual(
            [frame[0] for frame in transport.sent_frames],
            [0xE0, 0xE1, 0xE2, 0xE0, 0xE1, 0xE2],
        )
        self.assertEqual(
            [frame[1] for frame in transport.sent_frames],
            [0xF3, 0xF3, 0xF3, 0xFD, 0xFD, 0xFD],
        )
        self.assertEqual(
            [frame[2] for frame in transport.sent_frames[-3:]],
            [0x40, 0xC0, 0xC0],
        )

    def test_replay_cli_defaults_to_real_execution(self) -> None:
        """复刻 CLI 无执行开关时应把 execute=True 传入核心流程。"""

        observed_execute: list[bool] = []

        def fake_run_replay(
            config: MotionSerialConfig,
            **kwargs: object,
        ) -> dict[str, object]:
            """记录 CLI 解析出的执行开关并返回最小摘要。"""

            observed_execute.append(bool(kwargs["execute"]))
            return {"executed": bool(kwargs["execute"])}

        with patch.object(
            replay_current_behavior,
            "run_replay",
            side_effect=fake_run_replay,
        ):
            result = self.runner.invoke(replay_app, [])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(observed_execute, [True])
        self.assertTrue(json.loads(result.stdout)["executed"])

    def test_control_cli_defaults_to_open_and_send(self) -> None:
        """通用 CLI 默认应通过串口发送所选轴的命令。"""

        transport = SuccessTransport()
        with patch.object(
            control_axes,
            "Servo42CSerial",
            return_value=transport,
        ) as serial_constructor:
            result = self.runner.invoke(
                control_app,
                ["enable", "--axes", "z"],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        summary = json.loads(result.stdout)
        self.assertFalse(summary["dry_run"])
        self.assertTrue(summary["executed"])
        self.assertEqual(
            [frame[1] for frame in transport.sent_frames],
            [Servo42CProtocol.CODE_SET_ENABLE],
        )
        self.assertTrue(transport.closed)
        serial_constructor.assert_called_once()
        constructor_args, constructor_kwargs = (
            serial_constructor.call_args
        )
        self.assertIsInstance(constructor_args[0], MotionSerialConfig)
        observer = constructor_kwargs["write_observer"]
        self.assertTrue(callable(observer))
        self.assertIsInstance(
            observer.__self__,
            helpers.SerialWriteEventReporter,
        )

    def test_timestamp_reporter_flushes_all_events(self) -> None:
        """后台输出器退出前应打印全部已发布事件。"""

        event = SerialWriteEvent(
            sequence=1,
            frame=bytes.fromhex("E3 F3 01 D7"),
            wall_time_ns=1_700_000_000_123_456_789,
            started_monotonic_ns=1_000,
            completed_monotonic_ns=1_100,
            delta_from_previous_start_ns=None,
            bytes_written=4,
            write_calls=1,
        )

        with patch.object(
            helpers,
            "emit_serial_write_event",
        ) as emit:
            with helpers.SerialWriteEventReporter() as reporter:
                reporter.publish(event)

        emit.assert_called_once_with(event)

    def test_replay_reply_failure_only_closes_serial(self) -> None:
        """复刻流程回包异常时不得自动停止或失能。"""

        transport = ReplyFailureTransport()
        with patch.object(
            replay_current_behavior,
            "Servo42CSerial",
            return_value=transport,
        ):
            with self.assertRaises(MotionTimeoutError):
                replay_current_behavior.run_replay(
                    transport.config,
                    execute=True,
                    sleep=lambda seconds: None,
                    prompt=lambda message: "",
                )

        self.assertEqual(
            [frame[1] for frame in transport.sent_frames],
            [0xF3, 0xFD],
        )
        self.assertTrue(transport.closed)

    def test_control_reply_failure_only_closes_serial(self) -> None:
        """通用移动回包异常时不得自动停止或失能。"""

        transport = ReplyFailureTransport()
        context = control_axes.MotionCliContext(
            config=transport.config,
            execute=True,
        )
        with patch.object(
            control_axes,
            "Servo42CSerial",
            return_value=transport,
        ):
            with self.assertRaises(MotionTimeoutError):
                control_axes.run_move(context, z_degrees=1.0)

        self.assertEqual(
            [frame[1] for frame in transport.sent_frames],
            [0xF3, 0xFD],
        )
        self.assertTrue(transport.closed)


if __name__ == "__main__":
    unittest.main()
