"""验证 Servo42CSerial 完整写入、回包保留、异常和资源释放。

该文件使用内存假串口，不依赖 pyserial 或真实设备，可通过
``python -m unittest tests.unit.motion.test_serial_transport`` 独立运行。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import unittest

from ec_26_e.motion import (
    MotionProtocolError,
    MotionRejectedError,
    MotionSerialConfig,
    MotionSerialError,
    MotionTimeoutError,
    MotorAddress,
    SerialWriteEvent,
    Servo42CProtocol,
    Servo42CSerial,
)


def reply_frame(address: MotorAddress, status: int) -> bytes:
    """构造合法 Servo42C 回包。"""

    return bytes((int(address), status, (int(address) + status) & 0xFF))


class FakeSerial:
    """模拟可分块写入、可注入接收数据的同步串口。"""

    def __init__(
        self,
        *,
        receive_data: bytes = b"",
        max_write_size: int | None = None,
        on_write: Callable[[bytes], bytes] | None = None,
    ) -> None:
        """创建打开状态的内存串口。"""

        self._receive = bytearray(receive_data)
        self.max_write_size = max_write_size
        self.on_write = on_write
        self.written = bytearray()
        self.write_calls = 0
        self.close_calls = 0

    @property
    def in_waiting(self) -> int:
        """返回当前接收缓冲区大小。"""

        return len(self._receive)

    def write(self, data: bytes) -> int:
        """写入配置允许的部分字节，并可生成同步测试回包。"""

        self.write_calls += 1
        count = (
            len(data)
            if self.max_write_size is None
            else min(len(data), self.max_write_size)
        )
        chunk = data[:count]
        self.written.extend(chunk)
        if self.on_write is not None:
            self._receive.extend(self.on_write(chunk))
        return count

    def read(self, size: int = 1) -> bytes:
        """从内存接收缓冲区取出最多指定字节。"""

        count = min(size, len(self._receive))
        result = bytes(self._receive[:count])
        del self._receive[:count]
        return result

    def close(self) -> None:
        """记录一次幂等关闭调用。"""

        self.close_calls += 1


def test_config() -> MotionSerialConfig:
    """创建等待时间很短的测试串口配置。"""

    return MotionSerialConfig(
        device_path=Path("/dev/fake-motion"),
        read_timeout_seconds=0.001,
        write_timeout_seconds=0.01,
        reply_timeout_seconds=0.01,
    )


class MotionSerialConfigTest(unittest.TestCase):
    """验证串口默认值和访问设备前的严格校验。"""

    def test_defaults_match_mks_uart(self) -> None:
        """默认配置应使用 ttyAS4 和 115200 波特率。"""

        config = MotionSerialConfig()

        self.assertEqual(config.device_path, Path("/dev/ttyAS4"))
        self.assertEqual(config.baudrate, 115200)

    def test_invalid_values_are_rejected(self) -> None:
        """错误路径类型、布尔波特率和非正超时应立即失败。"""

        with self.assertRaisesRegex(TypeError, "device_path"):
            MotionSerialConfig(device_path="/dev/ttyAS4")  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "baudrate"):
            MotionSerialConfig(baudrate=True)
        with self.assertRaisesRegex(ValueError, "read_timeout"):
            MotionSerialConfig(read_timeout_seconds=0)


class Servo42CSerialTest(unittest.TestCase):
    """验证同步传输的正常路径、队列语义和失败路径。"""

    def test_construction_is_side_effect_free_and_close_is_idempotent(
        self,
    ) -> None:
        """构造不调用工厂，重复关闭也不重复关设备。"""

        fake = FakeSerial()
        factory_calls = 0

        def factory(config: MotionSerialConfig) -> FakeSerial:
            """返回固定假串口并记录调用次数。"""

            nonlocal factory_calls
            factory_calls += 1
            return fake

        transport = Servo42CSerial(test_config(), serial_factory=factory)
        self.assertEqual(factory_calls, 0)

        with transport:
            self.assertTrue(transport.is_open)
        transport.close()

        self.assertEqual(factory_calls, 1)
        self.assertEqual(fake.close_calls, 1)

    def test_partial_writes_are_retried_until_frame_is_complete(self) -> None:
        """底层每次只写两字节时也必须完整发送整帧。"""

        fake = FakeSerial(max_write_size=2)
        frame = Servo42CProtocol.build_move(MotorAddress.Z, -64, 3200)

        with Servo42CSerial(
            test_config(),
            serial_factory=lambda config: fake,
        ) as transport:
            transport.send_frame(frame)

        self.assertEqual(bytes(fake.written), frame)
        self.assertGreater(fake.write_calls, 1)

    def test_write_observer_receives_nanosecond_timestamps_and_delta(
        self,
    ) -> None:
        """观察器应收到每帧开始、完成及相邻开始时间差。"""

        fake = FakeSerial()
        events: list[SerialWriteEvent] = []
        wall_values = iter(
            (
                1_700_000_000_123_456_789,
                1_700_000_000_124_456_789,
            )
        )
        monotonic_values = iter(
            (
                1_000_000,
                1_030_000,
                1_800_000,
                1_850_000,
            )
        )
        first = Servo42CProtocol.build_enable(MotorAddress.X0, True)
        second = Servo42CProtocol.build_enable(MotorAddress.X1, True)

        with Servo42CSerial(
            test_config(),
            serial_factory=lambda config: fake,
            write_observer=events.append,
            wall_time_ns=lambda: next(wall_values),
            monotonic_time_ns=lambda: next(monotonic_values),
        ) as transport:
            transport.send_frame(first)
            transport.send_frame(second)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].sequence, 1)
        self.assertIsNone(events[0].delta_from_previous_start_ns)
        self.assertEqual(events[0].write_duration_ns, 30_000)
        self.assertEqual(events[1].sequence, 2)
        self.assertEqual(events[1].delta_from_previous_start_ns, 800_000)
        self.assertEqual(events[1].write_duration_ns, 50_000)
        self.assertTrue(events[1].success)
        self.assertEqual(events[1].frame, second)
        summary = events[1].to_dict()
        self.assertEqual(
            summary["delta_from_previous_start_us"],
            800.0,
        )
        self.assertEqual(summary["address"], "X1")
        self.assertEqual(summary["function_hex"], "0xF3")

    def test_context_closes_port_when_body_raises(self) -> None:
        """上下文主体异常时也必须释放已经打开的串口。"""

        fake = FakeSerial()
        transport = Servo42CSerial(
            test_config(),
            serial_factory=lambda config: fake,
        )

        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with transport:
                raise RuntimeError("test failure")

        self.assertFalse(transport.is_open)
        self.assertEqual(fake.close_calls, 1)

    def test_wait_for_address_preserves_other_replies(self) -> None:
        """等待 E0 时先到达的 E1 回包应留给下一次任意读取。"""

        fake = FakeSerial(
            receive_data=(
                reply_frame(MotorAddress.X1, 1)
                + reply_frame(MotorAddress.X0, 1)
            )
        )

        with Servo42CSerial(
            test_config(),
            serial_factory=lambda config: fake,
        ) as transport:
            e0 = transport.wait_reply(
                MotorAddress.X0,
                expected_status=1,
            )
            next_reply = transport.read_reply()

        self.assertEqual(e0.address, MotorAddress.X0)
        self.assertEqual(next_reply.address, MotorAddress.X1)

    def test_take_and_discard_replies_match_c_nonblocking_queue(self) -> None:
        """非阻塞取回包后，清空操作必须移除队列和底层剩余字节。"""

        fake = FakeSerial(
            receive_data=(
                reply_frame(MotorAddress.X0, 1)
                + reply_frame(MotorAddress.X1, 1)
                + reply_frame(MotorAddress.Y, 1)
            )
        )

        with Servo42CSerial(
            test_config(),
            serial_factory=lambda config: fake,
        ) as transport:
            first = transport.take_reply()
            self.assertIsNotNone(first)
            self.assertEqual(first.address, MotorAddress.X0)  # type: ignore[union-attr]
            transport.discard_replies()
            self.assertIsNone(transport.take_reply())
            self.assertEqual(fake.in_waiting, 0)

    def test_enable_waits_for_success_reply(self) -> None:
        """使能命令应完整发送并返回状态 1。"""

        expected = Servo42CProtocol.build_enable(MotorAddress.Z, True)
        fake = FakeSerial(
            on_write=lambda data: (
                reply_frame(MotorAddress.Z, 1)
                if data == expected
                else b""
            )
        )

        with Servo42CSerial(
            test_config(),
            serial_factory=lambda config: fake,
        ) as transport:
            reply = transport.set_enable(MotorAddress.Z, True)

        self.assertEqual(bytes(fake.written), expected)
        self.assertEqual(reply.status, 1)

    def test_move_waits_for_started_then_completed(self) -> None:
        """E3 移动应发送完整报文并依次消费状态 1 和 2。"""

        expected = Servo42CProtocol.build_move(
            MotorAddress.Z,
            -64,
            3200,
        )
        fake = FakeSerial(
            receive_data=(
                reply_frame(MotorAddress.Z, 1)
                + reply_frame(MotorAddress.Z, 2)
            )
        )

        with Servo42CSerial(
            test_config(),
            serial_factory=lambda config: fake,
        ) as transport:
            replies = transport.move(
                MotorAddress.Z,
                -64,
                3200,
                completion_timeout_seconds=0.7,
            )

        self.assertEqual(bytes(fake.written), expected)
        self.assertEqual(
            [reply.status for reply in replies],
            [1, 2],
        )

    def test_rejected_unexpected_and_timeout_replies_raise(self) -> None:
        """状态 0、错误状态和无回包应映射为不同异常。"""

        cases = (
            (
                reply_frame(MotorAddress.Z, 0),
                MotionRejectedError,
            ),
            (
                reply_frame(MotorAddress.Z, 2),
                MotionProtocolError,
            ),
            (
                b"",
                MotionTimeoutError,
            ),
        )
        for receive_data, error_type in cases:
            with self.subTest(error_type=error_type.__name__):
                fake = FakeSerial(receive_data=receive_data)
                with Servo42CSerial(
                    test_config(),
                    serial_factory=lambda config, fake=fake: fake,
                ) as transport:
                    with self.assertRaises(error_type):
                        transport.wait_reply(
                            MotorAddress.Z,
                            expected_status=1,
                        )

    def test_send_before_open_is_rejected(self) -> None:
        """未显式打开串口时发送命令应失败。"""

        transport = Servo42CSerial(
            test_config(),
            serial_factory=lambda config: FakeSerial(),
        )

        with self.assertRaisesRegex(MotionSerialError, "尚未打开"):
            transport.send_frame(b"\x01")


if __name__ == "__main__":
    unittest.main()
