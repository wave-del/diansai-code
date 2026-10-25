"""验证 Servo42C 命令编码、参数边界和增量回包重同步。

该文件不依赖串口设备，可通过
``python -m unittest tests.unit.motion.test_protocol`` 独立运行。
"""

from __future__ import annotations

import unittest

from ec_26_e.motion import MotorAddress, Servo42CProtocol


def reply_frame(address: MotorAddress, status: int) -> bytes:
    """构造测试使用的合法三字节回包。"""

    return bytes((int(address), status, (int(address) + status) & 0xFF))


class Servo42CEncodingTest(unittest.TestCase):
    """验证固件使用的五类命令和速度、脉冲编码。"""

    def test_e3_enable_and_90_degree_frames_match_firmware(self) -> None:
        """E3 使能和 90°上下移动应与 C 固件报文完全一致。"""

        self.assertEqual(
            Servo42CProtocol.build_enable(MotorAddress.Z, True),
            bytes.fromhex("E3 F3 01 D7"),
        )
        self.assertEqual(
            Servo42CProtocol.build_move(
                MotorAddress.Z,
                -64,
                3200,
            ),
            bytes.fromhex("E3 FD C0 00 00 0C 80 2C"),
        )
        self.assertEqual(
            Servo42CProtocol.build_move(
                MotorAddress.Z,
                64,
                3200,
            ),
            bytes.fromhex("E3 FD 40 00 00 0C 80 AC"),
        )

    def test_simple_commands_append_checksum(self) -> None:
        """失能、停止和保护命令应追加低八位和校验。"""

        self.assertEqual(
            Servo42CProtocol.build_enable(MotorAddress.X0, False),
            bytes.fromhex("E0 F3 00 D3"),
        )
        self.assertEqual(
            Servo42CProtocol.build_stop(MotorAddress.Y),
            bytes.fromhex("E2 F7 D9"),
        )
        self.assertEqual(
            Servo42CProtocol.build_read_protect(MotorAddress.X1),
            bytes.fromhex("E1 3E 1F"),
        )
        self.assertEqual(
            Servo42CProtocol.build_clear_protect(MotorAddress.X1),
            bytes.fromhex("E1 3D 1E"),
        )

    def test_invalid_speed_pulses_and_address_are_rejected(self) -> None:
        """编码器应拒绝方向位溢出、零速移动和松散地址整数。"""

        with self.assertRaisesRegex(ValueError, "speed"):
            Servo42CProtocol.build_move(MotorAddress.Z, 128, 1)
        with self.assertRaisesRegex(ValueError, "速度 0"):
            Servo42CProtocol.build_move(MotorAddress.Z, 0, 1)
        with self.assertRaisesRegex(ValueError, "uint32"):
            Servo42CProtocol.build_move(MotorAddress.Z, 1, 0x100000000)
        with self.assertRaisesRegex(TypeError, "MotorAddress"):
            Servo42CProtocol.build_stop(0xE3)  # type: ignore[arg-type]


class Servo42CParserTest(unittest.TestCase):
    """验证流式解析的分块、连续帧、噪声和校验恢复。"""

    def test_fragmented_and_consecutive_replies_are_emitted(self) -> None:
        """一个回包可跨多次 feed，多个回包也可一次返回。"""

        parser = Servo42CProtocol()
        e0_started = reply_frame(MotorAddress.X0, 1)
        e3_completed = reply_frame(MotorAddress.Z, 2)

        self.assertEqual(parser.feed(e0_started[:2]), ())
        replies = parser.feed(e0_started[2:] + e3_completed)

        self.assertEqual(
            [(reply.address, reply.status) for reply in replies],
            [(MotorAddress.X0, 1), (MotorAddress.Z, 2)],
        )

    def test_noise_and_bad_checksum_resynchronize_one_byte_at_a_time(
        self,
    ) -> None:
        """错误候选只丢弃起点，后续合法地址仍应被发现。"""

        parser = Servo42CProtocol()
        invalid = bytes.fromhex("E1 01 00")
        valid = reply_frame(MotorAddress.Z, 2)

        replies = parser.feed(b"\x00\x7F" + invalid + valid)

        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0].address, MotorAddress.Z)
        self.assertEqual(replies[0].status, 2)
        self.assertEqual(parser.checksum_error_count, 1)
        self.assertGreaterEqual(parser.ignored_byte_count, 2)

    def test_unknown_addresses_are_ignored(self) -> None:
        """E4 等未配置地址不能形成可见回包。"""

        parser = Servo42CProtocol()

        replies = parser.feed(bytes.fromhex("E4 01 E5") + reply_frame(
            MotorAddress.Y,
            1,
        ))

        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0].address, MotorAddress.Y)

    def test_reset_discards_partial_frame_but_keeps_diagnostics(self) -> None:
        """C 风格清空应丢弃半帧，同时保留累计错误计数。"""

        parser = Servo42CProtocol()
        parser.feed(bytes.fromhex("E0 01 00"))
        parser.feed(bytes.fromhex("E1 01"))
        checksum_errors = parser.checksum_error_count

        parser.reset()
        replies = parser.feed(reply_frame(MotorAddress.Y, 1))

        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0].address, MotorAddress.Y)
        self.assertEqual(parser.checksum_error_count, checksum_errors)


if __name__ == "__main__":
    unittest.main()
