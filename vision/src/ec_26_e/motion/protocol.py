"""实现 Servo42C 报文编码、校验和流式三字节回包解析。"""

from __future__ import annotations

from collections.abc import Iterable

from ec_26_e.motion.contracts import MotorAddress, ServoReply


class Servo42CProtocol:
    """提供无设备依赖的 Servo42C 编码器和增量回包解析器。"""

    CODE_READ_PROTECT = 0x3E
    CODE_CLEAR_PROTECT = 0x3D
    CODE_SET_ENABLE = 0xF3
    CODE_STOP = 0xF7
    CODE_MOVE = 0xFD
    REPLY_LENGTH = 3

    def __init__(self) -> None:
        """创建空接收缓冲区和解析诊断计数器。"""

        self._buffer = bytearray()
        self.checksum_error_count = 0
        self.ignored_byte_count = 0

    def reset(self) -> None:
        """丢弃未完成帧，同时保留累计诊断计数。"""

        self._buffer.clear()

    @staticmethod
    def checksum(data: bytes | bytearray | memoryview) -> int:
        """返回数据逐字节相加后的低 8 位校验值。"""

        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data 必须是字节序列")
        return sum(data) & 0xFF

    @staticmethod
    def _address_value(address: MotorAddress) -> int:
        """校验并返回强类型电机地址的整数值。"""

        if not isinstance(address, MotorAddress):
            raise TypeError("address 必须是 MotorAddress")
        return int(address)

    @classmethod
    def _frame(
        cls,
        payload: Iterable[int],
    ) -> bytes:
        """为命令载荷追加 CHECKSUM-8 并返回不可变报文。"""

        try:
            data = bytes(payload)
        except (TypeError, ValueError) as error:
            raise ValueError("命令载荷必须只包含字节值") from error
        return data + bytes((cls.checksum(data),))

    @staticmethod
    def encode_speed(speed: int) -> int:
        """把有符号速度转换为方向位加低七位速度。"""

        if isinstance(speed, bool) or not isinstance(speed, int):
            raise TypeError("speed 必须是整数")
        if not -127 <= speed <= 127:
            raise ValueError("speed 必须位于 [-127, 127]")
        if speed < 0:
            return abs(speed) | 0x80
        return speed

    @classmethod
    def build_enable(
        cls,
        address: MotorAddress,
        enabled: bool,
    ) -> bytes:
        """构造使能或失能命令。"""

        if not isinstance(enabled, bool):
            raise TypeError("enabled 必须是 bool")
        return cls._frame(
            (
                cls._address_value(address),
                cls.CODE_SET_ENABLE,
                0x01 if enabled else 0x00,
            )
        )

    @classmethod
    def build_move(
        cls,
        address: MotorAddress,
        speed: int,
        pulses: int,
    ) -> bytes:
        """构造相对脉冲移动命令，脉冲按 uint32 大端序编码。"""

        if isinstance(pulses, bool) or not isinstance(pulses, int):
            raise TypeError("pulses 必须是整数")
        if not 0 <= pulses <= 0xFFFFFFFF:
            raise ValueError("pulses 必须位于 uint32 范围")
        encoded_speed = cls.encode_speed(speed)
        if pulses > 0 and encoded_speed == 0:
            raise ValueError("非零脉冲移动不能使用速度 0")
        return cls._frame(
            (
                cls._address_value(address),
                cls.CODE_MOVE,
                encoded_speed,
                (pulses >> 24) & 0xFF,
                (pulses >> 16) & 0xFF,
                (pulses >> 8) & 0xFF,
                pulses & 0xFF,
            )
        )

    @classmethod
    def build_stop(cls, address: MotorAddress) -> bytes:
        """构造立即停止命令。"""

        return cls._frame(
            (cls._address_value(address), cls.CODE_STOP)
        )

    @classmethod
    def build_read_protect(cls, address: MotorAddress) -> bytes:
        """构造查询堵转保护状态命令。"""

        return cls._frame(
            (cls._address_value(address), cls.CODE_READ_PROTECT)
        )

    @classmethod
    def build_clear_protect(cls, address: MotorAddress) -> bytes:
        """构造清除堵转保护命令。"""

        return cls._frame(
            (cls._address_value(address), cls.CODE_CLEAR_PROTECT)
        )

    def feed(self, data: bytes | bytearray | memoryview) -> tuple[ServoReply, ...]:
        """追加任意分块字节并返回新解析出的全部有效回包。"""

        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data 必须是字节序列")
        self._buffer.extend(data)
        replies: list[ServoReply] = []
        valid_addresses = {int(address) for address in MotorAddress}

        while self._buffer:
            if self._buffer[0] not in valid_addresses:
                del self._buffer[0]
                self.ignored_byte_count += 1
                continue
            if len(self._buffer) < self.REPLY_LENGTH:
                break

            candidate = bytes(self._buffer[: self.REPLY_LENGTH])
            if self.checksum(candidate[:2]) != candidate[2]:
                # 只丢弃当前候选起点，保留后续字节用于重新同步。
                del self._buffer[0]
                self.checksum_error_count += 1
                continue

            del self._buffer[: self.REPLY_LENGTH]
            replies.append(
                ServoReply(
                    address=MotorAddress(candidate[0]),
                    status=candidate[1],
                    frame=candidate,
                )
            )

        return tuple(replies)
