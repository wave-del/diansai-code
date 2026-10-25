"""定义 Servo42C 串口运动模块的稳定类型、异常和依赖契约。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
import math
from pathlib import Path
from typing import Protocol, runtime_checkable


DEFAULT_MOTION_DEVICE_PATH = Path("/dev/ttyAS4")
DEFAULT_MOTION_BAUDRATE = 115200
DEFAULT_MOTION_READ_TIMEOUT_SECONDS = 0.05
DEFAULT_MOTION_WRITE_TIMEOUT_SECONDS = 1.0
DEFAULT_MOTION_REPLY_TIMEOUT_SECONDS = 1.0


def _validate_positive_number(name: str, value: float) -> None:
    """校验配置值为有限正数，并拒绝布尔值冒充数字。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是实数")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须是有限正数")


class MotorAddress(IntEnum):
    """列出当前机构四个 Servo42C 的固定串口地址。"""

    X0 = 0xE0
    X1 = 0xE1
    Y = 0xE2
    Z = 0xE3


class XYState(IntEnum):
    """逐项对应 C 固件 ``XY_State`` 的状态编号。"""

    WAIT_POWER = 0
    WAIT_ENABLE_X0 = 1
    WAIT_ENABLE_X1 = 2
    WAIT_ENABLE_Y = 3
    READY = 4
    WAIT_MOVE_START = 5
    WAIT_MOVE_COMPLETE = 6
    FAULT = 7


class XYFault(IntEnum):
    """逐项对应 C 固件 ``XY_Fault`` 的锁存故障编号。"""

    NONE = 0
    ENABLE_FAILED = 1
    ENABLE_TIMEOUT = 2
    MOVE_REJECTED = 3
    MOVE_START_TIMEOUT = 4
    MOVE_COMPLETE_TIMEOUT = 5
    EMERGENCY_STOP = 6


class XYResult(IntEnum):
    """逐项对应 C 固件 ``XY_Result`` 的调用结果编号。"""

    OK = 0
    NOT_READY = 1
    BUSY = 2
    POSITION_INVALID = 3
    OUT_OF_RANGE = 4


@dataclass(frozen=True, slots=True)
class MotionSerialConfig:
    """描述打开 Servo42C 串口所需的部署参数和等待上限。"""

    device_path: Path = DEFAULT_MOTION_DEVICE_PATH
    baudrate: int = DEFAULT_MOTION_BAUDRATE
    read_timeout_seconds: float = DEFAULT_MOTION_READ_TIMEOUT_SECONDS
    write_timeout_seconds: float = DEFAULT_MOTION_WRITE_TIMEOUT_SECONDS
    reply_timeout_seconds: float = DEFAULT_MOTION_REPLY_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        """在访问设备前校验路径、波特率和全部超时。"""

        if not isinstance(self.device_path, Path):
            raise TypeError("device_path 必须是 pathlib.Path")
        if not str(self.device_path):
            raise ValueError("device_path 不能为空")
        if isinstance(self.baudrate, bool) or not isinstance(
            self.baudrate,
            int,
        ):
            raise TypeError("baudrate 必须是整数")
        if self.baudrate <= 0:
            raise ValueError("baudrate 必须大于 0")
        _validate_positive_number(
            "read_timeout_seconds",
            self.read_timeout_seconds,
        )
        _validate_positive_number(
            "write_timeout_seconds",
            self.write_timeout_seconds,
        )
        _validate_positive_number(
            "reply_timeout_seconds",
            self.reply_timeout_seconds,
        )


@dataclass(frozen=True, slots=True)
class ServoReply:
    """描述一条已通过地址和校验和验证的三字节回包。"""

    address: MotorAddress
    status: int
    frame: bytes

    def __post_init__(self) -> None:
        """校验回包结构，防止未验证的松散数据进入控制层。"""

        if not isinstance(self.address, MotorAddress):
            raise TypeError("address 必须是 MotorAddress")
        if isinstance(self.status, bool) or not isinstance(self.status, int):
            raise TypeError("status 必须是整数")
        if not 0 <= self.status <= 0xFF:
            raise ValueError("status 必须位于 [0, 255]")
        if not isinstance(self.frame, bytes):
            raise TypeError("frame 必须是 bytes")
        if len(self.frame) != 3:
            raise ValueError("Servo42C 回包必须为 3 字节")
        if self.frame[0] != int(self.address) or self.frame[1] != self.status:
            raise ValueError("frame 与 address/status 不一致")
        if (sum(self.frame[:2]) & 0xFF) != self.frame[2]:
            raise ValueError("frame 校验和无效")

    def to_dict(self) -> dict[str, object]:
        """转换为适合诊断 JSON 输出的结构。"""

        return {
            "address": self.address.name,
            "address_hex": f"0x{int(self.address):02X}",
            "status": self.status,
            "frame_hex": self.frame.hex(" ").upper(),
        }


@dataclass(frozen=True, slots=True)
class SerialWriteEvent:
    """记录一次完整报文写入调用的纳秒级软件时间戳。"""

    sequence: int
    frame: bytes
    wall_time_ns: int
    started_monotonic_ns: int
    completed_monotonic_ns: int
    delta_from_previous_start_ns: int | None
    bytes_written: int
    write_calls: int
    error_message: str | None = None

    def __post_init__(self) -> None:
        """校验时间戳、写入计数和错误信息的一致性。"""

        integer_fields = {
            "sequence": self.sequence,
            "wall_time_ns": self.wall_time_ns,
            "started_monotonic_ns": self.started_monotonic_ns,
            "completed_monotonic_ns": self.completed_monotonic_ns,
            "bytes_written": self.bytes_written,
            "write_calls": self.write_calls,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} 必须是整数")
        if self.sequence <= 0:
            raise ValueError("sequence 必须大于 0")
        if not isinstance(self.frame, bytes) or not self.frame:
            raise ValueError("frame 必须是非空 bytes")
        if self.wall_time_ns < 0 or self.started_monotonic_ns < 0:
            raise ValueError("时间戳不能小于 0")
        if self.completed_monotonic_ns < self.started_monotonic_ns:
            raise ValueError("完成时间不能早于开始时间")
        if self.delta_from_previous_start_ns is not None:
            delta = self.delta_from_previous_start_ns
            if isinstance(delta, bool) or not isinstance(delta, int):
                raise TypeError("delta_from_previous_start_ns 必须是整数或 None")
            if delta < 0:
                raise ValueError("相邻发送开始时间差不能小于 0")
        if not 0 <= self.bytes_written <= len(self.frame):
            raise ValueError("bytes_written 超出报文长度")
        if self.write_calls < 0:
            raise ValueError("write_calls 不能小于 0")
        if self.error_message is not None:
            if not isinstance(self.error_message, str):
                raise TypeError("error_message 必须是字符串或 None")
            if not self.error_message:
                raise ValueError("error_message 不能为空字符串")

    @property
    def write_duration_ns(self) -> int:
        """返回完整写入循环的单调时钟耗时。"""

        return self.completed_monotonic_ns - self.started_monotonic_ns

    @property
    def success(self) -> bool:
        """返回完整报文是否无异常交给串口驱动。"""

        return (
            self.error_message is None
            and self.bytes_written == len(self.frame)
        )

    @property
    def utc_timestamp(self) -> str:
        """把 Unix 纳秒时间转换为带九位小数的 UTC 文本。"""

        seconds, nanoseconds = divmod(self.wall_time_ns, 1_000_000_000)
        prefix = datetime.fromtimestamp(
            seconds,
            tz=timezone.utc,
        ).strftime("%Y-%m-%dT%H:%M:%S")
        return f"{prefix}.{nanoseconds:09d}Z"

    def to_dict(self) -> dict[str, object]:
        """转换为适合终端单行 JSON 输出的诊断结构。"""

        address_value = self.frame[0]
        try:
            address_name: str | None = MotorAddress(address_value).name
        except ValueError:
            address_name = None
        function_value = self.frame[1] if len(self.frame) > 1 else None
        delta = self.delta_from_previous_start_ns
        return {
            "event": "motion_serial_tx",
            "sequence": self.sequence,
            "utc_timestamp": self.utc_timestamp,
            "wall_time_ns": self.wall_time_ns,
            "started_monotonic_ns": self.started_monotonic_ns,
            "completed_monotonic_ns": self.completed_monotonic_ns,
            "delta_from_previous_start_ns": delta,
            "delta_from_previous_start_us": (
                None if delta is None else round(delta / 1_000, 3)
            ),
            "write_duration_ns": self.write_duration_ns,
            "write_duration_us": round(self.write_duration_ns / 1_000, 3),
            "address": address_name,
            "address_hex": f"0x{address_value:02X}",
            "function_hex": (
                None
                if function_value is None
                else f"0x{function_value:02X}"
            ),
            "frame_hex": self.frame.hex(" ").upper(),
            "frame_bytes": len(self.frame),
            "bytes_written": self.bytes_written,
            "write_calls": self.write_calls,
            "success": self.success,
            "error": self.error_message,
        }


@dataclass(frozen=True, slots=True)
class MotionCommand:
    """描述一条待发送或已发送的轴级 Servo42C 命令。"""

    operation: str
    axis: str
    address: MotorAddress
    frame: bytes
    pulses: int | None = None
    expected_statuses: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """校验命令元数据和不可变报文字节。"""

        if not isinstance(self.operation, str) or not self.operation.strip():
            raise ValueError("operation 必须是非空字符串")
        if self.axis not in {"x", "y", "z"}:
            raise ValueError("axis 必须为 x、y 或 z")
        if not isinstance(self.address, MotorAddress):
            raise TypeError("address 必须是 MotorAddress")
        if not isinstance(self.frame, bytes) or not self.frame:
            raise ValueError("frame 必须是非空 bytes")
        if self.pulses is not None:
            if isinstance(self.pulses, bool) or not isinstance(
                self.pulses,
                int,
            ):
                raise TypeError("pulses 必须是整数或 None")
            if self.pulses < 0:
                raise ValueError("pulses 不能小于 0")
        if self.expected_statuses is not None:
            if not isinstance(self.expected_statuses, tuple):
                raise TypeError("expected_statuses 必须是元组或 None")
            for status in self.expected_statuses:
                if isinstance(status, bool) or not isinstance(status, int):
                    raise TypeError("expected_statuses 只能包含整数")
                if not 0 <= status <= 0xFF:
                    raise ValueError("expected_statuses 必须位于 [0, 255]")

    def to_dict(self) -> dict[str, object]:
        """转换为命令行诊断使用的 JSON 兼容字典。"""

        expected_statuses = self.expected_statuses
        if expected_statuses is None:
            expected_statuses = (
                (1, 2) if self.operation == "move" else (1,)
            )
        return {
            "operation": self.operation,
            "axis": self.axis,
            "address": self.address.name,
            "address_hex": f"0x{int(self.address):02X}",
            "pulses": self.pulses,
            "frame_hex": self.frame.hex(" ").upper(),
            "expected_replies": [
                {
                    "status": status,
                    "frame_hex": bytes(
                        (
                            int(self.address),
                            status,
                            (int(self.address) + status) & 0xFF,
                        )
                    ).hex(" ").upper(),
                }
                for status in expected_statuses
            ],
        }


@dataclass(frozen=True, slots=True)
class MotionExecution:
    """描述一次轴操作发送的命令、收到的回包和耗时。"""

    commands: tuple[MotionCommand, ...]
    replies: tuple[ServoReply, ...]
    elapsed_seconds: float

    def __post_init__(self) -> None:
        """校验执行结果使用冻结元组和非负有限耗时。"""

        if not isinstance(self.commands, tuple):
            raise TypeError("commands 必须是元组")
        if not isinstance(self.replies, tuple):
            raise TypeError("replies 必须是元组")
        if isinstance(self.elapsed_seconds, bool) or not isinstance(
            self.elapsed_seconds,
            (int, float),
        ):
            raise TypeError("elapsed_seconds 必须是实数")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds 必须是非负有限数")

    def to_dict(self) -> dict[str, object]:
        """转换为外部诊断脚本的稳定摘要。"""

        return {
            "commands": [command.to_dict() for command in self.commands],
            "replies": [reply.to_dict() for reply in self.replies],
            "elapsed_seconds": round(self.elapsed_seconds, 6),
        }


class MotionSerialError(RuntimeError):
    """表示串口打开、读写或资源生命周期发生故障。"""


class MotionTimeoutError(MotionSerialError, TimeoutError):
    """表示在限定时间内没有收到目标 Servo42C 回包。"""


class MotionRejectedError(MotionSerialError):
    """表示 Servo42C 以状态 0 拒绝了当前命令。"""


class MotionProtocolError(MotionSerialError):
    """表示回包状态或帧序列不符合当前命令语义。"""


@runtime_checkable
class SerialPort(Protocol):
    """约束 Servo42CSerial 使用的最小同步串口对象。"""

    @property
    def in_waiting(self) -> int:
        """返回当前可立即读取的字节数。"""

    def write(self, data: bytes) -> int | None:
        """写入部分或全部报文字节，并返回写入数量。"""

    def read(self, size: int = 1) -> bytes:
        """读取最多指定数量的字节。"""

    def close(self) -> None:
        """关闭底层设备；重复关闭必须安全。"""


@runtime_checkable
class ServoTransport(Protocol):
    """约束轴控制器所需的报文发送和按地址等待能力。"""

    @property
    def config(self) -> MotionSerialConfig:
        """返回当前串口与回包超时配置。"""

    def send_frame(self, frame: bytes) -> None:
        """完整发送一条已经编码的 Servo42C 报文。"""

    def wait_reply(
        self,
        address: MotorAddress,
        *,
        expected_status: int | None = None,
        timeout_seconds: float | None = None,
    ) -> ServoReply:
        """等待指定地址回包，并可要求唯一状态值。"""


@runtime_checkable
class XYServoTransport(Protocol):
    """约束 C 风格 XY 状态机使用的发送和非阻塞回包接口。"""

    def send_frame(self, frame: bytes) -> None:
        """完整发送一条已经编码的 Servo42C 报文。"""

    def take_reply(self) -> ServoReply | None:
        """非阻塞取出当前最早有效回包，没有时返回 None。"""

    def discard_replies(self) -> None:
        """清空串口输入、解析中间状态和全部待处理回包。"""
