"""封装 pyserial 同步生命周期、完整写入和按地址回包等待。"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
import math
import time

from ec_26_e.motion.contracts import (
    MotionProtocolError,
    MotionRejectedError,
    MotionSerialConfig,
    MotionSerialError,
    MotionTimeoutError,
    MotorAddress,
    SerialWriteEvent,
    SerialPort,
    ServoReply,
)
from ec_26_e.motion.protocol import Servo42CProtocol


SerialFactory = Callable[[MotionSerialConfig], SerialPort]
WriteObserver = Callable[[SerialWriteEvent], None]
NanosecondClock = Callable[[], int]


def _open_pyserial(config: MotionSerialConfig) -> SerialPort:
    """延迟导入并按 8N1、无流控配置打开真实串口。"""

    try:
        import serial
    except ImportError as error:
        raise MotionSerialError(
            "未安装 pyserial；请先安装项目依赖"
        ) from error

    try:
        return serial.Serial(
            port=str(config.device_path),
            baudrate=config.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=config.read_timeout_seconds,
            write_timeout=config.write_timeout_seconds,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )
    except (OSError, serial.SerialException) as error:
        raise MotionSerialError(
            f"无法打开运动串口 {config.device_path}：{error}"
        ) from error


class Servo42CSerial:
    """管理单一同步串口，并保留未被当前请求消费的有效回包。"""

    def __init__(
        self,
        config: MotionSerialConfig = MotionSerialConfig(),
        *,
        serial_factory: SerialFactory = _open_pyserial,
        write_observer: WriteObserver | None = None,
        wall_time_ns: NanosecondClock = time.time_ns,
        monotonic_time_ns: NanosecondClock = time.monotonic_ns,
    ) -> None:
        """保存配置和依赖；构造时不打开串口或访问设备。"""

        if not isinstance(config, MotionSerialConfig):
            raise TypeError("config 必须是 MotionSerialConfig")
        if not callable(serial_factory):
            raise TypeError("serial_factory 必须可调用")
        if write_observer is not None and not callable(write_observer):
            raise TypeError("write_observer 必须可调用或为 None")
        if not callable(wall_time_ns) or not callable(monotonic_time_ns):
            raise TypeError("纳秒时钟必须可调用")
        self._config = config
        self._serial_factory = serial_factory
        self._write_observer = write_observer
        self._wall_time_ns = wall_time_ns
        self._monotonic_time_ns = monotonic_time_ns
        self._write_sequence = 0
        self._previous_write_started_monotonic_ns: int | None = None
        self._port: SerialPort | None = None
        self._protocol = Servo42CProtocol()
        self._pending_order: deque[ServoReply] = deque()
        self._pending_by_address: dict[
            MotorAddress,
            deque[ServoReply],
        ] = {
            address: deque()
            for address in MotorAddress
        }

    @property
    def config(self) -> MotionSerialConfig:
        """返回冻结的串口配置。"""

        return self._config

    @property
    def is_open(self) -> bool:
        """返回当前对象是否持有已创建的串口实例。"""

        return self._port is not None

    @property
    def protocol(self) -> Servo42CProtocol:
        """返回解析器，以便诊断校验错误和忽略字节计数。"""

        return self._protocol

    def open(self) -> None:
        """幂等打开底层串口。"""

        if self._port is not None:
            return
        try:
            port = self._serial_factory(self._config)
        except MotionSerialError:
            raise
        except Exception as error:
            raise MotionSerialError(
                f"创建运动串口失败：{error}"
            ) from error
        if not isinstance(port, SerialPort):
            try:
                port.close()
            except Exception:
                pass
            raise TypeError("serial_factory 必须返回 SerialPort")
        self._port = port

    def close(self) -> None:
        """幂等关闭串口，并保留关闭异常的明确语义。"""

        port = self._port
        self._port = None
        if port is None:
            return
        try:
            port.close()
        except Exception as error:
            raise MotionSerialError(f"关闭运动串口失败：{error}") from error

    def __enter__(self) -> Servo42CSerial:
        """打开串口并返回当前上下文对象。"""

        self.open()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        """退出上下文时始终关闭串口。"""

        self.close()

    def _require_port(self) -> SerialPort:
        """返回已打开串口，未打开时拒绝隐式访问设备。"""

        if self._port is None:
            raise MotionSerialError("运动串口尚未打开")
        return self._port

    def send_frame(self, frame: bytes) -> None:
        """完整发送报文，并向观察器报告纳秒级软件写入时间。"""

        if not isinstance(frame, bytes):
            raise TypeError("frame 必须是 bytes")
        if not frame:
            raise ValueError("frame 不能为空")
        port = self._require_port()
        self._write_sequence += 1
        wall_time_ns = self._wall_time_ns()
        started_monotonic_ns = self._monotonic_time_ns()
        previous_start = self._previous_write_started_monotonic_ns
        self._previous_write_started_monotonic_ns = started_monotonic_ns
        delta_from_previous_start_ns = (
            None
            if previous_start is None
            else started_monotonic_ns - previous_start
        )
        offset = 0
        write_calls = 0
        error_message: str | None = None
        try:
            while offset < len(frame):
                write_calls += 1
                written = port.write(frame[offset:])
                if written is None or written <= 0:
                    raise MotionSerialError("串口写入没有取得进展")
                if written > len(frame) - offset:
                    raise MotionSerialError("串口报告了无效写入长度")
                offset += written
        except MotionSerialError as error:
            error_message = str(error)
            raise
        except Exception as error:
            error_message = str(error)
            raise MotionSerialError(
                f"发送 Servo42C 报文失败：{error}"
            ) from error
        finally:
            completed_monotonic_ns = self._monotonic_time_ns()
            event = SerialWriteEvent(
                sequence=self._write_sequence,
                frame=frame,
                wall_time_ns=wall_time_ns,
                started_monotonic_ns=started_monotonic_ns,
                completed_monotonic_ns=completed_monotonic_ns,
                delta_from_previous_start_ns=delta_from_previous_start_ns,
                bytes_written=offset,
                write_calls=write_calls,
                error_message=error_message,
            )
            self._notify_write_observer(event)

    def _notify_write_observer(self, event: SerialWriteEvent) -> None:
        """报告写入事件，诊断输出失败不得改变电机控制结果。"""

        observer = self._write_observer
        if observer is None:
            return
        try:
            observer(event)
        except Exception:
            # 诊断输出故障不能把已发送报文改判为控制失败。
            pass

    def _read_into_pending(self) -> None:
        """从串口读取一块数据并把有效回包追加到待处理队列。"""

        port = self._require_port()
        try:
            waiting = port.in_waiting
            if isinstance(waiting, bool) or not isinstance(waiting, int):
                raise MotionSerialError("串口 in_waiting 必须是整数")
            data = port.read(max(1, min(waiting, 64)))
        except MotionSerialError:
            raise
        except Exception as error:
            raise MotionSerialError(
                f"读取 Servo42C 回包失败：{error}"
            ) from error
        if not isinstance(data, bytes):
            raise MotionSerialError("串口 read 必须返回 bytes")
        if data:
            for reply in self._protocol.feed(data):
                self._pending_order.append(reply)
                self._pending_by_address[reply.address].append(reply)

    def take_reply(self) -> ServoReply | None:
        """按 C 的 ``takeReply`` 语义非阻塞取出最早有效回包。"""

        reply = self._take_pending(None)
        if reply is not None:
            return reply
        port = self._require_port()
        try:
            waiting = port.in_waiting
        except Exception as error:
            raise MotionSerialError(
                f"查询 Servo42C 回包失败：{error}"
            ) from error
        if isinstance(waiting, bool) or not isinstance(waiting, int):
            raise MotionSerialError("串口 in_waiting 必须是整数")
        if waiting <= 0:
            return None
        self._read_into_pending()
        return self._take_pending(None)

    def discard_replies(self) -> None:
        """复刻 C 的 ``discardReplies``，清空所有接收中间状态。"""

        port = self._require_port()
        self._pending_order.clear()
        for replies in self._pending_by_address.values():
            replies.clear()
        self._protocol.reset()
        try:
            while True:
                waiting = port.in_waiting
                if isinstance(waiting, bool) or not isinstance(waiting, int):
                    raise MotionSerialError("串口 in_waiting 必须是整数")
                if waiting <= 0:
                    break
                data = port.read(min(waiting, 64))
                if not isinstance(data, bytes):
                    raise MotionSerialError("串口 read 必须返回 bytes")
                if not data:
                    break
        except MotionSerialError:
            raise
        except Exception as error:
            raise MotionSerialError(
                f"清空 Servo42C 回包失败：{error}"
            ) from error

    def _take_pending(
        self,
        address: MotorAddress | None,
    ) -> ServoReply | None:
        """取出任意或指定地址的最早回包，保留其他回包顺序。"""

        if address is None:
            if not self._pending_order:
                return None
            reply = self._pending_order.popleft()
            by_address = self._pending_by_address[reply.address]
            if not by_address or by_address.popleft() is not reply:
                raise MotionProtocolError("Servo42C 待处理回包队列失去同步")
            return reply

        by_address = self._pending_by_address[address]
        if not by_address:
            return None
        reply = by_address.popleft()
        for index, queued_reply in enumerate(self._pending_order):
            if queued_reply is reply:
                del self._pending_order[index]
                return reply
        raise MotionProtocolError("Servo42C 待处理回包队列失去同步")

    def read_reply(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> ServoReply:
        """读取下一条有效回包，超时后抛出 MotionTimeoutError。"""

        return self._wait_reply(
            None,
            expected_status=None,
            timeout_seconds=timeout_seconds,
        )

    def wait_reply(
        self,
        address: MotorAddress,
        *,
        expected_status: int | None = None,
        timeout_seconds: float | None = None,
    ) -> ServoReply:
        """等待指定地址和可选状态，同时保留其他地址回包。"""

        if not isinstance(address, MotorAddress):
            raise TypeError("address 必须是 MotorAddress")
        return self._wait_reply(
            address,
            expected_status=expected_status,
            timeout_seconds=timeout_seconds,
        )

    def _wait_reply(
        self,
        address: MotorAddress | None,
        *,
        expected_status: int | None,
        timeout_seconds: float | None,
    ) -> ServoReply:
        """实现任意或按地址等待的共同截止时间与状态检查。"""

        timeout = (
            self._config.reply_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("timeout_seconds 必须是实数或 None")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_seconds 必须是有限正数")
        if expected_status is not None:
            if isinstance(expected_status, bool) or not isinstance(
                expected_status,
                int,
            ):
                raise TypeError("expected_status 必须是整数或 None")
            if not 0 <= expected_status <= 0xFF:
                raise ValueError("expected_status 必须位于 [0, 255]")

        deadline = time.monotonic() + float(timeout)
        while True:
            reply = self._take_pending(address)
            if reply is not None:
                if reply.status == 0:
                    target = reply.address.name
                    raise MotionRejectedError(
                        f"Servo42C {target} 拒绝了当前命令"
                    )
                if (
                    expected_status is not None
                    and reply.status != expected_status
                ):
                    raise MotionProtocolError(
                        f"Servo42C {reply.address.name} 回包状态 "
                        f"{reply.status}，预期 {expected_status}"
                    )
                return reply
            if time.monotonic() >= deadline:
                target = address.name if address is not None else "任意地址"
                raise MotionTimeoutError(
                    f"等待 Servo42C {target} 回包超时（{timeout:.3f} 秒）"
                )
            self._read_into_pending()

    def set_enable(
        self,
        address: MotorAddress,
        enabled: bool,
    ) -> ServoReply:
        """发送使能/失能命令并要求状态 1 成功回包。"""

        frame = Servo42CProtocol.build_enable(address, enabled)
        self.send_frame(frame)
        return self.wait_reply(address, expected_status=1)

    def stop(self, address: MotorAddress) -> ServoReply:
        """发送停止命令并要求状态 1 成功回包。"""

        frame = Servo42CProtocol.build_stop(address)
        self.send_frame(frame)
        return self.wait_reply(address, expected_status=1)

    def move(
        self,
        address: MotorAddress,
        speed: int,
        pulses: int,
        *,
        completion_timeout_seconds: float,
    ) -> tuple[ServoReply, ServoReply]:
        """发送单电机移动并依次等待开始和完成回包。"""

        frame = Servo42CProtocol.build_move(address, speed, pulses)
        self.send_frame(frame)
        started = self.wait_reply(address, expected_status=1)
        completed = self.wait_reply(
            address,
            expected_status=2,
            timeout_seconds=completion_timeout_seconds,
        )
        return started, completed
