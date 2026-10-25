"""WHEELTEC F32C 无刷云台电机同步控制接口。

本模块只依赖 Python 标准库，直接使用 Linux 3.3V TTL UART。
Q6A 上必须显式配置串口；默认不选择设备，不允许使用系统控制台。

坐标约定：

* ``move_to()`` 使用驱动板中持久保存的机械零点作为 0°；
* 普通绝对目标范围为 ``0 <= angle < 360``；
* 普通绝对定位始终选择最短路径；
* 明确需要完整旋转一圈时使用 ``rotate_one_turn()``。

连接和查询不会使电机运动。运动前必须显式调用 ``enable()``。实测本机
F32C 的持久机械角度与上电多圈角度方向相反，控制器已在坐标换算中处理。
"""

from __future__ import annotations

import fcntl
import math
import os
import select
import struct
import termios
import threading
import time
from dataclasses import dataclass


__version__ = "1.1.0"

DEFAULT_PORT = ""
DEFAULT_BAUDRATE = 115200
DEFAULT_ADDRESS = 2
DEFAULT_RPM = 15
DEFAULT_MAX_RPM = 30
DEFAULT_MECHANICAL_DIRECTION = -1
DEFAULT_FEEDBACK_ATTEMPTS = 3
COMMAND_GAP_SECONDS = 0.002

FRAME_HEADER = 0x7A
FRAME_TAIL = 0x7B

FUNC_MODE = 0x00
FUNC_SPEED = 0x01
FUNC_MULTI_TURN_ANGLE = 0x02
FUNC_DISABLE = 0x05
FUNC_ENABLE = 0x06
FUNC_FEEDBACK_QUERY = 0x0E

MODE_MULTI_T_PROFILE = b"\x00\x01"
FEEDBACK_TOTAL_ANGLE = 0x01
FEEDBACK_MECHANICAL_ANGLE = 0x02


class F32CError(Exception):
    """F32C 控制错误基类。"""


class F32CConnectionError(F32CError):
    """串口无法打开、被占用或已断开。"""


class F32CProtocolError(F32CError):
    """收到的反馈帧不符合 F32C 协议。"""


class F32CTimeoutError(F32CError, TimeoutError):
    """等待合法反馈或等待电机到位超时。"""


def calculate_bcc(data: bytes) -> int:
    """返回逐字节异或 BCC。"""
    result = 0
    for value in data:
        result ^= value
    return result


def make_frame(address: int, function: int, data: bytes = b"") -> bytes:
    """构造 F32C 指令帧。"""
    if (
        not 0 <= int(address) <= 0xFF
        or int(address) != address
    ):
        raise ValueError("电机地址必须在 0～255")
    if (
        not 0 <= int(function) <= 0xFF
        or int(function) != function
    ):
        raise ValueError("功能码必须在 0～255")
    body = bytes((FRAME_HEADER, int(address), int(function))) + bytes(data)
    return body + bytes((calculate_bcc(body), FRAME_TAIL))


def build_speed_frame(rpm: int, address: int = DEFAULT_ADDRESS) -> bytes:
    """构造位置模式速度指令；速度单位为 RPM。"""
    speed = int(rpm)
    if speed != rpm or not 1 <= speed <= 1000:
        raise ValueError("速度必须在 1～1000 RPM")
    return make_frame(address, FUNC_SPEED, struct.pack(">H", speed))


def build_multi_turn_angle_frame(
    total_angle_deg: float,
    address: int = DEFAULT_ADDRESS,
) -> bytes:
    """构造内部多圈绝对位置指令。

    ``total_angle_deg`` 是相对于本次驱动板上电位置的有符号多圈角度，
    不是持久机械角度。普通业务不应直接调用本函数。
    """
    angle = float(total_angle_deg)
    if not math.isfinite(angle):
        raise ValueError("多圈目标角度必须是有限数字")
    angle_x10 = round(angle * 10.0)
    if not -(2**31) <= angle_x10 <= 2**31 - 1:
        raise ValueError("多圈目标超出32位协议范围")
    return make_frame(
        address,
        FUNC_MULTI_TURN_ANGLE,
        struct.pack(">i", angle_x10),
    )


def shortest_delta_deg(current_deg: float, target_deg: float) -> float:
    """计算从当前单圈角度到目标角度的最短有符号位移。"""
    current = float(current_deg) % 360.0
    target = float(target_deg) % 360.0
    delta = (target - current + 180.0) % 360.0 - 180.0
    # 正好相差180°时固定选择正方向，避免浮点/平台差异。
    if math.isclose(delta, -180.0, abs_tol=1e-9):
        return 180.0
    return delta


@dataclass(frozen=True)
class FeedbackFrame:
    """经过帧头、地址、帧尾和 BCC 校验的角度反馈。"""

    feedback_type: int
    angle_x10: int
    raw: bytes

    @property
    def angle_deg(self) -> float:
        return self.angle_x10 / 10.0


def decode_feedback_frame(
    frame: bytes,
    address: int = DEFAULT_ADDRESS,
) -> FeedbackFrame:
    """解析总角度或机械角度反馈。

    总角度反馈为4字节有符号值（完整帧9字节）。不同批次F32C资料中的
    单圈机械角度存在2字节和4字节两种表现，因此这里兼容两者。
    """
    raw = bytes(frame)
    if len(raw) not in (7, 9):
        raise F32CProtocolError("角度反馈帧长度必须是7或9字节")
    if raw[0] != FRAME_HEADER or raw[-1] != FRAME_TAIL:
        raise F32CProtocolError("反馈帧头或帧尾错误")
    if raw[1] != int(address):
        raise F32CProtocolError(
            f"反馈地址错误：期望 {address}，收到 {raw[1]}"
        )
    if raw[-2] != calculate_bcc(raw[:-2]):
        raise F32CProtocolError("反馈 BCC 校验失败")

    feedback_type = raw[2]
    payload = raw[3:-2]
    if feedback_type == FEEDBACK_TOTAL_ANGLE:
        if len(payload) != 4:
            raise F32CProtocolError("总角度反馈必须包含4字节数据")
        angle_x10 = int.from_bytes(payload, "big", signed=True)
    elif feedback_type == FEEDBACK_MECHANICAL_ANGLE:
        if len(payload) == 2:
            angle_x10 = int.from_bytes(payload, "big", signed=False)
        elif len(payload) == 4:
            angle_x10 = int.from_bytes(payload, "big", signed=True)
        else:
            raise F32CProtocolError("机械角度反馈必须包含2或4字节数据")
        if not 0 <= angle_x10 <= 3600:
            raise F32CProtocolError(
                f"机械角度反馈越界：{angle_x10 / 10.0:.1f}°"
            )
    else:
        raise F32CProtocolError(
            f"不支持的反馈类型：0x{feedback_type:02X}"
        )
    return FeedbackFrame(feedback_type, angle_x10, raw)


class FeedbackStreamParser:
    """可从分段、粘包或带噪声的串口字节流中恢复角度反馈。"""

    def __init__(self, address: int = DEFAULT_ADDRESS):
        self.address = int(address)
        self.buffer = bytearray()

    def clear(self) -> None:
        self.buffer.clear()

    def feed(self, data: bytes) -> list[FeedbackFrame]:
        self.buffer.extend(data)
        frames: list[FeedbackFrame] = []

        while True:
            try:
                header_index = self.buffer.index(FRAME_HEADER)
            except ValueError:
                self.buffer.clear()
                break

            if header_index:
                del self.buffer[:header_index]
            if len(self.buffer) < 3:
                break
            if self.buffer[1] != self.address:
                del self.buffer[0]
                continue

            feedback_type = self.buffer[2]
            if feedback_type == FEEDBACK_TOTAL_ANGLE:
                candidate_lengths = (9,)
            elif feedback_type == FEEDBACK_MECHANICAL_ANGLE:
                # 当前真实F32C返回9字节；仍兼容资料中的7字节批次。
                candidate_lengths = (9, 7)
            else:
                del self.buffer[0]
                continue

            parsed = None
            for frame_length in candidate_lengths:
                if len(self.buffer) < frame_length:
                    continue
                candidate = bytes(self.buffer[:frame_length])
                try:
                    parsed = decode_feedback_frame(candidate, self.address)
                except F32CProtocolError:
                    continue
                del self.buffer[:frame_length]
                frames.append(parsed)
                break

            if parsed is not None:
                continue
            if len(self.buffer) < max(candidate_lengths):
                break
            # 当前帧头无法组成合法帧；丢弃一个字节后重新寻找帧头。
            del self.buffer[0]

        return frames


class LinuxUart:
    """无第三方依赖的 Linux 115200 8N1 串口传输层。"""

    def __init__(self, port: str, baudrate: int = DEFAULT_BAUDRATE):
        if baudrate != 115200:
            raise ValueError("当前交付仅支持已验证的 115200 bit/s")
        self.port = str(port)
        if not self.port.strip():
            raise ValueError("Q6A F32C 串口未配置，必须显式指定 port")
        consoles = {"/dev/console", "/dev/ttyMSM0"}
        try:
            with open("/proc/cmdline", encoding="utf-8") as stream:
                for item in stream.read().split():
                    if item.startswith("console="):
                        consoles.add("/dev/" + item.split("=", 1)[1].split(",", 1)[0])
        except OSError:
            pass
        if os.path.realpath(self.port) in consoles:
            raise ValueError("拒绝使用系统控制台作为 F32C 串口：" + self.port)
        self.baudrate = int(baudrate)
        self._fd: int | None = None
        self._original_attributes = None

    @property
    def is_open(self) -> bool:
        return self._fd is not None

    def open(self) -> None:
        if self.is_open:
            raise F32CConnectionError("串口已经打开")
        try:
            fd = os.open(
                self.port,
                os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK | os.O_CLOEXEC,
            )
        except OSError as exc:
            raise F32CConnectionError(
                f"无法打开串口 {self.port}: {exc.strerror or exc}"
            ) from exc

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if hasattr(termios, "TIOCEXCL"):
                fcntl.ioctl(fd, termios.TIOCEXCL)

            self._original_attributes = termios.tcgetattr(fd)
            attributes = termios.tcgetattr(fd)
            attributes[0] = 0
            attributes[1] = 0
            attributes[2] = (
                termios.B115200 | termios.CS8 | termios.CLOCAL | termios.CREAD
            )
            attributes[3] = 0
            attributes[4] = termios.B115200
            attributes[5] = termios.B115200
            attributes[6][termios.VMIN] = 0
            attributes[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSANOW, attributes)
            termios.tcflush(fd, termios.TCIOFLUSH)
        except OSError as exc:
            os.close(fd)
            raise F32CConnectionError(
                f"无法独占或配置串口 {self.port}: {exc.strerror or exc}"
            ) from exc

        self._fd = fd

    def close(self) -> None:
        if self._fd is None:
            return
        fd = self._fd
        self._fd = None
        try:
            if self._original_attributes is not None:
                termios.tcsetattr(
                    fd, termios.TCSANOW, self._original_attributes
                )
        except OSError:
            pass
        finally:
            self._original_attributes = None
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)

    def flush_input(self) -> None:
        fd = self._require_fd()
        termios.tcflush(fd, termios.TCIFLUSH)

    def write_all(self, data: bytes, timeout: float = 1.0) -> None:
        fd = self._require_fd()
        payload = memoryview(bytes(data))
        offset = 0
        deadline = time.monotonic() + float(timeout)
        while offset < len(payload):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise F32CTimeoutError(
                    f"串口发送超时：{offset}/{len(payload)} 字节"
                )
            _, writable, _ = select.select(
                [], [fd], [], min(0.1, remaining)
            )
            if writable:
                try:
                    offset += os.write(fd, payload[offset:])
                except BlockingIOError:
                    continue
                except OSError as exc:
                    raise F32CConnectionError(
                        f"串口发送失败: {exc.strerror or exc}"
                    ) from exc
        try:
            termios.tcdrain(fd)
        except OSError as exc:
            raise F32CConnectionError(
                f"等待串口发送完成失败: {exc.strerror or exc}"
            ) from exc

    def read_some(self, timeout: float) -> bytes:
        fd = self._require_fd()
        try:
            readable, _, _ = select.select([fd], [], [], max(0.0, timeout))
            if not readable:
                return b""
            return os.read(fd, 4096)
        except BlockingIOError:
            return b""
        except OSError as exc:
            raise F32CConnectionError(
                f"串口读取失败: {exc.strerror or exc}"
            ) from exc

    def _require_fd(self) -> int:
        if self._fd is None:
            raise F32CConnectionError("串口未打开")
        return self._fd


class F32CMotor:
    """F32C 电机同步高层接口。

    同一个对象应由同一业务线程顺序调用。类内部的锁用于防止意外重入，
    不代表允许多个进程或多个控制器同时操作同一个 UART。
    """

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        address: int = DEFAULT_ADDRESS,
        feedback_timeout: float = 0.3,
        command_gap: float = COMMAND_GAP_SECONDS,
        max_rpm: int = DEFAULT_MAX_RPM,
        mechanical_direction: int = DEFAULT_MECHANICAL_DIRECTION,
        feedback_attempts: int = DEFAULT_FEEDBACK_ATTEMPTS,
    ):
        if (
            not 0 <= int(address) <= 0xFF
            or int(address) != address
        ):
            raise ValueError("电机地址必须在 0～255")
        if not math.isfinite(feedback_timeout) or feedback_timeout <= 0:
            raise ValueError("反馈超时必须大于0")
        if not math.isfinite(command_gap) or command_gap < 0.001:
            raise ValueError("命令间隔不得小于1ms")
        if not 1 <= int(max_rpm) <= 1000 or int(max_rpm) != max_rpm:
            raise ValueError("速度上限必须是 1～1000 之间的整数 RPM")
        if (
            int(mechanical_direction) not in (-1, 1)
            or int(mechanical_direction) != mechanical_direction
        ):
            raise ValueError("机械角度方向必须是 1 或 -1")
        if (
            not 1 <= int(feedback_attempts) <= 10
            or int(feedback_attempts) != feedback_attempts
        ):
            raise ValueError("反馈查询次数必须是 1～10 之间的整数")

        self.port = str(port)
        self.address = int(address)
        self.feedback_timeout = float(feedback_timeout)
        self.command_gap = float(command_gap)
        self.max_rpm = int(max_rpm)
        self.mechanical_direction = int(mechanical_direction)
        self.feedback_attempts = int(feedback_attempts)
        self._uart = LinuxUart(self.port)
        self._parser = FeedbackStreamParser(self.address)
        self._lock = threading.RLock()
        self._enabled = False
        self._rpm = DEFAULT_RPM

    @property
    def is_connected(self) -> bool:
        return self._uart.is_open

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def rpm(self) -> int:
        return self._rpm

    def connect(self, verify: bool = False) -> None:
        """打开 UART，不使能、不运动。

        实测F32C在失能状态可能不返回角度反馈，因此默认不做查询验证。
        ``verify=True`` 仅适用于已确认失能时仍会响应的驱动状态。
        """
        with self._lock:
            self._uart.open()
            self._parser.clear()
            try:
                if verify:
                    self.get_total_angle()
                    self.get_angle()
            except Exception:
                self._uart.close()
                raise

    def close(self, disable: bool = True) -> None:
        """关闭 UART；默认先尽力失能电机。"""
        with self._lock:
            if disable and self.is_connected and self._enabled:
                self._safe_disable()
            self._uart.close()
            self._enabled = False
            self._parser.clear()

    def enable(self, rpm: int = DEFAULT_RPM) -> None:
        """安全使能电机、选择多圈T型位置模式并设置速度。

        F32C可能在重新使能时继续追踪上次残留目标。本方法先在失能状态
        预设1 RPM，随后使能、立即读取当前位置并下发同位置保持目标，最后
        才设置请求速度，以避免启动后继续执行旧目标。
        """
        speed = self._validate_rpm(rpm)
        with self._lock:
            self._require_connected()
            # 模式和最低速度先于使能下发，降低残留目标带来的启动位移。
            self._send(
                make_frame(self.address, FUNC_MODE, MODE_MULTI_T_PROFILE)
            )
            self._send(build_speed_frame(1, self.address))
            self._rpm = 1
            self._send(make_frame(self.address, FUNC_ENABLE))
            self._enabled = True
            try:
                current_total = self.get_total_angle()
                self._send(
                    build_multi_turn_angle_frame(
                        current_total,
                        self.address,
                    )
                )
                if speed != 1:
                    self._send(build_speed_frame(speed, self.address))
                self._rpm = speed
            except Exception:
                self._safe_disable()
                raise

    def disable(self) -> None:
        """立即发送失能命令。"""
        with self._lock:
            self._require_connected()
            self._safe_disable()

    def set_speed(self, rpm: int) -> None:
        """设置后续位置运动的速度，单位 RPM。"""
        speed = self._validate_rpm(rpm)
        with self._lock:
            self._require_connected()
            self._send(build_speed_frame(speed, self.address))
            self._rpm = speed

    def get_total_angle(self) -> float:
        """读取相对于本次驱动板上电位置的有符号多圈角度，单位°。"""
        with self._lock:
            feedback = self._query(FEEDBACK_TOTAL_ANGLE)
            return feedback.angle_deg

    def get_angle(self) -> float:
        """读取相对于持久机械零点的单圈角度，范围为 [0, 360)，单位°。"""
        with self._lock:
            feedback = self._query(FEEDBACK_MECHANICAL_ANGLE)
            return feedback.angle_deg % 360.0

    def get_status(self) -> dict[str, float | int | bool | str]:
        """读取并返回适合日志/界面显示的状态快照。"""
        with self._lock:
            return {
                "port": self.port,
                "address": self.address,
                "connected": self.is_connected,
                "enabled": self._enabled,
                "rpm": self._rpm,
                "mechanical_angle_deg": self.get_angle(),
                "total_angle_deg": self.get_total_angle(),
            }

    def move_to(
        self,
        angle_deg: float,
        rpm: int | None = None,
        wait: bool = True,
        timeout: float | None = None,
        tolerance_deg: float = 1.0,
    ) -> float:
        """移动到持久机械坐标中的绝对角度，始终选择最短路径。

        Args:
            angle_deg: 目标机械角度，单位°，范围 ``-360 <= angle <= 360``；
                负数和 360° 均先按一圈归一化，例如 -50° 等同 310°。
            rpm: 可选速度，单位 RPM；省略时沿用当前速度。
            wait: True 时等待反馈连续三次到位后返回。
            timeout: 到位超时秒数；省略时根据角度和速度自动估算。
            tolerance_deg: 到位误差，单位°，默认1°。

        Returns:
            下发给驱动板的内部多圈目标角度，单位°。
        """
        target_mechanical = self._validate_public_angle(angle_deg)
        with self._lock:
            self._require_enabled()
            try:
                if rpm is not None:
                    self.set_speed(rpm)
                current_total = self.get_total_angle()
                current_mechanical = self.get_angle()
                delta = shortest_delta_deg(
                    current_mechanical, target_mechanical
                )
                # 本机实测机械角度与协议多圈角度反向：
                # 多圈-59.8°时，机械角度同步+60.1°。
                target_total = round(
                    current_total + self.mechanical_direction * delta,
                    1,
                )
                self._send(
                    build_multi_turn_angle_frame(target_total, self.address)
                )
                if wait:
                    actual_timeout = (
                        self._estimate_timeout(abs(delta))
                        if timeout is None
                        else self._validate_timeout(timeout)
                    )
                    self._wait_for_total_target(
                        target_total,
                        actual_timeout,
                        tolerance_deg,
                    )
                    self._verify_mechanical_target(
                        target_mechanical, tolerance_deg
                    )
                return target_total
            except Exception:
                self._safe_disable()
                raise

    def return_to_mechanical_zero(
        self,
        rpm: int | None = None,
        wait: bool = True,
        timeout: float | None = None,
    ) -> float:
        """沿最短路径回到驱动板内持久保存的机械零点。"""
        return self.move_to(
            0.0,
            rpm=rpm,
            wait=wait,
            timeout=timeout,
        )

    def rotate_relative(
        self,
        delta_deg: float,
        rpm: int | None = None,
        wait: bool = True,
        timeout: float | None = None,
        tolerance_deg: float = 1.0,
    ) -> float:
        """按指定正负角度相对旋转，范围为 ``[-180, 180]`` 度。

        该接口会保留 ``+180`` 和 ``-180`` 的方向差异，适合算法直接表达
        “正/负方向转多少度”。实际视觉上的顺/逆时针由电机安装决定。
        """
        delta = float(delta_deg)
        if not math.isfinite(delta) or not -180.0 <= delta <= 180.0:
            raise ValueError("相对角度必须在 -180～180°")
        with self._lock:
            self._require_enabled()
            try:
                if rpm is not None:
                    self.set_speed(rpm)
                current_total = self.get_total_angle()
                current_mechanical = self.get_angle()
                target_mechanical = (current_mechanical + delta) % 360.0
                target_total = round(
                    current_total + self.mechanical_direction * delta,
                    1,
                )
                self._send(build_multi_turn_angle_frame(target_total, self.address))
                if wait:
                    actual_timeout = (
                        self._estimate_timeout(abs(delta))
                        if timeout is None
                        else self._validate_timeout(timeout)
                    )
                    self._wait_for_total_target(target_total, actual_timeout, tolerance_deg)
                    self._verify_mechanical_target(target_mechanical, tolerance_deg)
                return target_total
            except Exception:
                self._safe_disable()
                raise

    def rotate_one_turn(
        self,
        direction: int = 1,
        rpm: int | None = None,
        wait: bool = True,
        timeout: float | None = None,
        tolerance_deg: float = 1.0,
    ) -> float:
        """从当前位置明确完整旋转一圈。

        Args:
            direction: ``1``=协议正方向，``-1``=协议负方向。
                实际顺/逆时针取决于电机安装方向。
        """
        if int(direction) not in (-1, 1) or int(direction) != direction:
            raise ValueError("direction 只能是 1 或 -1")
        with self._lock:
            self._require_enabled()
            try:
                if rpm is not None:
                    self.set_speed(rpm)
                current_total = self.get_total_angle()
                current_mechanical = self.get_angle()
                target_total = round(
                    current_total + int(direction) * 360.0, 1
                )
                self._send(
                    build_multi_turn_angle_frame(target_total, self.address)
                )
                if wait:
                    actual_timeout = (
                        self._estimate_timeout(360.0)
                        if timeout is None
                        else self._validate_timeout(timeout)
                    )
                    self._wait_for_total_target(
                        target_total,
                        actual_timeout,
                        tolerance_deg,
                    )
                    self._verify_mechanical_target(
                        current_mechanical, tolerance_deg
                    )
                return target_total
            except Exception:
                self._safe_disable()
                raise

    def _query(self, feedback_type: int) -> FeedbackFrame:
        self._require_connected()
        for _attempt in range(1, self.feedback_attempts + 1):
            self._parser.clear()
            self._uart.flush_input()
            self._send(
                make_frame(
                    self.address,
                    FUNC_FEEDBACK_QUERY,
                    bytes((feedback_type,)),
                )
            )
            deadline = time.monotonic() + self.feedback_timeout
            while time.monotonic() < deadline:
                data = self._uart.read_some(
                    min(0.05, max(0.0, deadline - time.monotonic()))
                )
                for feedback in self._parser.feed(data):
                    if feedback.feedback_type == feedback_type:
                        return feedback
        name = (
            "总角度"
            if feedback_type == FEEDBACK_TOTAL_ANGLE
            else "机械角度"
        )
        raise F32CTimeoutError(
            f"{self.port} 连续{self.feedback_attempts}次未返回合法{name}反馈"
            f"（每次等待 {self.feedback_timeout:.3f}s）"
        )

    def _send(self, frame: bytes) -> None:
        self._uart.write_all(frame)
        time.sleep(self.command_gap)

    def _wait_for_total_target(
        self,
        target_total: float,
        timeout: float,
        tolerance_deg: float,
    ) -> None:
        tolerance = float(tolerance_deg)
        if not math.isfinite(tolerance) or not 0.1 <= tolerance <= 10.0:
            raise ValueError("到位容差必须在 0.1～10°")
        deadline = time.monotonic() + timeout
        stable_count = 0
        last_total = None
        try:
            while time.monotonic() < deadline:
                last_total = self.get_total_angle()
                if abs(last_total - target_total) <= tolerance:
                    stable_count += 1
                    if stable_count >= 3:
                        return
                else:
                    stable_count = 0
                time.sleep(0.02)
        except Exception:
            self._safe_disable()
            raise

        self._safe_disable()
        raise F32CTimeoutError(
            f"等待电机到位超时：目标={target_total:.1f}°，"
            f"最后反馈={last_total!r}°；电机已失能"
        )

    def _verify_mechanical_target(
        self,
        target_mechanical: float,
        tolerance_deg: float,
    ) -> None:
        actual = self.get_angle()
        error = abs(shortest_delta_deg(actual, target_mechanical))
        if error > float(tolerance_deg):
            raise F32CProtocolError(
                "多圈坐标已到位，但机械角度复核失败："
                f"目标={target_mechanical:.1f}°，反馈={actual:.1f}°，"
                f"误差={error:.1f}°"
            )

    def _safe_disable(self) -> None:
        try:
            if self.is_connected:
                self._send(make_frame(self.address, FUNC_DISABLE))
        except Exception:
            pass
        self._enabled = False

    def _estimate_timeout(self, distance_deg: float) -> float:
        # 1 RPM = 6°/s；额外留3秒用于T型加减速和反馈确认。
        return max(3.0, float(distance_deg) / (self._rpm * 6.0) + 3.0)

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise F32CConnectionError("串口未连接")

    def _require_enabled(self) -> None:
        self._require_connected()
        if not self._enabled:
            raise F32CError("电机未使能；请先调用 enable()")

    @staticmethod
    def _validate_public_angle(angle_deg: float) -> float:
        angle = float(angle_deg)
        if not math.isfinite(angle) or not -360.0 <= angle <= 360.0:
            raise ValueError("普通绝对目标必须满足 -360 <= angle_deg <= 360")
        return angle % 360.0

    def _validate_rpm(self, rpm: int) -> int:
        speed = int(rpm)
        if speed != rpm or not 1 <= speed <= self.max_rpm:
            raise ValueError(
                f"速度必须是 1～{self.max_rpm} 之间的整数 RPM；"
                "如已确认机械能力，可在构造函数中显式提高 max_rpm"
            )
        return speed

    @staticmethod
    def _validate_timeout(timeout: float) -> float:
        value = float(timeout)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("超时时间必须大于0秒")
        return value

    def __enter__(self) -> "F32CMotor":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(disable=True)


__all__ = [
    "__version__",
    "DEFAULT_ADDRESS",
    "DEFAULT_BAUDRATE",
    "DEFAULT_FEEDBACK_ATTEMPTS",
    "DEFAULT_MAX_RPM",
    "DEFAULT_MECHANICAL_DIRECTION",
    "DEFAULT_PORT",
    "DEFAULT_RPM",
    "F32CConnectionError",
    "F32CError",
    "F32CMotor",
    "F32CProtocolError",
    "F32CTimeoutError",
    "FeedbackFrame",
    "FeedbackStreamParser",
    "build_multi_turn_angle_frame",
    "build_speed_frame",
    "calculate_bcc",
    "decode_feedback_frame",
    "make_frame",
    "shortest_delta_deg",
]
