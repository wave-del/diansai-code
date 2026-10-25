"""Q6A 电磁铁控制：物理 13 脚，gpiochip4 line 0（PIN_13）。

高电平吸合、低电平释放。直接使用 Linux GPIO 字符设备 ABI v1；
导入模块及创建控制器均不占用 GPIO，open() 才访问设备。
"""

from __future__ import annotations

import errno
import fcntl
import math
import os
import struct
import threading
import time
from typing import Protocol


DEFAULT_GPIO_CHIP = "/dev/gpiochip4"
DEFAULT_LINE_OFFSET = 0
DEFAULT_LINE_NAME = "PIN_13"
DEFAULT_CONSUMER = "tangram-electromagnet"

GPIOHANDLE_REQUEST_OUTPUT = 1 << 1
GPIOHANDLE_REQUEST_BIAS_PULL_DOWN = 1 << 6

_GPIO_MAGIC = 0xB4
_IOC_READ_WRITE = 3
_GPIO_LINE_INFO_SIZE = 72
_GPIO_HANDLE_REQUEST_SIZE = 364
_GPIO_HANDLE_DATA_SIZE = 64


def _iowr(number: int, size: int) -> int:
    """按Linux通用ioctl布局生成GPIO ABI v1读写请求码。"""
    return (
        (_IOC_READ_WRITE << 30)
        | (int(size) << 16)
        | (_GPIO_MAGIC << 8)
        | int(number)
    )


GPIO_GET_LINEINFO_IOCTL = _iowr(0x02, _GPIO_LINE_INFO_SIZE)
GPIO_GET_LINEHANDLE_IOCTL = _iowr(
    0x03,
    _GPIO_HANDLE_REQUEST_SIZE,
)
GPIOHANDLE_GET_LINE_VALUES_IOCTL = _iowr(
    0x08,
    _GPIO_HANDLE_DATA_SIZE,
)
GPIOHANDLE_SET_LINE_VALUES_IOCTL = _iowr(
    0x09,
    _GPIO_HANDLE_DATA_SIZE,
)


class ElectromagnetError(Exception):
    """电磁铁控制错误基类。"""


class ElectromagnetConnectionError(ElectromagnetError):
    """GPIO设备、映射、权限或占用错误。"""


class ElectromagnetStateError(ElectromagnetError):
    """GPIO输出状态无法设置或复核。"""


class _DigitalOutput(Protocol):
    def open(self, initial_value: bool) -> None: ...

    def set_value(self, value: bool) -> None: ...

    def get_value(self) -> bool: ...

    def close(self) -> None: ...


class LinuxGpioOutput:
    """使用Linux GPIO字符设备ABI v1独占并控制一根输出线。"""

    def __init__(
        self,
        chip: str,
        line_offset: int,
        expected_line_name: str | None,
        consumer: str,
        request_pull_down: bool = True,
    ):
        if int(line_offset) != line_offset or int(line_offset) < 0:
            raise ValueError("GPIO line offset必须是大于等于0的整数")
        if not consumer:
            raise ValueError("GPIO consumer名称不能为空")

        self.chip = str(chip)
        self.line_offset = int(line_offset)
        self.expected_line_name = expected_line_name
        self.consumer = str(consumer)
        self.request_pull_down = bool(request_pull_down)
        self.line_name: str | None = None
        self.previous_consumer: str | None = None
        self.pull_down_applied = False
        self._fd: int | None = None

    @property
    def is_open(self) -> bool:
        return self._fd is not None

    def open(self, initial_value: bool) -> None:
        if self.is_open:
            raise ElectromagnetConnectionError("GPIO输出已经打开")

        try:
            chip_fd = os.open(
                self.chip,
                os.O_RDWR | os.O_CLOEXEC,
            )
        except OSError as exc:
            raise ElectromagnetConnectionError(
                f"无法打开GPIO设备 {self.chip}: {exc.strerror or exc}"
            ) from exc

        handle_fd: int | None = None
        try:
            self._read_and_verify_line_info(chip_fd)
            base_flags = GPIOHANDLE_REQUEST_OUTPUT
            flags = base_flags
            if self.request_pull_down:
                flags |= GPIOHANDLE_REQUEST_BIAS_PULL_DOWN

            try:
                handle_fd = self._request_line(
                    chip_fd,
                    flags,
                    bool(initial_value),
                )
                self.pull_down_applied = self.request_pull_down
            except OSError as exc:
                # 较旧内核可能支持GPIO句柄但不支持bias flag。仅对明确的
                # “不支持参数”回退；占用、权限等错误绝不绕过。
                if self.request_pull_down and exc.errno in (
                    errno.EINVAL,
                    errno.ENOTTY,
                    errno.EOPNOTSUPP,
                ):
                    handle_fd = self._request_line(
                        chip_fd,
                        base_flags,
                        bool(initial_value),
                    )
                    self.pull_down_applied = False
                else:
                    raise
        except ElectromagnetError:
            if handle_fd is not None:
                os.close(handle_fd)
            raise
        except OSError as exc:
            if handle_fd is not None:
                os.close(handle_fd)
            consumer = self.previous_consumer or "无"
            raise ElectromagnetConnectionError(
                f"无法申请 {self.chip} line {self.line_offset}"
                f"（{self.line_name or '未命名'}，当前使用者={consumer}）: "
                f"{exc.strerror or exc}"
            ) from exc
        finally:
            os.close(chip_fd)

        if handle_fd is None:
            raise ElectromagnetConnectionError("GPIO句柄申请没有返回文件描述符")
        os.set_inheritable(handle_fd, False)
        self._fd = handle_fd

        try:
            if self.get_value() != bool(initial_value):
                raise ElectromagnetStateError(
                    "GPIO初始输出状态复核失败"
                )
        except Exception:
            self.close()
            raise

    def set_value(self, value: bool) -> None:
        fd = self._require_fd()
        data = bytearray(_GPIO_HANDLE_DATA_SIZE)
        data[0] = 1 if bool(value) else 0
        try:
            fcntl.ioctl(
                fd,
                GPIOHANDLE_SET_LINE_VALUES_IOCTL,
                data,
                True,
            )
        except OSError as exc:
            raise ElectromagnetStateError(
                f"设置GPIO输出失败: {exc.strerror or exc}"
            ) from exc

    def get_value(self) -> bool:
        fd = self._require_fd()
        data = bytearray(_GPIO_HANDLE_DATA_SIZE)
        try:
            fcntl.ioctl(
                fd,
                GPIOHANDLE_GET_LINE_VALUES_IOCTL,
                data,
                True,
            )
        except OSError as exc:
            raise ElectromagnetStateError(
                f"读取GPIO输出失败: {exc.strerror or exc}"
            ) from exc
        return bool(data[0])

    def close(self) -> None:
        if self._fd is None:
            return
        fd = self._fd
        self._fd = None
        os.close(fd)

    def _read_and_verify_line_info(self, chip_fd: int) -> None:
        data = bytearray(_GPIO_LINE_INFO_SIZE)
        struct.pack_into("=I", data, 0, self.line_offset)
        try:
            fcntl.ioctl(
                chip_fd,
                GPIO_GET_LINEINFO_IOCTL,
                data,
                True,
            )
        except OSError as exc:
            raise ElectromagnetConnectionError(
                f"无法读取 {self.chip} line {self.line_offset} 信息: "
                f"{exc.strerror or exc}"
            ) from exc

        _, _, raw_name, raw_consumer = struct.unpack(
            "=II32s32s",
            data,
        )
        self.line_name = _decode_c_string(raw_name) or None
        self.previous_consumer = _decode_c_string(raw_consumer) or None

        if (
            self.expected_line_name
            and self.line_name
            and self.line_name != self.expected_line_name
        ):
            raise ElectromagnetConnectionError(
                f"GPIO映射不匹配：期望 {self.expected_line_name}，"
                f"实际为 {self.line_name}"
            )

    def _request_line(
        self,
        chip_fd: int,
        flags: int,
        initial_value: bool,
    ) -> int:
        request = bytearray(_GPIO_HANDLE_REQUEST_SIZE)
        struct.pack_into("=I", request, 0, self.line_offset)
        struct.pack_into("=I", request, 256, int(flags))
        request[260] = 1 if initial_value else 0
        label = self.consumer.encode("utf-8")[:31]
        request[324 : 324 + len(label)] = label
        struct.pack_into("=I", request, 356, 1)
        struct.pack_into("=i", request, 360, -1)
        fcntl.ioctl(
            chip_fd,
            GPIO_GET_LINEHANDLE_IOCTL,
            request,
            True,
        )
        return struct.unpack_from("=i", request, 360)[0]

    def _require_fd(self) -> int:
        if self._fd is None:
            raise ElectromagnetConnectionError("GPIO输出尚未打开")
        return self._fd


class ElectromagnetController:
    """高电平吸合、低电平松开的同步电磁铁控制器。"""

    def __init__(
        self,
        chip: str = DEFAULT_GPIO_CHIP,
        line_offset: int = DEFAULT_LINE_OFFSET,
        active_high: bool = True,
        expected_line_name: str | None = DEFAULT_LINE_NAME,
        consumer: str = DEFAULT_CONSUMER,
        *,
        _backend: _DigitalOutput | None = None,
    ):
        if int(line_offset) != line_offset or int(line_offset) < 0:
            raise ValueError("GPIO line offset必须是大于等于0的整数")
        self.chip = str(chip)
        self.line_offset = int(line_offset)
        self.active_high = bool(active_high)
        self.expected_line_name = expected_line_name
        self.consumer = str(consumer)
        self._backend = _backend or LinuxGpioOutput(
            chip=self.chip,
            line_offset=self.line_offset,
            expected_line_name=self.expected_line_name,
            consumer=self.consumer,
        )
        self._lock = threading.RLock()
        self._is_open = False
        self._engaged = False

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def is_engaged(self) -> bool:
        """读取GPIO并返回当前逻辑状态；True表示吸合。"""
        with self._lock:
            self._require_open()
            physical = self._backend.get_value()
            self._engaged = (
                physical if self.active_high else not physical
            )
            return self._engaged

    def open(self) -> None:
        """独占GPIO_0并原子地初始化为松开状态（默认物理低电平）。"""
        with self._lock:
            if self._is_open:
                raise ElectromagnetConnectionError("电磁铁GPIO已经打开")
            released_level = self._physical_value(False)
            self._backend.open(initial_value=released_level)
            self._is_open = True
            self._engaged = False
            try:
                if self.is_engaged:
                    raise ElectromagnetStateError(
                        "GPIO打开后未处于松开状态"
                    )
            except Exception:
                self._backend.close()
                self._is_open = False
                raise

    def engage(self) -> None:
        """输出有效电平；默认高电平，使电磁铁吸合。"""
        self.set_engaged(True)

    def release(self) -> None:
        """输出无效电平；默认低电平，使电磁铁松开。"""
        self.set_engaged(False)

    def set_engaged(self, engaged: bool) -> None:
        """设置逻辑吸合状态并读取GPIO复核。"""
        if not isinstance(engaged, bool):
            raise ValueError("engaged必须是布尔值")
        with self._lock:
            self._require_open()
            expected_physical = self._physical_value(engaged)
            try:
                self._backend.set_value(expected_physical)
                actual_physical = self._backend.get_value()
            except Exception:
                self._engaged = False
                raise
            if actual_physical != expected_physical:
                self._engaged = False
                raise ElectromagnetStateError(
                    "GPIO输出写入后的状态复核失败"
                )
            self._engaged = engaged

    def pulse(self, duration_seconds: float) -> None:
        """吸合指定秒数，随后在finally中恢复松开。"""
        duration = float(duration_seconds)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("吸合时长必须是大于0的有限秒数")
        self.engage()
        try:
            time.sleep(duration)
        finally:
            self.release()

    def close(self, release: bool = True) -> None:
        """默认先输出松开电平，再释放GPIO句柄。"""
        with self._lock:
            if not self._is_open:
                return
            try:
                if release:
                    try:
                        self._backend.set_value(
                            self._physical_value(False)
                        )
                    except Exception:
                        # 仍必须关闭句柄，避免GPIO长期被故障进程占用。
                        pass
            finally:
                self._backend.close()
                self._is_open = False
                self._engaged = False

    def _physical_value(self, engaged: bool) -> bool:
        return bool(engaged) if self.active_high else not bool(engaged)

    def _require_open(self) -> None:
        if not self._is_open:
            raise ElectromagnetConnectionError(
                "电磁铁GPIO尚未打开；请先调用 open()"
            )

    def __enter__(self) -> "ElectromagnetController":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(release=True)


def _decode_c_string(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")


__all__ = [
    "DEFAULT_CONSUMER",
    "DEFAULT_GPIO_CHIP",
    "DEFAULT_LINE_NAME",
    "DEFAULT_LINE_OFFSET",
    "ElectromagnetConnectionError",
    "ElectromagnetController",
    "ElectromagnetError",
    "ElectromagnetStateError",
    "LinuxGpioOutput",
]
