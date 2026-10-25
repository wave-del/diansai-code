"""
写字机 GRBL 串口控制器（独立交付版）。

这台设备使用带第四轴的 GRBL 变体。状态中的 MPos 是机器坐标，
而 G0/G1 使用的是工作坐标；控制器会结合 WCO/WPos 汇报真实工作坐标。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QObject, QIODevice, QTimer, Signal
from PySide6.QtSerialPort import QSerialPort, QSerialPortInfo


class GCode:
    HOME = "$H"
    UNLOCK = "$X"
    SET_ZERO = "G92 X0 Y0"
    QUERY_STATUS = "?"
    SETTINGS = "$$"
    RELATIVE = "G91"
    ABSOLUTE = "G90"


def _parse_coordinates(value: str) -> tuple[float, ...] | None:
    try:
        return tuple(float(item) for item in value.split(","))
    except (TypeError, ValueError):
        return None


@dataclass
class GrblStatus:
    """解析标准 GRBL 以及本机四轴状态行。"""

    raw: str
    state: str = ""
    mpos: tuple[float, ...] | None = None
    wpos: tuple[float, ...] | None = None
    wco: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        text = self.raw.strip()
        if not (text.startswith("<") and text.endswith(">")):
            return

        fields = text[1:-1].split("|")
        if not fields:
            return

        # Hold:0 / Door:1 等状态的主状态在冒号前。
        self.state = fields[0].split(":", 1)[0]
        for field in fields[1:]:
            if ":" not in field:
                continue
            name, value = field.split(":", 1)
            if name == "MPos":
                self.mpos = _parse_coordinates(value)
            elif name == "WPos":
                self.wpos = _parse_coordinates(value)
            elif name == "WCO":
                self.wco = _parse_coordinates(value)

    @property
    def mpos_x(self) -> float:
        return self.mpos[0] if self.mpos else 0.0

    @property
    def mpos_y(self) -> float:
        return self.mpos[1] if self.mpos and len(self.mpos) > 1 else 0.0

    @property
    def mpos_z(self) -> float:
        return self.mpos[2] if self.mpos and len(self.mpos) > 2 else 0.0

    @property
    def work_position(self) -> tuple[float, float] | None:
        if self.wpos and len(self.wpos) >= 2:
            return self.wpos[0], self.wpos[1]
        if self.mpos and self.wco and len(self.mpos) >= 2 and len(self.wco) >= 2:
            return self.mpos[0] - self.wco[0], self.mpos[1] - self.wco[1]
        return None

    @property
    def work_x(self) -> Optional[float]:
        position = self.work_position
        return position[0] if position else None

    @property
    def work_y(self) -> Optional[float]:
        position = self.work_position
        return position[1] if position else None

    @property
    def is_idle(self) -> bool:
        return self.state == "Idle"

    @property
    def is_alarm(self) -> bool:
        return self.state == "Alarm"


def build_absolute_move_command(
    x: float,
    y: float,
    feedrate: float = 2500,
    rapid: bool = False,
) -> str:
    motion = "G0" if rapid else "G1"
    feed = "" if rapid else f" F{feedrate:.0f}"
    return f"G21 G90 {motion} X{x:.3f} Y{y:.3f}{feed}"


def build_pen_z_command(z: float, feedrate: float = 3000) -> str:
    """生成步进笔轴的绝对 Z 移动命令。"""
    return f"G21 G90 G1 Z{z:.3f} F{feedrate:.0f}"


def build_manual_origin_command() -> str:
    """人工把机构放到固定基准后，将当前 X/Y/Z 定义为本次上电原点。"""
    return "G92 X0 Y0 Z0"


def apply_transform(
    x: float,
    y: float,
    swap_xy: bool = False,
    reverse_x: bool = False,
    reverse_y: bool = False,
) -> tuple[float, float]:
    if swap_xy:
        x, y = y, x
    if reverse_x:
        x = -x
    if reverse_y:
        y = -y
    return x, y


class GrblController(QObject):
    position_changed = Signal(float, float)
    pen_position_changed = Signal(float)
    position_valid_changed = Signal(bool)
    machine_state = Signal(str)
    response_line = Signal(str)
    command_completed = Signal(int, bool, str)
    status_updated = Signal(int, str)
    connected = Signal()
    disconnected = Signal()
    error_occurred = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Qt 原生异步串口：不需要阻塞读循环，也不会跨线程直接调用 QObject。
        self._serial = QSerialPort(self)
        self._serial.readyRead.connect(self._on_ready_read)
        self._serial.errorOccurred.connect(self._on_serial_error)
        self._rx_buffer = bytearray()

        self._connected = False
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._machine_x = 0.0
        self._machine_y = 0.0
        self._machine_z = 0.0
        self._pos_z = 0.0
        self._wco: tuple[float, ...] | None = None
        self._position_valid = False
        self._current_state = ""
        self._grbl_report_count = 0
        self._command_sequence = 0
        self._acknowledged_sequence = 0
        self._pending_command_ids: deque[int] = deque()
        self._command_results: dict[int, tuple[bool, str]] = {}
        self._recent_responses: deque[str] = deque(maxlen=200)
        self._startup_banner_seen = False
        self._startup_status_seen = False

        # 这台实机的电机轴与用户坐标定义相反，默认对调 X/Y。
        # reverse_x=True / reverse_y=True：以右下角为原点时，
        # X+ 往左、Y+ 往上进入工作区域。
        self.swap_xy = True
        self.reverse_x = True
        self.reverse_y = True

        # 官方 default.properties 的 GRBL 步进笔控参数。
        self.pen_up_z = 0.0
        self.pen_down_z = 2.0
        self.pen_z_speed = 3000.0
        self.pen_delay = 0.3

    @property
    def is_connected(self) -> bool:
        return self._connected and self._serial.isOpen()

    @property
    def position(self) -> tuple[float, float]:
        return self._pos_x, self._pos_y

    @property
    def position_valid(self) -> bool:
        return self._position_valid

    @property
    def pen_position(self) -> float:
        return self._pos_z

    @property
    def state(self) -> str:
        return self._current_state

    @property
    def startup_ready(self) -> bool:
        """ESP32 启动横幅完成且已收到至少一条有效状态。"""
        return self._startup_banner_seen and self._startup_status_seen

    @property
    def command_sequence(self) -> int:
        """本次连接已成功排入串口发送缓冲区的普通命令序号。"""
        return self._command_sequence

    @property
    def acknowledged_sequence(self) -> int:
        """本次连接已收到 ok/error/ALARM 终结应答的最大命令序号。"""
        return self._acknowledged_sequence

    @property
    def pending_command_count(self) -> int:
        return len(self._pending_command_ids)

    @property
    def status_sequence(self) -> int:
        return self._grbl_report_count

    @property
    def bytes_to_write(self) -> int:
        return self._serial.bytesToWrite() if self._serial.isOpen() else 0

    def command_result(self, sequence: int) -> tuple[bool, str] | None:
        return self._command_results.get(sequence)

    @property
    def recent_responses(self) -> tuple[str, ...]:
        return tuple(self._recent_responses)

    @property
    def debug_info(self) -> str:
        valid = "有效" if self._position_valid else "等待 WCO/WPos"
        return (
            f"工作坐标: ({self._pos_x:.3f}, {self._pos_y:.3f}) [{valid}] | "
            f"笔轴 Z: {self._pos_z:.3f} | "
            f"机器坐标: ({self._machine_x:.3f}, {self._machine_y:.3f}, "
            f"{self._machine_z:.3f}) | 状态汇报: {self._grbl_report_count} 次"
        )

    def open_port(self, port: str, baud: int = 115200):
        if self.is_connected:
            self.error_occurred.emit("已经连接，请先断开")
            return

        self._serial.setPortName(port)
        self._serial.setBaudRate(int(baud))
        self._serial.setDataBits(QSerialPort.Data8)
        self._serial.setParity(QSerialPort.NoParity)
        self._serial.setStopBits(QSerialPort.OneStop)
        self._serial.setFlowControl(QSerialPort.NoFlowControl)
        self._rx_buffer.clear()

        if not self._serial.open(QIODevice.ReadWrite):
            self.error_occurred.emit(
                f"连接 {port} 失败: {self._serial.errorString()}"
            )
            return

        self._connected = True
        self._current_state = ""
        self._startup_banner_seen = False
        self._startup_status_seen = False
        self._grbl_report_count = 0
        self._command_sequence = 0
        self._acknowledged_sequence = 0
        self._pending_command_ids.clear()
        self._command_results.clear()
        self._recent_responses.clear()
        self._position_valid = False
        self.position_valid_changed.emit(False)
        self.connected.emit()

        # 实机是 ESP32 GRBL。CH340 串口打开后必须产生一次 DTR 脉冲，
        # 否则固件可能能打开端口却不返回任何数据。
        self._serial.setDataTerminalReady(False)
        QTimer.singleShot(100, lambda: self._serial.setDataTerminalReady(True))
        # 此固件每隔若干次状态才附带 WCO，因此启动后连查以尽快拿到
        # 可用于绝对移动的真实工作坐标。
        for delay_ms in range(1300, 2501, 120):
            QTimer.singleShot(delay_ms, self.query_status)

    def close_port(self):
        was_connected = self._connected
        self._connected = False
        if self._serial.isOpen():
            self._serial.close()
        self._rx_buffer.clear()
        self._pending_command_ids.clear()
        self._startup_banner_seen = False
        self._startup_status_seen = False
        if was_connected:
            self.disconnected.emit()

    def send_gcode(self, cmd: str) -> bool:
        if not self.is_connected:
            self.error_occurred.emit("串口未连接")
            return False

        cmd = cmd.strip()
        if not cmd:
            return False
        if cmd == GCode.QUERY_STATUS:
            return self._write_bytes(b"?")

        ok = True
        for line in cmd.splitlines():
            line = line.strip()
            if line:
                payload = (line + "\n").encode("ascii", errors="strict")
                queued = self._write_bytes(payload)
                if queued:
                    self._command_sequence += 1
                    self._pending_command_ids.append(self._command_sequence)
                ok = queued and ok
        return ok

    def _write_bytes(self, payload: bytes) -> bool:
        written = self._serial.write(payload)
        if written != len(payload):
            self.error_occurred.emit(f"发送失败: {self._serial.errorString()}")
            return False
        return True

    def _finish_command(self, success: bool, response: str) -> None:
        if not self._pending_command_ids:
            # 启动阶段或外部残留回复，不错误地归属到新命令。
            return
        sequence = self._pending_command_ids.popleft()
        self._acknowledged_sequence = max(self._acknowledged_sequence, sequence)
        self._command_results[sequence] = (success, response)
        # 控制内存占用；同步封装只需要近期结果。
        if len(self._command_results) > 1000:
            oldest = min(self._command_results)
            del self._command_results[oldest]
        self.command_completed.emit(sequence, success, response)

    def _transform(self, x: float, y: float) -> tuple[float, float]:
        return apply_transform(
            x, y, self.swap_xy, self.reverse_x, self.reverse_y
        )

    def _untransform(self, x: float, y: float) -> tuple[float, float]:
        if self.reverse_y:
            y = -y
        if self.reverse_x:
            x = -x
        if self.swap_xy:
            x, y = y, x
        return x, y

    def move_to(self, x: float, y: float, feedrate: float = 2500):
        """移动到工作坐标系中的任意绝对坐标。"""
        tx, ty = self._transform(x, y)
        self.send_gcode(build_absolute_move_command(tx, ty, feedrate))
        self._request_position_updates()

    def fast_move_to(self, x: float, y: float):
        tx, ty = self._transform(x, y)
        self.send_gcode(build_absolute_move_command(tx, ty, rapid=True))
        self._request_position_updates()

    def jog(self, dx: float, dy: float, feedrate: float = 2500):
        tx, ty = self._transform(dx, dy)
        # 一行内显式指定 G91，随后恢复 G90，避免依赖之前的模态状态。
        self.send_gcode(
            f"G21 G91 G1 X{tx:.3f} Y{ty:.3f} F{feedrate:.0f}\nG90"
        )
        self._request_position_updates()

    def _request_position_updates(self):
        QTimer.singleShot(100, self.query_status)
        QTimer.singleShot(350, self.query_status)

    def pen_up(self, z: float | None = None) -> float:
        target = self.pen_up_z if z is None else float(z)
        self.send_gcode(build_pen_z_command(target, self.pen_z_speed))
        self._request_position_updates()
        return target

    def pen_down(self, z: float | None = None) -> float:
        target = self.pen_down_z if z is None else float(z)
        self.send_gcode(build_pen_z_command(target, self.pen_z_speed))
        self._request_position_updates()
        return target

    def jog_pen(self, dz: float, feedrate: float = 300):
        """小幅相对移动笔轴，用于寻找合适的抬笔高度。"""
        self.send_gcode(
            f"G21 G91 G1 Z{float(dz):.3f} F{float(feedrate):.0f}\nG90"
        )
        self._request_position_updates()

    def set_pen_up_origin(self):
        """把当前机械笔高定义为工作坐标 Z0（官方默认抬笔位置）。"""
        self.send_gcode("G92 Z0")
        self._wco = None
        self._pos_z = 0.0
        self.pen_position_changed.emit(0.0)
        for delay_ms in range(100, 1301, 120):
            QTimer.singleShot(delay_ms, self.query_status)

    def set_origin(self):
        self.send_gcode(GCode.SET_ZERO)
        # G92 会改变 WCO，旧缓存不能再用于换算后续 MPos。
        self._wco = None
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._position_valid = True
        self.position_valid_changed.emit(True)
        self._emit_position()
        for delay_ms in range(100, 1301, 120):
            QTimer.singleShot(delay_ms, self.query_status)

    def set_manual_origin(self):
        """确认人工归位后，为本次上电同时建立 X/Y/Z 工作原点。"""
        self.send_gcode(build_manual_origin_command())
        self._wco = None
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._pos_z = 0.0
        self._position_valid = True
        self.position_valid_changed.emit(True)
        self._emit_position()
        self.pen_position_changed.emit(0.0)
        for delay_ms in range(100, 1301, 120):
            QTimer.singleShot(delay_ms, self.query_status)

    def return_to_manual_origin(self):
        """先抬笔，再返回已经建立的人工 XY 原点；不执行无传感器的 $H。"""
        self.pen_up()
        self.move_to(0.0, 0.0)

    def home(self):
        self.send_gcode(GCode.HOME)
        self._position_valid = False
        self.position_valid_changed.emit(False)
        QTimer.singleShot(500, self.query_status)
        QTimer.singleShot(1500, self.query_status)

    def unlock(self):
        self.send_gcode(GCode.UNLOCK)

    def query_status(self):
        if self.is_connected:
            self.send_gcode(GCode.QUERY_STATUS)

    def get_settings(self):
        self.send_gcode(GCode.SETTINGS)

    def _on_ready_read(self):
        self._rx_buffer.extend(bytes(self._serial.readAll()))
        while b"\n" in self._rx_buffer:
            raw_line, _, remainder = self._rx_buffer.partition(b"\n")
            self._rx_buffer = bytearray(remainder)
            line = raw_line.rstrip(b"\r").decode("utf-8", errors="replace").strip()
            if line:
                self._on_data(line)

    def _on_serial_error(self, error):
        if error == QSerialPort.NoError:
            return
        # ResourceError/DeviceNotFoundError 表示连接已经不可用。
        self.error_occurred.emit(f"串口错误: {self._serial.errorString()}")
        if error in (QSerialPort.ResourceError, QSerialPort.DeviceNotFoundError):
            self.close_port()

    def _on_data(self, line: str):
        self._recent_responses.append(line)
        self.response_line.emit(line)

        if line.startswith("Grbl "):
            self._startup_banner_seen = True
            return
        if line == "ok":
            self._finish_command(True, line)
            return
        if line.startswith("error:"):
            self._finish_command(False, line)
            self.error_occurred.emit(f"GRBL 错误: {line}")
            return
        if line.startswith("ALARM:"):
            self._finish_command(False, line)
            self._current_state = "Alarm"
            self.error_occurred.emit(f"GRBL 报警: {line}")
            self.machine_state.emit("Alarm")
            return

        if not line.startswith("<"):
            return

        status = GrblStatus(line)
        if not status.state:
            return

        self._current_state = status.state
        self._startup_status_seen = True
        self.machine_state.emit(status.state)
        self._grbl_report_count += 1
        self.status_updated.emit(self._grbl_report_count, status.raw)

        if status.mpos and len(status.mpos) >= 2:
            self._machine_x = status.mpos[0]
            self._machine_y = status.mpos[1]
            if len(status.mpos) >= 3:
                self._machine_z = status.mpos[2]
        if status.wco:
            self._wco = status.wco

        work_position = status.work_position
        if work_position is None and status.mpos and self._wco:
            if len(status.mpos) >= 2 and len(self._wco) >= 2:
                work_position = (
                    status.mpos[0] - self._wco[0],
                    status.mpos[1] - self._wco[1],
                )

        if work_position is not None:
            self._pos_x, self._pos_y = self._untransform(*work_position)
            if not self._position_valid:
                self._position_valid = True
                self.position_valid_changed.emit(True)
            self._emit_position()

        work_z: float | None = None
        if status.wpos and len(status.wpos) >= 3:
            work_z = status.wpos[2]
        elif status.mpos and self._wco:
            if len(status.mpos) >= 3 and len(self._wco) >= 3:
                work_z = status.mpos[2] - self._wco[2]
        if work_z is not None:
            self._pos_z = work_z
            self.pen_position_changed.emit(work_z)

    def _emit_position(self):
        self.position_changed.emit(self._pos_x, self._pos_y)


def list_serial_ports() -> list[dict]:
    ports = []
    for port in QSerialPortInfo.availablePorts():
        description = port.description() or port.portName()
        ports.append(
            {
                "name": port.portName(),
                "desc": description,
                "hwid": f"VID:{port.vendorIdentifier():04X} "
                f"PID:{port.productIdentifier():04X}",
            }
        )
    return ports
