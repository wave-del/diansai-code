"""算法侧唯一的拼图硬件 API：不需要接触 G-code、UART 帧或 GPIO。"""
from __future__ import annotations

import math
from pathlib import Path
import sys

HANDOVER = Path(__file__).resolve().parents[1] / "handover-puzzle"
if str(HANDOVER) not in sys.path:
    sys.path.insert(0, str(HANDOVER))

from electromagnet import ElectromagnetController
from f32c_controller import F32CMotor
from writer_controller import WriterController


class TangramAPI:
    """蓝纸坐标、正数抬高 Z、FOC 角度和电磁铁的高层同步接口。

    所有 XY 参数均是蓝纸右上角坐标：X+ 向下、Y+ 向左，单位 mm。底层的
    ``X-23.5/Y+16.45`` 校正和 ``X=0..190/Y=0..300`` 限幅已封装，调用方
    不得自行再做偏置。Z 参数均为正数抬起高度；0 表示自然下垂。
    """

    Z_FEED_MM_MIN = 1500.0
    MAX_LIFT_MM = 15.0

    def __init__(self, writer_port: str = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0", motor_port: str = "/dev/ttyAS3", motor_address: int = 2) -> None:
        self.writer_port, self.motor_port, self.motor_address = writer_port, motor_port, motor_address
        self.writer = WriterController()
        self.motor = F32CMotor(port=motor_port, address=motor_address)
        self.magnet = ElectromagnetController()
        self._writer_open = self._motor_open = self._magnet_open = self._origin_ready = False

    def connect(self, motor_rpm: int = 15) -> None:
        """连接并保持 FOC 使能；不移动任何轴，也不建立原点。"""
        try:
            self.writer.connect(self.writer_port)
            self._writer_open = True
            self.writer.set_motor_hold(True)
            self.motor.connect(verify=False)
            self._motor_open = True
            self.motor.enable(rpm=motor_rpm)
            self.magnet.open()
            self._magnet_open = True
        except BaseException:
            self.close()
            raise

    def establish_origin(self) -> None:
        """机构位于写字机自身原点后调用；不会主动运动。"""
        self._require_writer()
        self.writer.home()
        self.writer.set_motor_hold(True)
        self._origin_ready = True

    def move_xyz(self, x_mm: float, y_mm: float, lift_mm: float, *, xy_feed_mm_min: float = 2500.0, travel_lift_mm: float = 8.0) -> None:
        """先安全抬高、移动到蓝纸 XY、再到目标正数 Z 抬高高度。"""
        self._require_origin()
        self.writer.move_xyz(x_mm, y_mm, lift_mm, xy_feed=xy_feed_mm_min, travel_lift_mm=travel_lift_mm)

    def set_lift(self, lift_mm: float) -> None:
        """Z=正数向上抬起，0=自然下垂；固定 1500 mm/min，安全范围 0..15 mm。"""
        self._require_writer()
        self.writer.set_lift(lift_mm)

    def return_machine_origin(self, xy_feed_mm_min: float = 2500.0) -> None:
        """回写字机自身原点；不应用蓝纸坐标偏置。"""
        self._require_origin()
        self.writer.return_to_machine_origin(feedrate=xy_feed_mm_min)

    def foc_zero(self, rpm: int = 15) -> float:
        """沿最短路径回到驱动板保存的机械零点。"""
        self._require_motor()
        return self.motor.return_to_mechanical_zero(rpm=rpm, wait=True)

    def foc_angle(self, angle_deg: float, rpm: int = 15) -> float:
        """绝对定位到 -360..360°（按 360° 归一化），控制器自动选择最短路径。"""
        angle = float(angle_deg)
        if not math.isfinite(angle) or not -360.0 <= angle <= 360.0:
            raise ValueError("FOC 绝对角度必须在 -360～360°")
        self._require_motor()
        return self.motor.move_to(angle % 360.0, rpm=rpm, wait=True)

    def foc_rotate(self, delta_deg: float, rpm: int = 15) -> float:
        """按正/负方向相对旋转 -180..+180°，+180 与 -180 的方向不同。"""
        self._require_motor()
        return self.motor.rotate_relative(delta_deg, rpm=rpm, wait=True)

    def magnet_on(self) -> None:
        self._require_magnet(); self.magnet.engage()

    def magnet_off(self) -> None:
        self._require_magnet(); self.magnet.release()

    def close(self) -> None:
        """松开电磁铁、写字机继续保持；FOC 默认保持最后位置和使能。"""
        if self._magnet_open:
            try: self.magnet.close(release=True)
            finally: self._magnet_open = False
        if self._motor_open:
            try: self.motor.close(disable=False)
            finally: self._motor_open = False
        if self._writer_open:
            try: self.writer.set_motor_hold(True)
            finally:
                self.writer.disconnect(); self._writer_open = False

    def _require_writer(self) -> None:
        if not self._writer_open: raise RuntimeError("请先 connect()")
    def _require_origin(self) -> None:
        self._require_writer()
        if not self._origin_ready: raise RuntimeError("请先让机构回写字机自身原点并调用 establish_origin()")
    def _require_motor(self) -> None:
        if not self._motor_open: raise RuntimeError("请先 connect()")
    def _require_magnet(self) -> None:
        if not self._magnet_open: raise RuntimeError("请先 connect()")

    def __enter__(self) -> "TangramAPI":
        self.connect(); return self
    def __exit__(self, *_: object) -> None:
        self.close()
