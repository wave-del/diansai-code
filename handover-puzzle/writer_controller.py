"""
写字机控制接口
==============

给视觉同事使用的上层接口。封装了 GrblController，默认使用标定好的参数。

用法:
    from writer_controller import WriterController

    ctrl = WriterController()
    ctrl.connect(port="/dev/ttyUSB0")

    # 每次上电需要人工把电磁铁圆心放到蓝纸右上角后建立原点
    ctrl.home()

    # 移动到蓝纸坐标（X+ 往下, Y+ 往左，源自右上角原点）
    ctrl.move_to(x=100.0, y=50.0)

    # 笔控制
    ctrl.pen_up()
    ctrl.pen_down()

    # 查状态
    pos = ctrl.position        # (x, y) 工作坐标
    z   = ctrl.pen_position    # Z 轴位置

    # 断开
    ctrl.disconnect()
"""

from __future__ import annotations

import time
import sys
import math
from collections import deque
from typing import Iterable, Optional

from PySide6.QtCore import QCoreApplication
try:
    from .grbl_controller import (
        GrblController,
        build_absolute_move_command,
        build_pen_z_command,
        list_serial_ports,
    )
except ImportError:
    from grbl_controller import (
        GrblController,
        build_absolute_move_command,
        build_pen_z_command,
        list_serial_ports,
    )


class WriterController:
    """写字机高层控制接口。

    对外坐标体系说明（以蓝纸标定结果为准）:
        - 原点: 蓝纸右上角（通过 G92 X0 Y0 Z0 建立）
        - X+ 方向: 向下
        - Y+ 方向: 向左
        - 单位: 毫米 (mm)

    每次上电/重连后必须先让机构位于写字机自身机械原点，再调用
    :meth:`home` 建立原点。所有绝对 XY 移动都会自动使用本机实测的
    ``X-23.5``、``Y+16.45`` 校正并限幅到 ``X=0..190``、``Y=0..300``。
    调用方不得自行重复偏置。
    """

    A4_X_OFFSET_MM = -23.5
    A4_Y_OFFSET_MM = 16.45
    A4_WRITER_X_MIN_MM = 0.0
    A4_WRITER_X_MAX_MM = 190.0
    A4_WRITER_Y_MIN_MM = 0.0
    A4_WRITER_Y_MAX_MM = 300.0
    Z_LIFT_FEED_MM_MIN = 1500.0
    MAX_SAFE_LIFT_MM = 15.0

    def __init__(self):
        self._cnc = GrblController()
        self._cnc.swap_xy = True
        self._cnc.reverse_x = True
        self._cnc.reverse_y = True
        self._app: Optional[QCoreApplication] = None
        self._queued_lines: deque[str] = deque()
        self._stream_inflight: int | None = None
        self._stream_error: str | None = None

    # ── 连接管理 ────────────────────────────────────────────

    def connect(self, port: str, baud: int = 115200) -> None:
        """连接到写字机串口。

        Args:
            port: 串口设备名或路径，例如 ``"/dev/ttyUSB0"`` 或
                ``"COM19"``。
            baud: 串口波特率，单位 bit/s。本机固件固定使用 115200。

        Returns:
            无。完整 ESP32 启动握手及首次状态读取成功后才返回。

        Raises:
            TimeoutError: 规定时间内没有收到固件启动横幅或状态。
            RuntimeError: 串口打不开、断线或初始化命令被拒绝。
        """
        self._ensure_app()
        self._cnc.open_port(port, baud)
        deadline = time.monotonic() + 8.0
        next_query = time.monotonic() + 1.2
        while time.monotonic() < deadline:
            self._pump_once()
            if self._cnc.startup_ready:
                # 只有收到首个有效状态后才发普通 G-code；ESP32 启动期间
                # 发送的行可能被静默丢弃。
                self._execute_raw("G21 G90", wait_motion=False)
                return
            if time.monotonic() >= next_query:
                self._cnc.query_status()
                next_query = time.monotonic() + 0.1
        self._cnc.close_port()
        raise TimeoutError(f"连接 {port} 后 8 秒内未收到 GRBL 状态")

    def disconnect(self) -> None:
        """关闭串口连接；无返回值。

        注意：关闭串口不是急停，已经进入 GRBL planner 的运动可能继续。
        """
        self._queued_lines.clear()
        self._stream_inflight = None
        self._stream_error = None
        self._cnc.close_port()
        self._pump(2)

    def is_connected(self) -> bool:
        """返回串口是否已打开且可用。"""
        return self._cnc.is_connected

    # ── 归零 ────────────────────────────────────────────────

    def home(self) -> None:
        """人工归零。

        请确保机构已位于写字机自身机械原点，Z 已处于安全位置。
        发送 G92 X0 Y0 Z0，将当前位置设为工作坐标原点。

        注意:
            每次上电/控制器复位后，先让机构回到写字机自身机械原点，再
            调用本方法。本方法发送 ``$X`` 和
            ``$1=255`` 和 ``G92 X0 Y0 Z0``，不会主动移动任何轴，也不是
            ``$H``。``$1=255`` 保证后续运动结束后步进电机继续使能，
            机构受到轻微扰动或手推时保持位置。
        """
        self._ensure_no_stream()
        self._execute_raw("$X", wait_motion=False)
        self._execute_raw("$1=255", wait_motion=False)
        self._execute_action(self._cnc.set_manual_origin)

    def reanchor_at_a4_position(self, x: float, y: float, z: float) -> None:
        """将**当前位置**重新定义为已知的蓝纸坐标，不移动任何轴。

        写字机串口打开时会产生 DTR 脉冲，实机 ESP32 GRBL 会重启，因而
        会丢失前一进程的 G92 工作坐标。连续任务在上一轮已停到安全停靠点
        时，调用本方法把当前位置重新锚定为该安全点，而不是错误地再定义为
        机械零点。``x/y`` 使用蓝纸坐标，``z`` 使用本机工作 Z 坐标。
        """
        if not self._cnc.is_connected:
            raise RuntimeError("串口未连接")
        if not math.isfinite(float(z)) or not -self.MAX_SAFE_LIFT_MM <= float(z) <= 0.0:
            raise ValueError("重锚 Z 必须位于本机已验证的 [-15, 0] mm")
        machine_x, machine_y = self.map_a4_xy(x, y)
        grbl_x, grbl_y = self._cnc._transform(machine_x, machine_y)
        self._ensure_no_stream()
        self._execute_raw("$X", wait_motion=False)
        self._execute_raw("$1=255", wait_motion=False)
        self._execute_raw(
            f"G92 X{grbl_x:.3f} Y{grbl_y:.3f} Z{float(z):.3f}",
            wait_motion=False,
        )

    # ── 运动控制 ────────────────────────────────────────────

    def move_to(self, x: float, y: float, feedrate: float = 2500) -> None:
        """移动到蓝纸右上角坐标系中的指定绝对坐标。

        Args:
            x: 蓝纸绝对 X 坐标，单位 mm；X+ 从右上角向下。
            y: 蓝纸绝对 Y 坐标，单位 mm；Y+ 从右上角向左。
            feedrate: XY 轨迹进给速度，单位 mm/min，默认 2500。

        Returns:
            无。目标命令收到 ``ok`` 且机器真实回到 ``Idle`` 后才返回。
        """
        if not self._cnc.is_connected:
            raise RuntimeError("串口未连接")
        machine_x, machine_y = self.map_a4_xy(x, y)
        self._execute_action(
            lambda: self._cnc.move_to(machine_x, machine_y, feedrate)
        )

    @classmethod
    def map_a4_xy(cls, x: float, y: float) -> tuple[float, float]:
        """将蓝纸坐标转换为本机写字机命令坐标并执行硬限幅。

        这是本机视觉、算法和手动控制共用的唯一绝对 XY 映射。输入单位均为
        mm；转换为 ``X-23.5``、``Y+16.45`` 后，再限制到 ``X=0..190``、
        ``Y=0..300``。该函数不连接也不移动硬件，便于调用方预览日志。
        """
        input_x, input_y = float(x), float(y)
        if not math.isfinite(input_x) or not math.isfinite(input_y):
            raise ValueError("XY 坐标必须是有限数字")
        return (
            min(
                cls.A4_WRITER_X_MAX_MM,
                max(cls.A4_WRITER_X_MIN_MM, input_x + cls.A4_X_OFFSET_MM),
            ),
            min(
                cls.A4_WRITER_Y_MAX_MM,
                max(cls.A4_WRITER_Y_MIN_MM, input_y + cls.A4_Y_OFFSET_MM),
            ),
        )

    def jog(self, dx: float, dy: float, feedrate: float = 2500) -> None:
        """相对当前位置点动。

        Args:
            dx: 用户 X 方向相对位移，单位 mm；正数向左，负数向右。
            dy: 用户 Y 方向相对位移，单位 mm；正数向上，负数向下。
            feedrate: XY 轨迹进给速度，单位 mm/min。

        Returns:
            无。相对运动完成并回到 ``Idle`` 后才返回。
        """
        if not self._cnc.is_connected:
            raise RuntimeError("串口未连接")
        self._execute_action(lambda: self._cnc.jog(dx, dy, feedrate))

    def fast_move_to(self, x: float, y: float) -> None:
        """用 G0 快速移动到用户绝对坐标，完成后返回。

        Args:
            x: 蓝纸绝对 X 坐标，单位 mm。
            y: 蓝纸绝对 Y 坐标，单位 mm。

        G0 速度由 GRBL 固件参数决定，调用方不能通过本方法指定。
        """
        if not self._cnc.is_connected:
            raise RuntimeError("串口未连接")
        machine_x, machine_y = self.map_a4_xy(x, y)
        self._execute_action(lambda: self._cnc.fast_move_to(machine_x, machine_y))

    def return_to_machine_origin(self, feedrate: float = 2500.0) -> None:
        """回到写字机自身机械原点，刻意不应用蓝纸 XY 偏置。

        与 ``move_to(0, 0)`` 不同：后者表示蓝纸坐标 `(0,0)`，仍会应用
        ``X-23.5/Y+16.45``；本方法才是任务结束、异常恢复和上电归位后应使用的
        “机器回零”。
        """
        if not self._cnc.is_connected:
            raise RuntimeError("串口未连接")
        self._execute_action(lambda: self._cnc.move_to(0.0, 0.0, float(feedrate)))

    def move_path(
        self,
        points: Iterable[tuple[float, float]],
        feedrate: float = 2500,
    ) -> None:
        """连续移动经过一组用户绝对坐标点，并在整条路径完成后返回。

        每条 G-code 收到 ``ok`` 后立即发送下一条，不等待该段运动完成，
        因而会利用 GRBL planner buffer 保持路径连续；最后等待真实 Idle。

        Args:
            points: 按执行顺序排列的 ``(x, y)`` 用户绝对坐标；x/y 单位
                均为 mm。
            feedrate: 所有路径段的轨迹进给速度，单位 mm/min。

        Returns:
            无。所有点执行完且机器回到 ``Idle`` 后才返回。
        """
        self._ensure_no_stream()
        count = 0
        for x, y in points:
            machine_x, machine_y = self.map_a4_xy(x, y)
            tx, ty = self._cnc._transform(machine_x, machine_y)
            self.send_gcode_nowait(
                build_absolute_move_command(tx, ty, float(feedrate))
            )
            count += 1
        if count:
            self.wait_idle()

    # ── 笔控制 ──────────────────────────────────────────────

    @property
    def pen_up_height(self) -> float:
        """当前配置的抬笔绝对 Z 高度，单位 mm。"""
        return self._cnc.pen_up_z

    @property
    def pen_down_height(self) -> float:
        """当前配置的落笔绝对 Z 高度，单位 mm。"""
        return self._cnc.pen_down_z

    @property
    def pen_speed(self) -> float:
        """当前配置的笔轴进给速度，单位 mm/min。"""
        return self._cnc.pen_z_speed

    def configure_pen(
        self,
        up_z: float = 0.0,
        down_z: float = 2.0,
        speed: float = 3000.0,
    ) -> None:
        """设置本次进程使用的抬笔、落笔高度和 Z 轴速度。

        高度基于 ``home()`` 时建立的 Z0；此方法只更新上层参数，不写 EEPROM。

        Args:
            up_z: 抬笔绝对 Z 坐标，单位 mm。默认 0.0。
            down_z: 落笔绝对 Z 坐标，单位 mm。默认 2.0。
            speed: 笔轴进给速度，单位 mm/min。默认 3000。

        Returns:
            无。本方法只保存参数，不移动笔轴。
        """
        values = (float(up_z), float(down_z), float(speed))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("笔高度和速度必须是有限数字")
        if speed <= 0:
            raise ValueError("笔轴速度必须大于 0")
        self._cnc.pen_up_z = float(up_z)
        self._cnc.pen_down_z = float(down_z)
        self._cnc.pen_z_speed = float(speed)

    def pen_move_to(self, z: float, feedrate: float | None = None) -> None:
        """把笔轴移动到绝对工作高度。

        Args:
            z: 相对于 ``home()`` 所建立 Z0 的绝对位置，单位 mm。
            feedrate: 笔轴进给速度，单位 mm/min；省略时使用
                :attr:`pen_speed`。

        Returns:
            无。笔轴到位并回到 ``Idle`` 后才返回。
        """
        target = float(z)
        speed = self._cnc.pen_z_speed if feedrate is None else float(feedrate)
        if not math.isfinite(target) or not math.isfinite(speed) or speed <= 0:
            raise ValueError("Z 目标和速度必须是有效数字，且速度大于 0")
        self._execute_raw(build_pen_z_command(target, speed))

    def set_lift(self, lift_mm: float) -> None:
        """设置电磁铁相对自然下垂 Z0 的抬起高度。

        Args:
            lift_mm: 抬起高度，单位 mm；``0`` 为自然下垂 Z0，正数表示
                向上抬起。当前实机安全验证范围为 ``0..15 mm``。

        Z 轴已实测只有 ``1500 mm/min`` 可以稳定起动，因此该方法固定使用
        这一速度。底层 GRBL 的负 Z 才表示上抬；该符号转换完全封装在此处。
        """
        lift = float(lift_mm)
        if not math.isfinite(lift):
            raise ValueError("抬起高度必须是有限数字")
        bounded = min(self.MAX_SAFE_LIFT_MM, max(0.0, lift))
        if bounded != lift:
            print(
                f"⚠ Z 限幅：抬起高度 {lift:.3f} mm → {bounded:.3f} mm；继续执行。",
                flush=True,
            )
        lift = bounded
        self.pen_move_to(-lift, feedrate=self.Z_LIFT_FEED_MM_MIN)

    def move_xyz(
        self,
        x: float,
        y: float,
        lift_mm: float,
        *,
        xy_feed: float = 2500.0,
        travel_lift_mm: float = 8.0,
    ) -> None:
        """安全地移动到蓝纸 XY 并设置正数语义的 Z 抬起高度。

        先抬到 ``travel_lift_mm``，再移动 XY，最后到 ``lift_mm``。XY 速度可
        调；Z 速度固定为实测可靠的 1500 mm/min。
        """
        travel_lift = float(travel_lift_mm)
        if not math.isfinite(travel_lift):
            raise ValueError("搬运抬高必须是有限数字")
        bounded_travel_lift = min(self.MAX_SAFE_LIFT_MM, max(0.0, travel_lift))
        if bounded_travel_lift != travel_lift:
            print(
                f"⚠ Z 限幅：搬运抬高 {travel_lift:.3f} mm → "
                f"{bounded_travel_lift:.3f} mm；继续执行。",
                flush=True,
            )
        travel_lift = bounded_travel_lift
        self.set_lift(max(float(lift_mm), travel_lift))
        self.move_to(x, y, feedrate=xy_feed)
        self.set_lift(lift_mm)

    def pen_up(
        self,
        z: float | None = None,
        feedrate: float | None = None,
    ) -> None:
        """抬笔到配置高度。

        Args:
            z: 可选的本次绝对 Z 高度，单位 mm；省略时使用
                :attr:`pen_up_height`。
            feedrate: 可选的本次笔轴速度，单位 mm/min；省略时使用
                :attr:`pen_speed`。
        """
        self.pen_move_to(self._cnc.pen_up_z if z is None else z, feedrate)

    def pen_down(
        self,
        z: float | None = None,
        feedrate: float | None = None,
    ) -> None:
        """落笔到配置高度。

        Args:
            z: 可选的本次绝对 Z 高度，单位 mm；省略时使用
                :attr:`pen_down_height`。
            feedrate: 可选的本次笔轴速度，单位 mm/min；省略时使用
                :attr:`pen_speed`。
        """
        self.pen_move_to(self._cnc.pen_down_z if z is None else z, feedrate)

    def pen_jog(self, dz: float, feedrate: float = 300.0) -> None:
        """笔轴相对微调。

        Args:
            dz: 相对当前笔高的 Z 位移，单位 mm；正数向下（更接近纸面），
                负数向上。
            feedrate: 本次笔轴进给速度，单位 mm/min，默认 300。

        Example:
            ``pen_jog(dz=0.1, feedrate=200)`` 表示以 200mm/min 的速度
            向下移动 0.1mm。
        """
        self._execute_action(lambda: self._cnc.jog_pen(dz, feedrate))

    # ── 状态查询 ────────────────────────────────────────────

    @property
    def position(self) -> tuple[float, float]:
        """当前用户工作坐标 ``(x, y)``，两个数的单位均为 mm。"""
        if self._cnc.position_valid:
            return self._cnc.position
        # 手动查一次
        before = self._cnc.status_sequence
        self._cnc.query_status()
        self._wait_until(
            lambda: self._cnc.status_sequence > before,
            timeout_s=2.0,
            description="等待位置状态",
        )
        return self._cnc.position

    @property
    def pen_position(self) -> float:
        """当前笔轴绝对工作坐标 Z，单位 mm。"""
        return self._cnc.pen_position

    @property
    def is_busy(self) -> bool:
        """主机队列非空或 GRBL 正在运动时返回 ``True``。"""
        return (
            bool(self._queued_lines)
            or self._stream_inflight is not None
            or self._cnc.state not in ("Idle", "", "Alarm")
        )

    @property
    def machine_state(self) -> str:
        """当前 GRBL 状态字符串，例如 ``Idle``、``Run``、``Alarm``。"""
        return self._cnc.state

    # ── 自定义 G-code ───────────────────────────────────────

    def send_gcode(self, cmd: str) -> None:
        """发送原始 G-code 字符串，等待应答和真实 Idle。

        Args:
            cmd: 一行或多行原始 G-code。坐标是固件坐标，不应用用户
                XY 对调/反向。
        """
        self._ensure_no_stream()
        self._execute_raw(cmd)

    def send_gcode_nowait(self, cmd: str) -> None:
        """把原始 G-code 加入主机流队列，不等待执行完毕。

        队列严格以每条 ``ok/error`` 应答作为发送下一条的流控信号，不会
        无限制灌入固件串口缓冲区。最后必须调用 :meth:`wait_idle`。

        注意：这是原始固件坐标，不应用用户坐标的 swap/reverse 变换。
        用户坐标路径请优先使用 :meth:`move_path`。

        Args:
            cmd: 一行或多行原始 G-code 字符串。
        """
        if not self._cnc.is_connected:
            raise RuntimeError("串口未连接")
        lines = [line.strip() for line in cmd.splitlines() if line.strip()]
        self._queued_lines.extend(lines)
        self._send_next_queued()

    # ── 电机使能控制 ────────────────────────────────────

    def set_motor_hold(self, enable: bool = True) -> None:
        """设置电机保持使能状态。

        GRBL 的 $1 参数控制步进空闲延时：
          - enable=True  → $1=255  （电机持续使能，提供保持力矩）
          - enable=False → $1=25   （运动完 25ms 后失能，可手扳动）

        $1=255 不是机械刹车。特别是带重力负载的 Z 轴，在外力大于步进
        电机保持力矩时仍可能被反驱并丢失开环坐标；若它不施力就自行坠落，
        则应检查驱动使能/电流或机械配重。

        Args:
            enable: True=保持使能, False=运动完失能

        注意:
            本方法修改 GRBL ``$1`` 设置，会写入控制器 EEPROM。
        """
        if enable:
            command = "$1=255"
        else:
            command = "$1=25"
        self._execute_raw(command, wait_motion=False)
        # 验证设置
        self._execute_raw("$1", wait_motion=False)

    # ── 等待 ────────────────────────────────────────────────

    def wait_idle(self) -> None:
        """等待 GRBL 回到 Idle 状态（所有缓冲命令执行完毕）。

        用于 send_gcode_nowait() 批量发送后，确保所有命令完成。
        无返回值；等待超时、Alarm 或命令错误时抛出异常。
        """
        deadline = time.monotonic() + 60.0
        while self._queued_lines or self._stream_inflight is not None:
            if self._stream_error:
                error = self._stream_error
                self._stream_error = None
                self._queued_lines.clear()
                raise RuntimeError(f"GRBL 流发送失败: {error}")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "等待流队列应答超时，"
                    f"主机队列={len(self._queued_lines)}, "
                    f"固件待应答={self._cnc.pending_command_count}"
                )
            self._pump_once()
        if self._stream_error:
            error = self._stream_error
            self._stream_error = None
            raise RuntimeError(f"GRBL 流发送失败: {error}")
        self._wait_idle()

    def _ensure_app(self) -> None:
        """确保 Qt 事件循环在运行。"""
        if self._app is None:
            self._app = QCoreApplication.instance() or QCoreApplication(sys.argv)

    def _pump_once(self, delay_s: float = 0.005) -> None:
        self._app.processEvents()
        if delay_s > 0:
            time.sleep(delay_s)
        self._app.processEvents()
        self._advance_stream()

    def _pump(self, cycles: int) -> None:
        """泵 Qt 事件循环指定次数，每轮包含真实 10ms 墙钟等待。"""
        for _ in range(cycles):
            self._pump_once(0.01)

    def _wait_until(
        self,
        predicate,
        timeout_s: float,
        description: str,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._pump_once()
            if predicate():
                return
            if not self._cnc.is_connected:
                raise RuntimeError(f"{description}时串口已断开")
        raise TimeoutError(f"{description}超时 ({timeout_s:.1f}s)")

    def _wait_for_ack(self, target_sequence: int, timeout_s: float = 10.0) -> None:
        self._wait_until(
            lambda: self._cnc.acknowledged_sequence >= target_sequence,
            timeout_s,
            f"等待命令 #{target_sequence} 应答",
        )

    def _raise_for_results(self, first: int, last: int) -> None:
        for sequence in range(first, last + 1):
            result = self._cnc.command_result(sequence)
            if result is None:
                raise RuntimeError(f"命令 #{sequence} 没有终结应答")
            success, response = result
            if not success:
                raise RuntimeError(f"命令 #{sequence} 被 GRBL 拒绝: {response}")

    def _execute_action(self, action, wait_motion: bool = True) -> None:
        self._ensure_no_stream()
        first = self._cnc.command_sequence + 1
        action()
        last = self._cnc.command_sequence
        if last < first:
            raise RuntimeError("命令未能排入串口发送缓冲区")
        self._wait_for_ack(last)
        self._raise_for_results(first, last)
        if wait_motion:
            self._wait_idle()

    def _execute_raw(self, cmd: str, wait_motion: bool = True) -> None:
        self._execute_action(lambda: self._cnc.send_gcode(cmd), wait_motion)

    def _wait_idle(self, timeout_ms: int = 30000) -> None:
        """在所有命令已应答后，用新的状态汇报确认 planner 真正清空。"""
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            before = self._cnc.status_sequence
            self._cnc.query_status()
            status_deadline = min(deadline, time.monotonic() + 1.0)
            while (
                self._cnc.status_sequence <= before
                and time.monotonic() < status_deadline
            ):
                self._pump_once()
            state = self._cnc.state
            if self._cnc.status_sequence > before and state == "Idle":
                return
            if state in ("Alarm", "Door"):
                raise RuntimeError(f"GRBL 无法完成运动，当前状态: {state}")
            self._pump_once(0.02)
        raise TimeoutError(
            f"等待 Idle 超时 ({timeout_ms}ms)，当前状态: {self._cnc.state}"
        )

    def _ensure_no_stream(self) -> None:
        if self._queued_lines or self._stream_inflight is not None:
            self.wait_idle()

    def _send_next_queued(self) -> None:
        if self._stream_inflight is not None or not self._queued_lines:
            return
        line = self._queued_lines.popleft()
        before = self._cnc.command_sequence
        if not self._cnc.send_gcode(line):
            self._stream_error = f"无法发送: {line}"
            return
        if self._cnc.command_sequence <= before:
            self._stream_error = f"命令没有进入发送队列: {line}"
            return
        self._stream_inflight = self._cnc.command_sequence

    def _advance_stream(self) -> None:
        if self._stream_inflight is None:
            return
        result = self._cnc.command_result(self._stream_inflight)
        if result is None:
            return
        sequence = self._stream_inflight
        success, response = result
        self._stream_inflight = None
        if not success:
            self._stream_error = f"命令 #{sequence}: {response}"
            self._queued_lines.clear()
            return
        self._send_next_queued()


# ── 命令行模式 ────────────────────────────────────────────
# 用于快速调试:
#   python writer_controller.py /dev/ttyUSB0 move 100 50
#   python writer_controller.py /dev/ttyUSB0 pen-up
#   python writer_controller.py /dev/ttyUSB0 home

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="写字机控制")
    parser.add_argument("port", help="串口路径")
    parser.add_argument("action", choices=["home", "move", "jog", "pen-up", "pen-down",
                                            "status", "hold-on", "hold-off", "queue-test", "test"])
    parser.add_argument("args", nargs="*", help="额外参数")

    args = parser.parse_args()

    ctrl = WriterController()
    print(f"连接 {args.port} ...")
    ctrl.connect(args.port)
    print("已连接")

    if args.action == "home":
        input("请把机构放到右下角并按 Enter...")
        ctrl.home()
        print(f"归零完成, 位置: {ctrl.position}")

    elif args.action == "move":
        if len(args.args) >= 2:
            x, y = float(args.args[0]), float(args.args[1])
            print(f"移动到 ({x}, {y}) ...")
            ctrl.move_to(x, y)
            print(f"到达, 位置: {ctrl.position}")
        else:
            print("用法: move <x> <y>")

    elif args.action == "jog":
        if len(args.args) >= 2:
            dx, dy = float(args.args[0]), float(args.args[1])
            print(f"点动 ({dx:+}, {dy:+}) ...")
            ctrl.jog(dx, dy)
            print(f"完成, 位置: {ctrl.position}")
        else:
            print("用法: jog <dx> <dy>")

    elif args.action == "pen-up":
        ctrl.pen_up()
        print(f"抬笔, Z={ctrl.pen_position}")

    elif args.action == "pen-down":
        ctrl.pen_down()
        print(f"落笔, Z={ctrl.pen_position}")

    elif args.action == "status":
        print(f"位置: {ctrl.position}")
        print(f"笔轴: {ctrl.pen_position}")
        print(f"状态: {ctrl.machine_state}")
        print(f"忙: {ctrl.is_busy}")

    elif args.action == "hold-on":
        ctrl.set_motor_hold(True)
        print("电机保持使能: ON（提供保持力矩；外力过大仍可能反驱）")

    elif args.action == "hold-off":
        ctrl.set_motor_hold(False)
        print("电机保持使能: OFF (可手动扳动)")

    elif args.action == "queue-test":
        print("G-code 应答流控队列测试")
        input("放右下角, 按 Enter 归零...")
        ctrl.home()

        n = 20
        print(f"连续经过 {n} 个用户绝对坐标点（每条 ok 后发下一条）...")
        started = time.monotonic()
        ctrl.move_path(((float(i), 0.0) for i in range(1, n + 1)), 2000)
        elapsed = time.monotonic() - started
        x, y = ctrl.position
        print(f"完成, 耗时 {elapsed:.2f}s, 位置: ({x:.2f}, {y:.2f})")
        print(f"预期用户 X≈{n:.0f}mm, 实际 X={x:.2f}mm")

        print("回原点...")
        ctrl.move_to(0, 0)
        print(f"位置: {ctrl.position}")

    elif args.action == "test":
        print("执行快速测试: 归零 → 走矩形 → 抬落笔")
        input("放右下角, 按 Enter...")
        ctrl.home()
        ctrl.pen_up()
        print("抬笔")

        for x, y in [(50, 0), (50, 50), (0, 50), (0, 0)]:
            print(f"  移动到 ({x}, {y}) ...")
            ctrl.move_to(x, y)
            print(f"  到达")

        ctrl.pen_down()
        print("落笔")
        ctrl.pen_up()
        print("抬笔")

        print(f"最终位置: {ctrl.position}")

    ctrl.disconnect()
    print("断开")
