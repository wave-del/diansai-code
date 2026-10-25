"""基于 MKS ``xy_axes.c`` 实现并反转 XY 物理方向的绝对坐标状态机。"""

from __future__ import annotations

from collections.abc import Callable
import math
import time

from ec_26_e.motion.contracts import (
    MotorAddress,
    XYFault,
    XYResult,
    XYServoTransport,
    XYState,
)
from ec_26_e.motion.controller import X_RIGHT_COMPENSATION_PULSES
from ec_26_e.motion.protocol import Servo42CProtocol


XY_X_MAX_0P01MM = 23_000
XY_Y_MIN_0P01MM = -1_000
XY_Y_MAX_0P01MM = 36_000
XY_PULSES_PER_0P01MM = 4
XY_MOVE_SPEED = 64
XY_POWER_DELAY_MS = 500
XY_REPLY_TIMEOUT_MS = 100
XY_MOVE_TIMEOUT_MARGIN_MS = 500
XY_REQUIRE_MOTOR_REPLY = False

_MOTOR_X0_MASK = 0x01
_MOTOR_X1_MASK = 0x02
_MOTOR_Y_MASK = 0x04
_MOTOR_X_MASK = _MOTOR_X0_MASK | _MOTOR_X1_MASK
_MOTOR_ALL_MASK = _MOTOR_X_MASK | _MOTOR_Y_MASK


def mm_to_0p01mm(name: str, value: float) -> int:
    """按 C 接口精度把毫米转换为 0.01 mm 整数，拒绝额外小数。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是实数")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数")
    units = number * 100.0
    rounded = round(units)
    if not math.isclose(units, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{name} 必须精确到 0.01 mm")
    return int(rounded)


class MksXYController:
    """保持 C 状态流程，并使用已反转的 XY 物理方向。"""

    def __init__(
        self,
        transport: XYServoTransport,
        *,
        require_motor_reply: bool = XY_REQUIRE_MOTOR_REPLY,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """保存依赖；调用 ``initialize`` 后才进入 C 的 WAIT_POWER 状态。"""

        if not isinstance(transport, XYServoTransport):
            raise TypeError("transport 必须实现 XYServoTransport")
        if not isinstance(require_motor_reply, bool):
            raise TypeError("require_motor_reply 必须是 bool")
        if not callable(monotonic):
            raise TypeError("monotonic 必须可调用")
        self._transport = transport
        self._require_motor_reply = require_motor_reply
        self._monotonic = monotonic
        self._right_compensation_pending = True
        self._initialized = False
        self.state = XYState.WAIT_POWER
        self.fault = XYFault.NONE
        self.fault_address: MotorAddress | None = None
        self.position_valid = True
        self.x_position_0p01mm = 0
        self.y_position_0p01mm = 0
        self.x_target_0p01mm = 0
        self.y_target_0p01mm = 0
        self.active_motor_mask = 0
        self.started_motor_mask = 0
        self.completed_motor_mask = 0
        self.move_timeout_ms = 0
        self.reset_requested = False
        self.reset_in_progress = False
        self.reset_completed = False
        self._state_deadline_ms = 0
        self._move_deadline_ms = 0

    def _now_ms(self) -> int:
        """返回与 C 毫秒 Tick 对应的单调时钟毫秒值。"""

        return math.floor(self._monotonic() * 1000.0 + 1e-6)

    def _require_initialized(self) -> None:
        """拒绝在 ``initialize`` 之前驱动状态机。"""

        if not self._initialized:
            raise RuntimeError("XY 控制器尚未 initialize")

    def initialize(self) -> None:
        """复刻 ``XY_Init``，清空 UART 回包并假定当前位置为原点。"""

        now_ms = self._now_ms()
        self._transport.discard_replies()
        self.state = XYState.WAIT_POWER
        self.fault = XYFault.NONE
        self.fault_address = None
        self.position_valid = True
        self.x_position_0p01mm = 0
        self.y_position_0p01mm = 0
        self.x_target_0p01mm = 0
        self.y_target_0p01mm = 0
        self.active_motor_mask = 0
        self.started_motor_mask = 0
        self.completed_motor_mask = 0
        self.move_timeout_ms = 0
        self.reset_requested = False
        self.reset_in_progress = False
        self.reset_completed = False
        self._state_deadline_ms = now_ms + XY_POWER_DELAY_MS
        self._move_deadline_ms = 0
        self._initialized = True

    @staticmethod
    def _address_to_mask(address: MotorAddress) -> int:
        """逐项对应 C 的 ``xy_address_to_mask``。"""

        return {
            MotorAddress.X0: _MOTOR_X0_MASK,
            MotorAddress.X1: _MOTOR_X1_MASK,
            MotorAddress.Y: _MOTOR_Y_MASK,
        }.get(address, 0)

    def _first_missing_address(self, received_mask: int) -> MotorAddress:
        """按 C 的 X0、X1、Y 顺序返回首个缺失电机地址。"""

        missing_mask = self.active_motor_mask & ~received_mask
        if missing_mask & _MOTOR_X0_MASK:
            return MotorAddress.X0
        if missing_mask & _MOTOR_X1_MASK:
            return MotorAddress.X1
        return MotorAddress.Y

    def _send_enable(self, address: MotorAddress, enabled: bool) -> None:
        """发送与 C ``setEnable`` 完全相同的报文。"""

        self._transport.send_frame(
            Servo42CProtocol.build_enable(address, enabled)
        )

    def _send_stop(self, address: MotorAddress) -> None:
        """发送与 C ``stop`` 完全相同的报文。"""

        self._transport.send_frame(Servo42CProtocol.build_stop(address))

    def _send_move(
        self,
        address: MotorAddress,
        speed: int,
        pulses: int,
    ) -> None:
        """发送与 C ``move`` 完全相同的报文。"""

        self._transport.send_frame(
            Servo42CProtocol.build_move(address, speed, pulses)
        )

    def _stop_and_disable(self, motor_mask: int) -> None:
        """复刻 C 故障路径：X 任一侧参与时停止并释放 X 双电机。"""

        if motor_mask & _MOTOR_X_MASK:
            self._send_stop(MotorAddress.X0)
            self._send_stop(MotorAddress.X1)
            self._send_enable(MotorAddress.X0, False)
            self._send_enable(MotorAddress.X1, False)
        if motor_mask & _MOTOR_Y_MASK:
            self._send_stop(MotorAddress.Y)
            self._send_enable(MotorAddress.Y, False)

    def _latch_fault(
        self,
        fault: XYFault,
        address: MotorAddress | None,
        invalidate_position: bool,
    ) -> None:
        """停止并失能活动轴，随后锁存与 C 相同的故障状态。"""

        stop_mask = self.active_motor_mask or _MOTOR_ALL_MASK
        self._stop_and_disable(stop_mask)
        if invalidate_position:
            self.position_valid = False
        self.fault = fault
        self.fault_address = address
        self.reset_requested = False
        self.reset_in_progress = False
        self.reset_completed = False
        self.state = XYState.FAULT

    @staticmethod
    def calculate_expected_move_ms(x_pulses: int, y_pulses: int) -> int:
        """按速度档 64 计算无回包运动的理论完成毫秒数。"""

        maximum_pulses = max(x_pulses, y_pulses)
        return (
            2 * maximum_pulses + XY_MOVE_SPEED - 1
        ) // XY_MOVE_SPEED

    @staticmethod
    def calculate_move_timeout_ms(x_pulses: int, y_pulses: int) -> int:
        """保留 C 的双倍理论时间加固定余量故障超时。"""

        expected_ms = MksXYController.calculate_expected_move_ms(
            x_pulses,
            y_pulses,
        )
        return expected_ms * 2 + XY_MOVE_TIMEOUT_MARGIN_MS

    def _send_next_enable(
        self,
        address: MotorAddress,
        wait_state: XYState,
    ) -> None:
        """清空旧回包、使能一个地址并进入对应等待状态。"""

        self._transport.discard_replies()
        self._send_enable(address, True)
        self._state_deadline_ms = self._now_ms() + XY_REPLY_TIMEOUT_MS
        self.state = wait_state

    def _finish_move(self) -> None:
        """提交目标位置并复刻复位完成标志的更新。"""

        self.x_position_0p01mm = self.x_target_0p01mm
        self.y_position_0p01mm = self.y_target_0p01mm
        self.active_motor_mask = 0
        self.started_motor_mask = 0
        self.completed_motor_mask = 0
        self.state = XYState.READY
        if self.reset_in_progress:
            self.reset_in_progress = False
            self.reset_completed = True

    def _service_reset_request(self) -> None:
        """在 READY 时服务由 ``reset_to_origin`` 排队的复位请求。"""

        if not self.reset_requested or self.state != XYState.READY:
            return
        if self.x_position_0p01mm == 0 and self.y_position_0p01mm == 0:
            self.reset_requested = False
            self.reset_in_progress = False
            self.reset_completed = True
            return
        result = self.move_to(0, 0)
        if result == XYResult.OK:
            self.reset_requested = False
            self.reset_in_progress = True

    def _process_enable_reply(
        self,
        expected_address: MotorAddress,
        next_address: MotorAddress | None,
        next_state: XYState,
    ) -> None:
        """消费回包并复刻 C 的逐电机使能状态迁移。"""

        while True:
            reply = self._transport.take_reply()
            if reply is None:
                break
            if reply.address != expected_address:
                continue
            if reply.status != 0x01:
                self._latch_fault(
                    XYFault.ENABLE_FAILED,
                    expected_address,
                    False,
                )
            elif next_state == XYState.READY:
                self.state = XYState.READY
            else:
                if next_address is None:
                    raise RuntimeError("下一个使能地址不能为空")
                self._send_next_enable(next_address, next_state)
            return
        if self._now_ms() >= self._state_deadline_ms:
            self._latch_fault(
                XYFault.ENABLE_TIMEOUT,
                expected_address,
                False,
            )

    def _process_move_replies(self) -> None:
        """复刻 C 的活动电机开始/完成掩码和故障判断。"""

        while True:
            reply = self._transport.take_reply()
            if reply is None:
                break
            motor_mask = self._address_to_mask(reply.address)
            if motor_mask == 0 or not motor_mask & self.active_motor_mask:
                continue
            if reply.status not in {0x01, 0x02}:
                self._latch_fault(
                    XYFault.MOVE_REJECTED,
                    reply.address,
                    True,
                )
                return
            self.started_motor_mask |= motor_mask
            if reply.status == 0x02:
                self.completed_motor_mask |= motor_mask

        if self.completed_motor_mask == self.active_motor_mask:
            self._finish_move()
            return
        if self.state == XYState.WAIT_MOVE_START:
            if self.started_motor_mask == self.active_motor_mask:
                self.state = XYState.WAIT_MOVE_COMPLETE
            elif self._now_ms() >= self._state_deadline_ms:
                self._latch_fault(
                    XYFault.MOVE_START_TIMEOUT,
                    self._first_missing_address(self.started_motor_mask),
                    True,
                )
        if (
            self.state == XYState.WAIT_MOVE_COMPLETE
            and self._now_ms() >= self._move_deadline_ms
        ):
            self._latch_fault(
                XYFault.MOVE_COMPLETE_TIMEOUT,
                self._first_missing_address(self.completed_motor_mask),
                True,
            )

    def is_ready(self) -> bool:
        """复刻 ``XY_IsReady``。"""

        self._require_initialized()
        return self.state == XYState.READY and self.position_valid

    def is_at_origin(self) -> bool:
        """复刻 ``XY_IsAtOrigin``。"""

        return (
            self.is_ready()
            and self.x_position_0p01mm == 0
            and self.y_position_0p01mm == 0
        )

    def move_to(self, x_0p01mm: int, y_0p01mm: int) -> XYResult:
        """复刻 ``XY_MoveTo`` 的检查、发报顺序和状态迁移。"""

        self._require_initialized()
        for name, value in (
            ("x_0p01mm", x_0p01mm),
            ("y_0p01mm", y_0p01mm),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} 必须是整数")
        if self.state == XYState.FAULT:
            return (
                XYResult.NOT_READY
                if self.position_valid
                else XYResult.POSITION_INVALID
            )
        if self.state != XYState.READY:
            if self.state in {
                XYState.WAIT_MOVE_START,
                XYState.WAIT_MOVE_COMPLETE,
            }:
                return XYResult.BUSY
            return XYResult.NOT_READY
        if not self.position_valid:
            return XYResult.POSITION_INVALID
        if (
            x_0p01mm < 0
            or x_0p01mm > XY_X_MAX_0P01MM
            or y_0p01mm < XY_Y_MIN_0P01MM
            or y_0p01mm > XY_Y_MAX_0P01MM
        ):
            return XYResult.OUT_OF_RANGE

        x_delta = x_0p01mm - self.x_position_0p01mm
        y_delta = y_0p01mm - self.y_position_0p01mm
        if x_delta == 0 and y_delta == 0:
            return XYResult.OK
        x_pulses = abs(x_delta) * XY_PULSES_PER_0P01MM
        y_pulses = abs(y_delta) * XY_PULSES_PER_0P01MM
        if x_delta > 0 and self._right_compensation_pending:
            x_pulses += X_RIGHT_COMPENSATION_PULSES
            self._right_compensation_pending = False
        self.x_target_0p01mm = x_0p01mm
        self.y_target_0p01mm = y_0p01mm
        self.active_motor_mask = 0
        if x_pulses:
            self.active_motor_mask |= _MOTOR_X_MASK
        if y_pulses:
            self.active_motor_mask |= _MOTOR_Y_MASK
        self.started_motor_mask = 0
        self.completed_motor_mask = 0

        self._transport.discard_replies()
        if x_pulses:
            # 原固件增大 X 向左；此处反转为增大 X 向右。
            x_speed = -XY_MOVE_SPEED if x_delta > 0 else XY_MOVE_SPEED
            self._send_move(MotorAddress.X0, -x_speed, x_pulses)
            self._send_move(MotorAddress.X1, x_speed, x_pulses)
        if y_pulses:
            # 原固件增大 Y 向上；此处反转为增大 Y 向下。
            y_speed = -XY_MOVE_SPEED if y_delta > 0 else XY_MOVE_SPEED
            self._send_move(MotorAddress.Y, y_speed, y_pulses)

        now_ms = self._now_ms()
        expected_move_ms = self.calculate_expected_move_ms(
            x_pulses,
            y_pulses,
        )
        self.move_timeout_ms = self.calculate_move_timeout_ms(
            x_pulses,
            y_pulses,
        )
        self._state_deadline_ms = now_ms + XY_REPLY_TIMEOUT_MS
        # 无回包模式用理论完成时间提交软件位置；外层任务另加固定安全余量。
        # 要求回包时仍使用更宽松的故障超时，避免误报正常加减速过程。
        self._move_deadline_ms = now_ms + (
            self.move_timeout_ms
            if self._require_motor_reply
            else expected_move_ms
        )
        if self._require_motor_reply:
            self.state = XYState.WAIT_MOVE_START
        else:
            self.started_motor_mask = self.active_motor_mask
            self.state = XYState.WAIT_MOVE_COMPLETE
        return XYResult.OK

    def reset_to_origin(self) -> XYResult:
        """复刻 ``XY_ResetToOrigin`` 的可排队复位请求。"""

        self._require_initialized()
        if not self.position_valid:
            return XYResult.POSITION_INVALID
        if self.state == XYState.FAULT:
            return XYResult.NOT_READY
        if self.reset_requested or self.reset_in_progress:
            return XYResult.OK
        self.reset_completed = False
        self.reset_requested = True
        return XYResult.OK

    def emergency_stop(self) -> None:
        """复刻 ``XY_EmergencyStop``，停止失能全部 XY 并使位置失效。"""

        self._require_initialized()
        self.active_motor_mask = _MOTOR_ALL_MASK
        self._latch_fault(XYFault.EMERGENCY_STOP, None, True)

    def assume_origin(self) -> None:
        """复刻 ``XY_AssumeOrigin``，停止失能后重新经历上电等待。"""

        self._require_initialized()
        self._stop_and_disable(_MOTOR_ALL_MASK)
        self._transport.discard_replies()
        self.fault = XYFault.NONE
        self.fault_address = None
        self.position_valid = True
        self.x_position_0p01mm = 0
        self.y_position_0p01mm = 0
        self.x_target_0p01mm = 0
        self.y_target_0p01mm = 0
        self.active_motor_mask = 0
        self.started_motor_mask = 0
        self.completed_motor_mask = 0
        self.reset_requested = False
        self.reset_in_progress = False
        self.reset_completed = True
        self._state_deadline_ms = self._now_ms() + XY_POWER_DELAY_MS
        self.state = XYState.WAIT_POWER

    def process(self) -> None:
        """复刻 ``XY_Process`` 的单次非阻塞状态机推进。"""

        self._require_initialized()
        if self.state == XYState.WAIT_POWER:
            if self._now_ms() >= self._state_deadline_ms:
                if self._require_motor_reply:
                    self._send_next_enable(
                        MotorAddress.X0,
                        XYState.WAIT_ENABLE_X0,
                    )
                else:
                    self._transport.discard_replies()
                    self._send_enable(MotorAddress.X0, True)
                    self._send_enable(MotorAddress.X1, True)
                    self._send_enable(MotorAddress.Y, True)
                    self.state = XYState.READY
        elif self.state == XYState.WAIT_ENABLE_X0:
            self._process_enable_reply(
                MotorAddress.X0,
                MotorAddress.X1,
                XYState.WAIT_ENABLE_X1,
            )
        elif self.state == XYState.WAIT_ENABLE_X1:
            self._process_enable_reply(
                MotorAddress.X1,
                MotorAddress.Y,
                XYState.WAIT_ENABLE_Y,
            )
        elif self.state == XYState.WAIT_ENABLE_Y:
            self._process_enable_reply(
                MotorAddress.Y,
                None,
                XYState.READY,
            )
        elif self.state in {
            XYState.WAIT_MOVE_START,
            XYState.WAIT_MOVE_COMPLETE,
        }:
            if self._require_motor_reply:
                self._process_move_replies()
            else:
                self._transport.discard_replies()
                if self._now_ms() >= self._move_deadline_ms:
                    self.completed_motor_mask = self.active_motor_mask
                    self._finish_move()
        self._service_reset_request()
