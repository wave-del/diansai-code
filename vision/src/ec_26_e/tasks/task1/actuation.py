"""编排 MovePlan 无回包运动、磁铁时序、状态迁移和安全清理。"""

from __future__ import annotations

from collections.abc import Callable
import math
import time

from ec_26_e.actuation import (
    ActuationConfig,
    ActuationExecutionError,
    ActuationResult,
    ActuationShutdownError,
    ActuationValidationError,
    GpiodCompletionLedOutput,
    GpiodMagnetOutput,
    MotionLifecycleState,
    TransferCycle,
    compile_move_plan,
)
from ec_26_e.actuation.contracts import (
    CompletionIndicator,
    MagnetOutput,
    NoReplyTransport,
    RotationAxis,
)
from ec_26_e.control import MovePlan
from ec_26_e.motion import (
    FocRotationAxis,
    MksAxisController,
    MksXYController,
    Servo42CSerial,
    XYResult,
    XYState,
)


MonotonicClock = Callable[[], float]
Sleeper = Callable[[float], None]
MagnetFactory = Callable[[ActuationConfig], MagnetOutput]
CompletionIndicatorFactory = Callable[[], CompletionIndicator]
TransportFactory = Callable[[ActuationConfig], NoReplyTransport]
RotationFactory = Callable[[ActuationConfig], RotationAxis]


# 任务完成提示灯固定点亮 1 秒，避免受拾取/释放时序参数影响。
COMPLETION_LED_ON_SECONDS = 1.0


def _default_magnet_factory(config: ActuationConfig) -> MagnetOutput:
    """构造真实 libgpiod 磁铁输出。"""

    return GpiodMagnetOutput(config.magnet)


def _default_completion_indicator_factory() -> CompletionIndicator:
    """构造 A7Z 物理 pin33 上尚未打开的完成提示灯。"""

    return GpiodCompletionLedOutput()


def _default_transport_factory(config: ActuationConfig) -> NoReplyTransport:
    """构造尚未打开的真实 Servo42C 串口。"""

    return Servo42CSerial(config.serial)


def _default_rotation_factory(config: ActuationConfig) -> RotationAxis:
    """构造尚未打开的真实 F32C 定向旋转轴。"""

    return FocRotationAxis(config.rotation)


class MovePlanActuator:
    """执行一份 MovePlan，并在 FINISH 后立即安全失能退出。"""

    def __init__(
        self,
        config: ActuationConfig,
        *,
        magnet_factory: MagnetFactory = _default_magnet_factory,
        completion_indicator_factory: CompletionIndicatorFactory | None = (
            _default_completion_indicator_factory
        ),
        transport_factory: TransportFactory = _default_transport_factory,
        rotation_factory: RotationFactory = _default_rotation_factory,
        monotonic: MonotonicClock = time.monotonic,
        sleep: Sleeper = time.sleep,
    ) -> None:
        """保存强类型配置和可替换硬件、时钟依赖。"""

        if not isinstance(config, ActuationConfig):
            raise TypeError("config 必须是 ActuationConfig")
        for name, dependency in (
            ("magnet_factory", magnet_factory),
            ("transport_factory", transport_factory),
            ("rotation_factory", rotation_factory),
            ("monotonic", monotonic),
            ("sleep", sleep),
        ):
            if not callable(dependency):
                raise TypeError(f"{name} 必须可调用")
        if (
            completion_indicator_factory is not None
            and not callable(completion_indicator_factory)
        ):
            raise TypeError("completion_indicator_factory 必须可调用或为 None")
        self._config = config
        self._magnet_factory = magnet_factory
        self._completion_indicator_factory = completion_indicator_factory
        self._transport_factory = transport_factory
        self._rotation_factory = rotation_factory
        self._monotonic = monotonic
        self._sleep = sleep
        self._state = MotionLifecycleState.IDLE
        self._state_history = [self._state]
        self._completed_cycles = 0
        self._presumed_z_degrees = config.z_boot_degrees
        self._position_valid = True

    @property
    def state(self) -> MotionLifecycleState:
        """返回当前顶层生命周期状态。"""

        return self._state

    @property
    def state_history(self) -> tuple[MotionLifecycleState, ...]:
        """返回不可变状态迁移历史，供诊断和测试使用。"""

        return tuple(self._state_history)

    @property
    def position_valid(self) -> bool:
        """返回无反馈软件位置是否仍可作为后续计算依据。"""

        return self._position_valid

    def _transition(self, state: MotionLifecycleState) -> None:
        """提交一次顶层状态迁移。"""

        self._state = state
        self._state_history.append(state)

    def _wait_duration(
        self,
        seconds: float,
        *,
        xy_controller: MksXYController | None = None,
    ) -> None:
        """按单调时钟等待，并在等待期间推进 XY 状态机。"""

        deadline = self._monotonic() + seconds
        while True:
            if xy_controller is not None:
                xy_controller.process()
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                return
            self._sleep(min(0.001, remaining))

    @staticmethod
    def _send_commands(
        transport: NoReplyTransport,
        commands: tuple[object, ...],
    ) -> None:
        """按规划顺序发送命令且绝不自动重试。"""

        for command in commands:
            frame = getattr(command, "frame", None)
            if not isinstance(frame, bytes):
                raise TypeError("运动规划命令必须包含 bytes frame")
            transport.send_frame(frame)

    def _move_z_to(
        self,
        target_degrees: float,
        transport: NoReplyTransport,
        xy_controller: MksXYController,
    ) -> None:
        """把绝对 Z 软件目标拆成单条不超过 90°的相对指令。"""

        remaining = target_degrees - self._presumed_z_degrees
        while not math.isclose(remaining, 0.0, abs_tol=1e-12):
            step = math.copysign(min(abs(remaining), 90.0), remaining)
            commands = MksAxisController.plan_relative(
                z_degrees=step,
                speed=self._config.z_speed_level,
            )
            self._send_commands(transport, commands)
            self._wait_duration(
                self._config.z_move_wait_seconds,
                xy_controller=xy_controller,
            )
            self._presumed_z_degrees += step
            remaining = target_degrees - self._presumed_z_degrees

    def _start_xy_move(
        self,
        x_0p01mm: int,
        y_0p01mm: int,
        xy_controller: MksXYController,
    ) -> float:
        """发送 XY 绝对目标并返回外层最早完成截止时间。"""

        old_x = xy_controller.x_position_0p01mm
        old_y = xy_controller.y_position_0p01mm
        distance_mm = max(
            abs(x_0p01mm - old_x),
            abs(y_0p01mm - old_y),
        ) / 100.0
        minimum_wait = (
            distance_mm / self._config.xy_speed_mm_per_second
            + self._config.xy_fixed_wait_seconds
        )
        result = xy_controller.move_to(x_0p01mm, y_0p01mm)
        if result != XYResult.OK:
            raise RuntimeError(f"XY 移动请求被拒绝：{result.name}")
        return self._monotonic() + minimum_wait

    def _wait_xy_move(
        self,
        deadline: float,
        xy_controller: MksXYController,
    ) -> None:
        """等待 XY 同时满足估算截止时间和无回包 READY。"""

        while True:
            xy_controller.process()
            remaining = deadline - self._monotonic()
            if (
                remaining <= 0
                and xy_controller.state == XYState.READY
            ):
                return
            if xy_controller.state == XYState.FAULT:
                raise RuntimeError(
                    f"XY 状态机故障：{xy_controller.fault.name}"
                )
            self._sleep(
                min(0.001, remaining)
                if remaining > 0
                else 0.001
            )

    def _move_xy_to(
        self,
        x_0p01mm: int,
        y_0p01mm: int,
        xy_controller: MksXYController,
    ) -> None:
        """发送 XY 绝对目标并等待底层和标定时间同时完成。"""

        deadline = self._start_xy_move(
            x_0p01mm,
            y_0p01mm,
            xy_controller,
        )
        self._wait_xy_move(deadline, xy_controller)

    def _move_xy_and_rotate(
        self,
        cycle: TransferCycle,
        xy_controller: MksXYController,
        rotation: RotationAxis,
    ) -> None:
        """启动放置 XY 后立即顺时针旋转，并等待两者均完成。"""

        deadline = self._start_xy_move(
            cycle.place_x_0p01mm,
            cycle.place_y_0p01mm,
            xy_controller,
        )
        try:
            rotation.move_clockwise_from_zero(
                cycle.rotation_degrees_clockwise,
                wait=True,
            )
        except Exception as error:
            raise RuntimeError(f"旋转轴运动失败：{error}") from error
        self._wait_xy_move(deadline, xy_controller)

    @staticmethod
    def _circular_error(actual: float, expected: float) -> float:
        """计算两个机械单圈角之间的最小误差。"""

        return abs(
            (float(actual) - float(expected) + 180.0) % 360.0 - 180.0
        )

    def _return_rotation_to_zero(
        self,
        rotation: RotationAxis,
        target_degrees: float,
        xy_controller: MksXYController,
    ) -> None:
        """逆时针回零，并同时满足反馈到位和最小回零等待。"""

        deadline = (
            self._monotonic() + self._config.rotation_zero_wait_seconds
        )
        try:
            rotation.return_counterclockwise_to_zero(
                target_degrees,
                wait=True,
            )
        except Exception as error:
            raise RuntimeError(f"旋转轴回零失败：{error}") from error
        remaining = deadline - self._monotonic()
        if remaining > 0:
            self._wait_duration(remaining, xy_controller=xy_controller)
        if self._circular_error(rotation.current_angle_deg(), 0.0) > 1.0:
            raise RuntimeError("旋转轴回零反馈超出 1° 容差")

    def _move_next_pick_and_return_rotation_to_zero(
        self,
        cycle: TransferCycle,
        next_cycle: TransferCycle | None,
        xy_controller: MksXYController,
        rotation: RotationAxis,
    ) -> None:
        """并行移动到下一拾取点和旋转归零，末片只执行归零。"""

        xy_deadline: float | None = None
        if next_cycle is not None:
            xy_deadline = self._start_xy_move(
                next_cycle.pick_x_0p01mm,
                next_cycle.pick_y_0p01mm,
                xy_controller,
            )
        self._return_rotation_to_zero(
            rotation,
            cycle.rotation_degrees_clockwise,
            xy_controller,
        )
        if xy_deadline is not None:
            self._wait_xy_move(xy_deadline, xy_controller)

    def _signal_completion(
        self,
        transport: NoReplyTransport,
        magnet: MagnetOutput,
        xy_controller: MksXYController,
    ) -> None:
        """最后一片放置完成后，Z 保持抬升并输出一秒最终提示。"""

        self._move_z_to(
            self._config.z_up_degrees,
            transport,
            xy_controller,
        )
        if self._completion_indicator_factory is None:
            magnet.set_enabled(True)
            self._wait_duration(
                COMPLETION_LED_ON_SECONDS,
                xy_controller=xy_controller,
            )
            magnet.set_enabled(False)
            return
        indicator = self._completion_indicator_factory()
        indicator.open()
        try:
            indicator.set_enabled(True)
            self._wait_duration(
                COMPLETION_LED_ON_SECONDS,
                xy_controller=xy_controller,
            )
            indicator.set_enabled(False)
        finally:
            indicator.close()

    def _initialize_rotation_zero(
        self,
        rotation: RotationAxis,
        xy_controller: MksXYController,
    ) -> None:
        """把启动时当前机械角视为目标并逆时针回到零位。"""

        try:
            current = rotation.current_angle_deg() % 360.0
        except Exception as error:
            raise RuntimeError(
                f"旋转轴启动位置检查失败：{error}"
            ) from error
        sign = self._config.rotation.clockwise_mechanical_sign
        clockwise_target = (sign * current) % 360.0
        if self._circular_error(current, 0.0) <= 1.0:
            clockwise_target = 0.0
        self._return_rotation_to_zero(
            rotation,
            clockwise_target,
            xy_controller,
        )

    def _execute_cycle(
        self,
        cycle: TransferCycle,
        transport: NoReplyTransport,
        magnet: MagnetOutput,
        xy_controller: MksXYController,
        rotation: RotationAxis,
        *,
        pick_positioned: bool,
        next_cycle: TransferCycle | None,
    ) -> None:
        """执行一片纸，并为下一片并行完成 XY 预定位和旋转归零。"""

        if self._circular_error(rotation.current_angle_deg(), 0.0) > 1.0:
            raise RuntimeError("吸取纸片前旋转轴不在机械零点")
        if pick_positioned:
            if (
                xy_controller.x_position_0p01mm != cycle.pick_x_0p01mm
                or xy_controller.y_position_0p01mm != cycle.pick_y_0p01mm
            ):
                raise RuntimeError("下一片 XY 预定位结果与拾取目标不一致")
        else:
            self._move_xy_to(
                cycle.pick_x_0p01mm,
                cycle.pick_y_0p01mm,
                xy_controller,
            )
        self._move_z_to(
            self._config.z_down_degrees,
            transport,
            xy_controller,
        )
        magnet.set_enabled(True)
        self._wait_duration(
            self._config.magnet_attach_wait_seconds,
            xy_controller=xy_controller,
        )
        self._move_z_to(
            self._config.z_up_degrees,
            transport,
            xy_controller,
        )
        self._move_xy_and_rotate(
            cycle,
            xy_controller,
            rotation,
        )
        self._move_z_to(
            self._config.z_down_degrees,
            transport,
            xy_controller,
        )
        # Z 下降完成后立即断开电磁铁，不在断电前增加等待。
        magnet.set_enabled(False)
        self._wait_duration(
            self._config.magnet_release_wait_seconds,
            xy_controller=xy_controller,
        )
        self._move_z_to(
            self._config.z_up_degrees,
            transport,
            xy_controller,
        )
        self._move_next_pick_and_return_rotation_to_zero(
            cycle,
            next_cycle,
            xy_controller,
            rotation,
        )

    @staticmethod
    def _append_cleanup_error(
        errors: list[str],
        label: str,
        operation: Callable[[], None],
    ) -> None:
        """执行一项 best-effort 清理并收集中文错误。"""

        try:
            operation()
        except Exception as error:
            errors.append(f"{label}：{error}")

    def _emergency_cleanup(
        self,
        transport: NoReplyTransport | None,
        magnet: MagnetOutput | None,
        rotation: RotationAxis | None,
        *,
        motors_enabled: bool,
    ) -> list[str]:
        """按消磁、停止、失能、关闭的固定顺序尽力清理。"""

        errors: list[str] = []
        if magnet is not None:
            self._append_cleanup_error(
                errors,
                "磁铁消磁失败",
                lambda: magnet.set_enabled(False),
            )
        if transport is not None and motors_enabled:
            self._append_cleanup_error(
                errors,
                "电机停止失败",
                lambda: self._send_commands(
                    transport,
                    MksAxisController.plan_stop("xyz"),
                ),
            )
            self._append_cleanup_error(
                errors,
                "电机失能失败",
                lambda: self._send_commands(
                    transport,
                    MksAxisController.plan_power(
                        "xyz",
                        enabled=False,
                    ),
                ),
            )
        if rotation is not None:
            self._append_cleanup_error(
                errors,
                "旋转轴失能或串口关闭失败",
                rotation.close,
            )
        self._position_valid = False
        if transport is not None:
            self._append_cleanup_error(
                errors,
                "串口关闭失败",
                transport.close,
            )
        if magnet is not None:
            self._append_cleanup_error(
                errors,
                "GPIO 释放失败",
                magnet.close,
            )
        return errors

    def execute(self, plan: MovePlan) -> ActuationResult:
        """预检并执行 MovePlan，FINISH 后失能电机并安全退出。"""

        if self._state != MotionLifecycleState.IDLE:
            raise RuntimeError("同一运动执行器只能执行一次 MovePlan")
        self._transition(MotionLifecycleState.VALIDATING)
        try:
            cycles = compile_move_plan(plan, self._config)
        except ActuationValidationError as error:
            self._transition(MotionLifecycleState.FAILED)
            raise ActuationValidationError(
                f"移动计划安全检查失败：{error}"
            ) from error
        except Exception as error:
            self._transition(MotionLifecycleState.FAILED)
            raise ActuationValidationError(
                f"移动计划安全检查失败：{error}"
            ) from error

        magnet: MagnetOutput | None = None
        transport: NoReplyTransport | None = None
        rotation: RotationAxis | None = None
        motors_enabled = False
        plan_finished = False
        try:
            try:
                magnet = self._magnet_factory(self._config)
                magnet.open()
                magnet.set_enabled(False)
            except Exception as error:
                raise ActuationExecutionError(
                    f"磁铁控制初始化失败：{error}"
                ) from error
            try:
                transport = self._transport_factory(self._config)
                transport.open()
            except Exception as error:
                raise ActuationExecutionError(
                    f"电机使能失败：无法打开运动串口：{error}"
                ) from error
            try:
                rotation = self._rotation_factory(self._config)
                rotation.open()
            except Exception as error:
                raise ActuationExecutionError(
                    f"电机使能失败：旋转轴初始化失败：{error}"
                ) from error

            self._transition(MotionLifecycleState.ENABLING)
            xy_controller = MksXYController(
                transport,
                require_motor_reply=False,
                monotonic=self._monotonic,
            )
            xy_controller.initialize()
            # 即使首条使能报文部分写入后失败，也必须按已使能处理并清理。
            motors_enabled = True
            try:
                self._send_commands(
                    transport,
                    MksAxisController.plan_power("z", enabled=True),
                )
            except Exception as error:
                raise ActuationExecutionError(
                    f"电机使能失败：{error}"
                ) from error
            self._wait_duration(
                self._config.z_enable_wait_seconds,
                xy_controller=xy_controller,
            )
            while not xy_controller.is_ready():
                xy_controller.process()
                if xy_controller.state == XYState.FAULT:
                    raise RuntimeError(
                        f"XY 使能故障：{xy_controller.fault.name}"
                    )
                self._sleep(0.001)

            self._transition(MotionLifecycleState.INITIAL_Z_RAISING)
            self._move_z_to(
                self._config.z_up_degrees,
                transport,
                xy_controller,
            )
            self._initialize_rotation_zero(rotation, xy_controller)
            self._transition(MotionLifecycleState.EXECUTING)
            for index, cycle in enumerate(cycles):
                next_cycle = (
                    cycles[index + 1]
                    if index + 1 < len(cycles)
                    else None
                )
                self._execute_cycle(
                    cycle,
                    transport,
                    magnet,
                    xy_controller,
                    rotation,
                    pick_positioned=index > 0,
                    next_cycle=next_cycle,
                )
                self._completed_cycles += 1

            self._signal_completion(
                transport,
                magnet,
                xy_controller,
            )

            if (
                not math.isclose(
                    self._presumed_z_degrees,
                    self._config.z_up_degrees,
                    abs_tol=1e-9,
                )
                or not xy_controller.is_ready()
                or self._circular_error(
                    rotation.current_angle_deg(),
                    0.0,
                )
                > 1.0
            ):
                raise RuntimeError(
                    "结束状态不满足 Z 抬起、XY READY 且旋转轴归零"
                )
            magnet.set_enabled(False)
            self._transition(MotionLifecycleState.FINISH)
            plan_finished = True

            self._transition(MotionLifecycleState.SHUTTING_DOWN)
            self._send_commands(
                transport,
                MksAxisController.plan_power("xyz", enabled=False),
            )
            motors_enabled = False
            rotation.close()
            rotation = None
            transport.close()
            transport = None
            magnet.set_enabled(False)
            magnet.close()
            magnet = None
            self._transition(MotionLifecycleState.EXITED)
            return ActuationResult(
                state=self._state,
                completed_cycles=self._completed_cycles,
                state_history=self.state_history,
            )
        except BaseException as error:
            if not plan_finished:
                self._transition(
                    MotionLifecycleState.ABORTED
                    if isinstance(error, KeyboardInterrupt)
                    else MotionLifecycleState.FAILED
                )
            cleanup_errors = self._emergency_cleanup(
                transport,
                magnet,
                rotation,
                motors_enabled=motors_enabled,
            )
            cleanup_text = (
                ""
                if not cleanup_errors
                else f"；任务安全清理失败：{'；'.join(cleanup_errors)}"
            )
            if isinstance(error, KeyboardInterrupt):
                raise
            if plan_finished:
                self._state = MotionLifecycleState.FINISH
                raise ActuationShutdownError(
                    "MovePlan 已执行完成，但安全退出失败："
                    f"{error}{cleanup_text}"
                ) from error
            if isinstance(error, ActuationValidationError):
                raise
            if isinstance(error, ActuationExecutionError):
                raise ActuationExecutionError(
                    f"{error}{cleanup_text}"
                ) from error
            raise ActuationExecutionError(
                f"电机运动失败：{error}{cleanup_text}"
            ) from error
