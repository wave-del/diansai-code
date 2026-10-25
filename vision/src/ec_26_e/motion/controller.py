"""提供纯报文规划与不属于 C XY 状态机的独立 Z 同步控制。"""

from __future__ import annotations

from collections.abc import Iterable
import math
import time

from ec_26_e.motion.contracts import (
    MotionCommand,
    MotionExecution,
    MotorAddress,
    ServoReply,
    ServoTransport,
)
from ec_26_e.motion.protocol import Servo42CProtocol


DEFAULT_MOVE_SPEED = 64
XY_PULSES_PER_MM = 400
# 右移补偿当前关闭；正 X 与负 X 均按目标距离换算。
X_RIGHT_COMPENSATION_PULSES = 0
Z_PULSES_PER_90_DEGREES = 3200
X_MAX_DISTANCE_MM = 230.0
Y_MAX_DISTANCE_MM = 360.0
Z_MAX_DISTANCE_DEGREES = 90.0
MOVE_TIMEOUT_MARGIN_SECONDS = 0.5


class MksAxisController:
    """保留纯 XYZ 报文规划；真实相对执行只允许独立 Z 轴。"""

    _right_compensation_pending = True

    def __init__(self, transport: ServoTransport) -> None:
        """保存已构造的可替换串口传输对象。"""

        if not isinstance(transport, ServoTransport):
            raise TypeError("transport 必须实现 ServoTransport")
        self._transport = transport

    @staticmethod
    def _validate_distance(name: str, value: float, maximum: float) -> float:
        """校验一个带方向的有限机械位移。"""

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} 必须是实数")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} 必须是有限数")
        if abs(number) > maximum:
            raise ValueError(f"{name} 绝对值不能大于 {maximum}")
        return number

    @staticmethod
    def _validate_speed(speed: int) -> int:
        """校验控制器使用的正速度档。"""

        if isinstance(speed, bool) or not isinstance(speed, int):
            raise TypeError("speed 必须是整数")
        if not 1 <= speed <= 127:
            raise ValueError("speed 必须位于 [1, 127]")
        return speed

    @staticmethod
    def _round_pulses(value: float) -> int:
        """把非负连续脉冲数按 0.5 向上规则转换为整数。"""

        if not math.isfinite(value) or value < 0:
            raise ValueError("待转换脉冲数必须是非负有限数")
        return math.floor(value + 0.5)

    @staticmethod
    def normalize_axes(axes: str | Iterable[str]) -> tuple[str, ...]:
        """规范化轴选择并固定返回 x、y、z 的安全执行顺序。"""

        if isinstance(axes, str):
            raw_axes = tuple(axes.lower())
        else:
            try:
                raw_axes = tuple(str(axis).lower() for axis in axes)
            except TypeError as error:
                raise TypeError(
                    "axes 必须是字符串或字符串可迭代对象"
                ) from error
        if not raw_axes:
            raise ValueError("axes 不能为空")
        unknown = sorted(set(raw_axes) - {"x", "y", "z"})
        if unknown:
            raise ValueError(f"axes 包含未知轴：{', '.join(unknown)}")
        return tuple(axis for axis in ("x", "y", "z") if axis in raw_axes)

    @staticmethod
    def addresses_for_axes(
        axes: str | Iterable[str],
    ) -> tuple[tuple[str, MotorAddress], ...]:
        """把逻辑轴展开为固定顺序的电机地址。"""

        result: list[tuple[str, MotorAddress]] = []
        for axis in MksAxisController.normalize_axes(axes):
            if axis == "x":
                result.extend(
                    (
                        ("x", MotorAddress.X0),
                        ("x", MotorAddress.X1),
                    )
                )
            elif axis == "y":
                result.append(("y", MotorAddress.Y))
            else:
                result.append(("z", MotorAddress.Z))
        return tuple(result)

    @staticmethod
    def completion_timeout_seconds(pulses: int, speed: int) -> float:
        """按固件公式计算保守的移动完成超时。"""

        MksAxisController._validate_speed(speed)
        if isinstance(pulses, bool) or not isinstance(pulses, int):
            raise TypeError("pulses 必须是整数")
        if pulses < 0:
            raise ValueError("pulses 不能小于 0")
        expected_ms = math.ceil((2 * pulses) / speed)
        return (2 * expected_ms) / 1000.0 + MOVE_TIMEOUT_MARGIN_SECONDS

    @classmethod
    def plan_relative(
        cls,
        *,
        x_mm: float = 0.0,
        y_mm: float = 0.0,
        z_degrees: float = 0.0,
        speed: int = DEFAULT_MOVE_SPEED,
    ) -> tuple[MotionCommand, ...]:
        """纯计算生成 X→Y→Z 相对移动命令，不访问串口。"""

        x = cls._validate_distance("x_mm", x_mm, X_MAX_DISTANCE_MM)
        y = cls._validate_distance("y_mm", y_mm, Y_MAX_DISTANCE_MM)
        z = cls._validate_distance(
            "z_degrees",
            z_degrees,
            Z_MAX_DISTANCE_DEGREES,
        )
        speed_value = cls._validate_speed(speed)
        commands: list[MotionCommand] = []

        x_pulses = cls._round_pulses(abs(x) * XY_PULSES_PER_MM)
        if x != 0.0 and x_pulses == 0:
            raise ValueError("x_mm 太小，无法换算为至少一个脉冲")
        if x > 0.0 and cls._right_compensation_pending:
            x_pulses += X_RIGHT_COMPENSATION_PULSES
            cls._right_compensation_pending = False
        if x_pulses:
            # 机械 XY 方向已反转：正 X 对应横杆向右。
            logical_speed = -speed_value if x > 0 else speed_value
            commands.extend(
                (
                    MotionCommand(
                        operation="move",
                        axis="x",
                        address=MotorAddress.X0,
                        frame=Servo42CProtocol.build_move(
                            MotorAddress.X0,
                            -logical_speed,
                            x_pulses,
                        ),
                        pulses=x_pulses,
                    ),
                    MotionCommand(
                        operation="move",
                        axis="x",
                        address=MotorAddress.X1,
                        frame=Servo42CProtocol.build_move(
                            MotorAddress.X1,
                            logical_speed,
                            x_pulses,
                        ),
                        pulses=x_pulses,
                    ),
                )
            )

        y_pulses = cls._round_pulses(abs(y) * XY_PULSES_PER_MM)
        if y != 0.0 and y_pulses == 0:
            raise ValueError("y_mm 太小，无法换算为至少一个脉冲")
        if y_pulses:
            # 机械 XY 方向已反转：正 Y 对应机构向下。
            y_speed = -speed_value if y > 0 else speed_value
            commands.append(
                MotionCommand(
                    operation="move",
                    axis="y",
                    address=MotorAddress.Y,
                    frame=Servo42CProtocol.build_move(
                        MotorAddress.Y,
                        y_speed,
                        y_pulses,
                    ),
                    pulses=y_pulses,
                )
            )

        z_pulses = cls._round_pulses(
            abs(z) * Z_PULSES_PER_90_DEGREES / 90.0
        )
        if z != 0.0 and z_pulses == 0:
            raise ValueError("z_degrees 太小，无法换算为至少一个脉冲")
        if z_pulses:
            # 实机验证：E3 协议正方向为向上；正 Z 语义保持为抬起。
            z_speed = speed_value if z > 0 else -speed_value
            commands.append(
                MotionCommand(
                    operation="move",
                    axis="z",
                    address=MotorAddress.Z,
                    frame=Servo42CProtocol.build_move(
                        MotorAddress.Z,
                        z_speed,
                        z_pulses,
                    ),
                    pulses=z_pulses,
                )
            )

        return tuple(commands)

    @staticmethod
    def plan_power(
        axes: str | Iterable[str] = "xyz",
        *,
        enabled: bool,
    ) -> tuple[MotionCommand, ...]:
        """纯计算生成指定轴的使能或失能命令。"""

        if not isinstance(enabled, bool):
            raise TypeError("enabled 必须是 bool")
        operation = "enable" if enabled else "disable"
        return tuple(
            MotionCommand(
                operation=operation,
                axis=axis,
                address=address,
                frame=Servo42CProtocol.build_enable(address, enabled),
            )
            for axis, address in MksAxisController.addresses_for_axes(axes)
        )

    @staticmethod
    def plan_stop(
        axes: str | Iterable[str] = "xyz",
    ) -> tuple[MotionCommand, ...]:
        """纯计算生成指定轴的停止命令。"""

        return tuple(
            MotionCommand(
                operation="stop",
                axis=axis,
                address=address,
                frame=Servo42CProtocol.build_stop(address),
            )
            for axis, address in MksAxisController.addresses_for_axes(axes)
        )

    def _execute_simple(
        self,
        commands: tuple[MotionCommand, ...],
    ) -> MotionExecution:
        """逐条发送普通命令并要求每个地址返回状态 1。"""

        started_at = time.monotonic()
        replies: list[ServoReply] = []
        for command in commands:
            self._transport.send_frame(command.frame)
            replies.append(
                self._transport.wait_reply(
                    command.address,
                    expected_status=1,
                )
            )
        return MotionExecution(
            commands=commands,
            replies=tuple(replies),
            elapsed_seconds=time.monotonic() - started_at,
        )

    def enable_axes(
        self,
        axes: str | Iterable[str] = "xyz",
    ) -> MotionExecution:
        """按地址顺序使能指定逻辑轴。"""

        return self._execute_simple(self.plan_power(axes, enabled=True))

    def disable_axes(
        self,
        axes: str | Iterable[str] = "xyz",
    ) -> MotionExecution:
        """按地址顺序失能指定逻辑轴。"""

        return self._execute_simple(self.plan_power(axes, enabled=False))

    def stop_axes(
        self,
        axes: str | Iterable[str] = "xyz",
    ) -> MotionExecution:
        """按地址顺序停止指定逻辑轴。"""

        return self._execute_simple(self.plan_stop(axes))

    def move_relative(
        self,
        *,
        x_mm: float = 0.0,
        y_mm: float = 0.0,
        z_degrees: float = 0.0,
        speed: int = DEFAULT_MOVE_SPEED,
    ) -> MotionExecution:
        """执行独立 Z 相对移动；XY 必须交由 Python 复刻状态机。"""

        if x_mm != 0.0 or y_mm != 0.0:
            raise ValueError(
                "X/Y 不允许相对执行；请使用 MksXYController.move_to"
            )

        commands = self.plan_relative(
            x_mm=x_mm,
            y_mm=y_mm,
            z_degrees=z_degrees,
            speed=speed,
        )
        started_at = time.monotonic()
        replies: list[ServoReply] = []

        for axis in ("z",):
            group = tuple(command for command in commands if command.axis == axis)
            if not group:
                continue
            for command in group:
                self._transport.send_frame(command.frame)
            for command in group:
                replies.append(
                    self._transport.wait_reply(
                        command.address,
                        expected_status=1,
                    )
                )

            maximum_pulses = max(command.pulses or 0 for command in group)
            timeout = self.completion_timeout_seconds(maximum_pulses, speed)
            completion_deadline = time.monotonic() + timeout
            for command in group:
                remaining = completion_deadline - time.monotonic()
                replies.append(
                    self._transport.wait_reply(
                        command.address,
                        expected_status=2,
                        timeout_seconds=max(remaining, 1e-6),
                    )
                )

        return MotionExecution(
            commands=commands,
            replies=tuple(replies),
            elapsed_seconds=time.monotonic() - started_at,
        )
