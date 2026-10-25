"""封装已验收 F32C 驱动板的单旋转轴，且不改写其底层控制逻辑。

本模块只负责项目层的配置、生命周期和语义明确的旋转轴入口。实际 UART、
F32C 协议、安全使能、反馈重试、到位确认及异常失能均直接委托给正式
``motion.f32c_controller`` 实现。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Protocol, runtime_checkable

from ec_26_e.motion.f32c_controller import F32CMotor


DEFAULT_FOC_PORT = "/dev/ttyAS3"
DEFAULT_FOC_ADDRESS = 2
DEFAULT_FOC_RPM = 15
DEFAULT_FOC_MAX_RPM = 30
DEFAULT_FOC_FEEDBACK_TIMEOUT_SECONDS = 0.3
DEFAULT_FOC_FEEDBACK_ATTEMPTS = 3
DEFAULT_FOC_CLOCKWISE_MECHANICAL_SIGN = 1


def _environment_text(
    environ: Mapping[str, str],
    name: str,
    default: str,
) -> str:
    """从环境映射读取非空文本，拒绝非字符串部署配置。"""

    value = environ.get(name, default)
    if not isinstance(value, str):
        raise TypeError(f"环境变量 {name} 必须是字符串")
    text = value.strip()
    if not text:
        raise ValueError(f"环境变量 {name} 不能为空")
    return text


def _environment_int(
    environ: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    """严格读取十进制整数，避免布尔值或小数被隐式接受。"""

    text = _environment_text(environ, name, str(default))
    if not text.lstrip("+-").isdigit():
        raise ValueError(f"环境变量 {name} 必须是十进制整数")
    return int(text)


def _environment_float(
    environ: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    """读取有限浮点数，并把范围校验交给冻结配置对象。"""

    text = _environment_text(environ, name, str(default))
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"环境变量 {name} 必须是实数") from error
    if not math.isfinite(value):
        raise ValueError(f"环境变量 {name} 必须是有限数")
    return value


@dataclass(frozen=True, slots=True)
class FocRotationConfig:
    """描述 F32C 旋转轴的部署参数，默认对应 Cubie A7Z UART3。"""

    port: str = DEFAULT_FOC_PORT
    address: int = DEFAULT_FOC_ADDRESS
    rpm: int = DEFAULT_FOC_RPM
    max_rpm: int = DEFAULT_FOC_MAX_RPM
    feedback_timeout_seconds: float = DEFAULT_FOC_FEEDBACK_TIMEOUT_SECONDS
    feedback_attempts: int = DEFAULT_FOC_FEEDBACK_ATTEMPTS
    clockwise_mechanical_sign: int = DEFAULT_FOC_CLOCKWISE_MECHANICAL_SIGN

    def __post_init__(self) -> None:
        """在打开 UART 前校验所有 F32C 部署参数。"""

        if not isinstance(self.port, str) or not self.port.strip():
            raise ValueError("FOC 串口不能为空")
        for name, value, minimum, maximum in (
            ("address", self.address, 0, 255),
            ("rpm", self.rpm, 1, 1000),
            ("max_rpm", self.max_rpm, 1, 1000),
            ("feedback_attempts", self.feedback_attempts, 1, 10),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} 必须是整数")
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} 必须位于 [{minimum}, {maximum}]")
        if self.rpm > self.max_rpm:
            raise ValueError("rpm 不能大于 max_rpm")
        if (
            isinstance(self.clockwise_mechanical_sign, bool)
            or not isinstance(self.clockwise_mechanical_sign, int)
            or self.clockwise_mechanical_sign not in (-1, 1)
        ):
            raise ValueError("clockwise_mechanical_sign 必须是 1 或 -1")
        if (
            isinstance(self.feedback_timeout_seconds, bool)
            or not isinstance(self.feedback_timeout_seconds, (int, float))
            or not math.isfinite(self.feedback_timeout_seconds)
            or self.feedback_timeout_seconds <= 0
        ):
            raise ValueError("feedback_timeout_seconds 必须是有限正数")

    @classmethod
    def from_environ(
        cls,
        environ: Mapping[str, str],
    ) -> FocRotationConfig:
        """从已加载的 ``.env`` 环境映射构造强类型旋转轴配置。"""

        if not isinstance(environ, Mapping):
            raise TypeError("environ 必须是字符串映射")
        return cls(
            port=_environment_text(
                environ,
                "FOC_SERIAL_PORT",
                DEFAULT_FOC_PORT,
            ),
            address=_environment_int(
                environ,
                "FOC_ADDRESS",
                DEFAULT_FOC_ADDRESS,
            ),
            rpm=_environment_int(environ, "FOC_RPM", DEFAULT_FOC_RPM),
            max_rpm=_environment_int(
                environ,
                "FOC_MAX_RPM",
                DEFAULT_FOC_MAX_RPM,
            ),
            feedback_timeout_seconds=_environment_float(
                environ,
                "FOC_FEEDBACK_TIMEOUT_SECONDS",
                DEFAULT_FOC_FEEDBACK_TIMEOUT_SECONDS,
            ),
            feedback_attempts=_environment_int(
                environ,
                "FOC_FEEDBACK_ATTEMPTS",
                DEFAULT_FOC_FEEDBACK_ATTEMPTS,
            ),
            clockwise_mechanical_sign=_environment_int(
                environ,
                "MOTION_ROTATION_CLOCKWISE_SIGN",
                DEFAULT_FOC_CLOCKWISE_MECHANICAL_SIGN,
            ),
        )


@runtime_checkable
class F32CMotorPort(Protocol):
    """约束已验收 F32C 实现对旋转轴公开的最小同步接口。"""

    @property
    def is_connected(self) -> bool: ...

    @property
    def is_enabled(self) -> bool: ...

    def connect(self, verify: bool = False) -> None: ...

    def enable(self, rpm: int = DEFAULT_FOC_RPM) -> None: ...

    def disable(self) -> None: ...

    def get_angle(self) -> float: ...

    def move_clockwise_from_zero(
        self,
        target_degrees: float,
        rpm: int | None = None,
        wait: bool = True,
        timeout: float | None = None,
    ) -> float: ...

    def return_counterclockwise_to_zero(
        self,
        target_degrees: float,
        rpm: int | None = None,
        wait: bool = True,
        timeout: float | None = None,
    ) -> float: ...

    def rotate_one_turn(
        self,
        direction: int = 1,
        rpm: int | None = None,
        wait: bool = True,
        timeout: float | None = None,
    ) -> float: ...

    def close(self, disable: bool = True) -> None: ...


def _reference_motor_class() -> type[F32CMotorPort]:
    """返回已纳入正式 motion 包且不会在导入时接触硬件的控制器。"""

    return F32CMotor


class FocRotationAxis:
    """以持久机械零点为公开坐标的 F32C 旋转轴生命周期封装。"""

    def __init__(
        self,
        config: FocRotationConfig = FocRotationConfig(),
        *,
        motor: F32CMotorPort | None = None,
    ) -> None:
        """保存配置；未注入电机时才惰性采用已验收参考实现。"""

        if not isinstance(config, FocRotationConfig):
            raise TypeError("config 必须是 FocRotationConfig")
        self._config = config
        self._motor = motor

    @property
    def config(self) -> FocRotationConfig:
        """返回不可变的有效部署配置。"""

        return self._config

    @property
    def is_connected(self) -> bool:
        """返回参考 F32C 控制器当前的串口连接状态。"""

        return self._motor is not None and self._motor.is_connected

    @property
    def is_enabled(self) -> bool:
        """返回参考 F32C 控制器当前的使能状态。"""

        return self._motor is not None and self._motor.is_enabled

    def open(self) -> None:
        """打开 UART3 并安全使能保持当前位置，不主动改变旋转角度。"""

        motor = self._ensure_motor()
        if motor.is_connected:
            raise RuntimeError("旋转轴 UART 已打开")
        motor.connect(verify=False)
        try:
            motor.enable(rpm=self._config.rpm)
        except Exception:
            motor.close(disable=True)
            raise

    def close(self) -> None:
        """按参考实现先失能，再关闭 F32C 串口；重复调用安全。"""

        if self._motor is not None:
            self._motor.close(disable=True)

    def __enter__(self) -> FocRotationAxis:
        """进入上下文时安全打开并使能旋转轴。"""

        self.open()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        """无论业务成功或异常，都让参考实现失能并释放 UART。"""

        self.close()

    def current_angle_deg(self) -> float:
        """读取持久机械单圈角度，范围和反馈策略由参考实现保证。"""

        return self._require_enabled().get_angle()

    def move_clockwise_from_zero(
        self,
        target_degrees: float,
        *,
        rpm: int | None = None,
        wait: bool = True,
        timeout: float | None = None,
    ) -> float:
        """从机械零点按顺时针完整路径旋转 ``[0, 360]`` 度。"""

        return self._require_enabled().move_clockwise_from_zero(
            target_degrees,
            rpm=rpm,
            wait=wait,
            timeout=timeout,
        )

    def return_counterclockwise_to_zero(
        self,
        target_degrees: float,
        *,
        rpm: int | None = None,
        wait: bool = True,
        timeout: float | None = None,
    ) -> float:
        """从定向目标按逆时针沿原角度路径返回机械零点。"""

        return self._require_enabled().return_counterclockwise_to_zero(
            target_degrees,
            rpm=rpm,
            wait=wait,
            timeout=timeout,
        )

    def rotate_one_turn(
        self,
        direction: int = 1,
        *,
        rpm: int | None = None,
        wait: bool = True,
        timeout: float | None = None,
    ) -> float:
        """按参考实现的协议正负方向完整旋转一圈。"""

        return self._require_enabled().rotate_one_turn(
            direction=direction,
            rpm=rpm,
            wait=wait,
            timeout=timeout,
        )

    def _ensure_motor(self) -> F32CMotorPort:
        """首次真实使用时原样构造参考 F32C 控制器。"""

        if self._motor is None:
            motor_class = _reference_motor_class()
            self._motor = motor_class(
                port=self._config.port,
                address=self._config.address,
                max_rpm=self._config.max_rpm,
                feedback_timeout=self._config.feedback_timeout_seconds,
                feedback_attempts=self._config.feedback_attempts,
                clockwise_mechanical_sign=(
                    self._config.clockwise_mechanical_sign
                ),
            )
        return self._motor

    def _require_enabled(self) -> F32CMotorPort:
        """拒绝未安全使能时的业务运动或反馈请求。"""

        motor = self._ensure_motor()
        if not motor.is_connected or not motor.is_enabled:
            raise RuntimeError("旋转轴未打开并安全使能；请先调用 open()")
        return motor
