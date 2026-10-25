"""通过命令行运行 Python 复刻 XY 状态机或独立控制 Servo42C Z 轴。

该脚本依赖真实串口和机械机构，不属于默认单元测试。
全部命令默认打开串口并发送报文；独立运行：
``poetry run python tests/external/motion/control_axes.py --help``
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Callable
from pathlib import Path
import sys
import time

import typer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ec_26_e.config import load_env_file
from ec_26_e.motion import (
    DEFAULT_MOTION_BAUDRATE,
    DEFAULT_MOTION_DEVICE_PATH,
    DEFAULT_MOTION_READ_TIMEOUT_SECONDS,
    DEFAULT_MOTION_REPLY_TIMEOUT_SECONDS,
    DEFAULT_MOTION_WRITE_TIMEOUT_SECONDS,
    DEFAULT_MOVE_SPEED,
    MksXYController,
    MotionExecution,
    MotionSerialConfig,
    MksAxisController,
    Servo42CSerial,
    XYResult,
    XYState,
    XY_X_MAX_0P01MM,
    XY_Y_MIN_0P01MM,
    XY_Y_MAX_0P01MM,
    mm_to_0p01mm,
)
from tests.external.motion.helpers import (
    SerialWriteEventReporter,
    create_serial_config,
    dry_execution,
    execution_summary,
    run_with_cli_errors,
)


load_env_file(PROJECT_ROOT / ".env")

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="执行 Python 复刻的 XY 绝对移动，或独立预览/控制 Z。",
)


@dataclass(frozen=True, slots=True)
class MotionCliContext:
    """保存通用串口配置和本次是否允许真实执行。"""

    config: MotionSerialConfig
    execute: bool


def _context(value: object) -> MotionCliContext:
    """从 Typer 上下文取回强类型 motion 公共选项。"""

    if not isinstance(value, MotionCliContext):
        raise RuntimeError("motion CLI 上下文尚未初始化")
    return value


def _run_simple(
    context: MotionCliContext,
    operation: str,
    axes: str,
) -> dict[str, object]:
    """执行或预览使能、失能或停止命令。"""

    normalized_axes = "".join(MksAxisController.normalize_axes(axes))
    if "x" in normalized_axes or "y" in normalized_axes:
        raise ValueError(
            "X/Y 禁止绕过 C 状态机执行普通命令；该入口仅允许 Z"
        )
    if operation == "enable":
        commands = MksAxisController.plan_power(
            normalized_axes,
            enabled=True,
        )
    elif operation == "disable":
        commands = MksAxisController.plan_power(
            normalized_axes,
            enabled=False,
        )
    elif operation == "stop":
        commands = MksAxisController.plan_stop(normalized_axes)
    else:
        raise ValueError(f"未知操作：{operation}")

    if not context.execute:
        execution = dry_execution(commands)
    else:
        with SerialWriteEventReporter() as write_reporter:
            with Servo42CSerial(
                context.config,
                write_observer=write_reporter.publish,
            ) as transport:
                controller = MksAxisController(transport)
                if operation == "enable":
                    execution = controller.enable_axes(normalized_axes)
                elif operation == "disable":
                    execution = controller.disable_axes(normalized_axes)
                else:
                    execution = controller.stop_axes(normalized_axes)

    return execution_summary(
        operation=operation,
        execute=context.execute,
        config=context.config,
        executions=(execution,),
        requested={"axes": normalized_axes},
    )


def run_move(
    context: MotionCliContext,
    *,
    x_mm: float | None = None,
    y_mm: float | None = None,
    z_degrees: float = 0.0,
    speed: int = DEFAULT_MOVE_SPEED,
    enable_before: bool = True,
    disable_after: bool = True,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """执行 Python 复刻的 XY 绝对移动，并可随后独立执行 Z。"""

    if not callable(sleep) or not callable(monotonic):
        raise TypeError("sleep 和 monotonic 必须可调用")
    xy_requested = x_mm is not None or y_mm is not None
    x_units = mm_to_0p01mm("x_mm", 0.0 if x_mm is None else x_mm)
    y_units = mm_to_0p01mm("y_mm", 0.0 if y_mm is None else y_mm)
    if not 0 <= x_units <= XY_X_MAX_0P01MM:
        raise ValueError("x_mm 必须位于 [0, 230] 的绝对坐标范围")
    if not XY_Y_MIN_0P01MM <= y_units <= XY_Y_MAX_0P01MM:
        raise ValueError("y_mm 必须位于 [-10, 360] 的绝对坐标范围")

    xy_move_commands = (
        tuple(
            replace(command, expected_statuses=())
            for command in MksAxisController.plan_relative(
                x_mm=x_units / 100.0,
                y_mm=y_units / 100.0,
                speed=64,
            )
        )
        if xy_requested
        else ()
    )
    xy_enable_commands = (
        tuple(
            replace(command, expected_statuses=())
            for command in MksAxisController.plan_power(
                "xy",
                enabled=True,
            )
        )
        if xy_requested
        else ()
    )
    z_move_commands = MksAxisController.plan_relative(
        z_degrees=z_degrees,
        speed=speed,
    )
    if any(command.axis != "z" for command in z_move_commands):
        raise RuntimeError("Z 规划不得包含 XY 命令")
    requested = {
        "x_mm": x_mm,
        "y_mm": y_mm,
        "xy_coordinate_mode": "absolute_0p01mm",
        "xy_target_0p01mm": {"x": x_units, "y": y_units},
        "xy_require_motor_reply": False,
        "xy_power_delay_ms": 500,
        "xy_leave_enabled_after_success": xy_requested,
        "z_degrees": z_degrees,
        "speed": speed,
        "z_enable_before": enable_before,
        "z_disable_after": disable_after,
    }
    if not xy_requested and not z_move_commands:
        return execution_summary(
            operation="move",
            execute=context.execute,
            config=context.config,
            executions=(),
            requested=requested,
            no_op=True,
        )

    if not context.execute:
        executions: list[MotionExecution] = []
        if xy_requested:
            executions.append(dry_execution(xy_enable_commands))
            if xy_move_commands:
                executions.append(dry_execution(xy_move_commands))
        if z_move_commands and enable_before:
            executions.append(
                dry_execution(
                    MksAxisController.plan_power(
                        "z",
                        enabled=True,
                    )
                )
            )
        if z_move_commands:
            executions.append(dry_execution(z_move_commands))
        if z_move_commands and disable_after:
            executions.append(
                dry_execution(
                    MksAxisController.plan_power(
                        "z",
                        enabled=False,
                    )
                )
            )
        return execution_summary(
            operation="move",
            execute=False,
            config=context.config,
            executions=tuple(executions),
            requested=requested,
        )

    executions = []
    with SerialWriteEventReporter() as write_reporter:
        with Servo42CSerial(
            context.config,
            write_observer=write_reporter.publish,
        ) as transport:
            if xy_requested:
                xy_controller = MksXYController(
                    transport,
                    monotonic=monotonic,
                )
                xy_controller.initialize()
                while xy_controller.state != XYState.READY:
                    xy_controller.process()
                    if xy_controller.state == XYState.FAULT:
                        raise RuntimeError(
                            f"XY 初始化故障：{xy_controller.fault.name}"
                        )
                    if xy_controller.state != XYState.READY:
                        sleep(0.001)
                executions.append(dry_execution(xy_enable_commands))
                result = xy_controller.move_to(x_units, y_units)
                if result != XYResult.OK:
                    raise RuntimeError(f"XY_MoveTo 返回 {result.name}")
                if xy_move_commands:
                    executions.append(dry_execution(xy_move_commands))
                while xy_controller.state != XYState.READY:
                    xy_controller.process()
                    if xy_controller.state == XYState.FAULT:
                        raise RuntimeError(
                            f"XY 移动故障：{xy_controller.fault.name}"
                        )
                    if xy_controller.state != XYState.READY:
                        sleep(0.001)

            if z_move_commands:
                z_controller = MksAxisController(transport)
                if enable_before:
                    executions.append(z_controller.enable_axes("z"))
                executions.append(
                    z_controller.move_relative(
                        z_degrees=z_degrees,
                        speed=speed,
                    )
                )
                if disable_after:
                    executions.append(z_controller.disable_axes("z"))

    return execution_summary(
        operation="move",
        execute=True,
        config=context.config,
        executions=tuple(executions),
        requested=requested,
    )


@app.callback()
def configure(
    ctx: typer.Context,
    device_path: Path = typer.Option(
        DEFAULT_MOTION_DEVICE_PATH,
        "--device",
        envvar="MOTION_SERIAL_DEVICE",
        help="Servo42C 串口设备；可由 MOTION_SERIAL_DEVICE 配置。",
    ),
    baudrate: int = typer.Option(
        DEFAULT_MOTION_BAUDRATE,
        "--baudrate",
        envvar="MOTION_SERIAL_BAUDRATE",
        min=1,
        help="串口波特率，当前默认使用 115200。",
    ),
    read_timeout_seconds: float = typer.Option(
        DEFAULT_MOTION_READ_TIMEOUT_SECONDS,
        "--read-timeout",
        envvar="MOTION_SERIAL_READ_TIMEOUT_SECONDS",
        min=0.001,
        help="单次底层读取超时，单位秒。",
    ),
    write_timeout_seconds: float = typer.Option(
        DEFAULT_MOTION_WRITE_TIMEOUT_SECONDS,
        "--write-timeout",
        envvar="MOTION_SERIAL_WRITE_TIMEOUT_SECONDS",
        min=0.001,
        help="单次底层写入超时，单位秒。",
    ),
    reply_timeout_seconds: float = typer.Option(
        DEFAULT_MOTION_REPLY_TIMEOUT_SECONDS,
        "--reply-timeout",
        envvar="MOTION_SERIAL_REPLY_TIMEOUT_SECONDS",
        min=0.001,
        help="等待使能或开始回包的超时，单位秒。",
    ),
    execute: bool = typer.Option(
        True,
        "--execute/--dry-run",
        help="默认执行真实串口动作；传入 --dry-run 只输出报文。",
    ),
) -> None:
    """校验通用串口参数，但不在此阶段打开设备。"""

    ctx.obj = MotionCliContext(
        config=create_serial_config(
            device_path,
            baudrate,
            read_timeout_seconds,
            write_timeout_seconds,
            reply_timeout_seconds,
        ),
        execute=execute,
    )


@app.command("move")
def move_command(
    ctx: typer.Context,
    x_mm: float | None = typer.Option(
        None,
        "--x",
        min=0.0,
        max=230.0,
        help="X 绝对目标坐标，单位 mm，精度 0.01 mm；未传则为 0。",
    ),
    y_mm: float | None = typer.Option(
        None,
        "--y",
        min=0.0,
        max=360.0,
        help="Y 绝对目标坐标，单位 mm，精度 0.01 mm；未传则为 0。",
    ),
    z_degrees: float = typer.Option(
        0.0,
        "--z",
        min=-90.0,
        max=90.0,
        help="Z 相对机械角度；正值向上，当前安全上限为 90°。",
    ),
    speed: int = typer.Option(
        DEFAULT_MOVE_SPEED,
        "--speed",
        min=1,
        max=127,
        help="Servo42C 低七位速度档。",
    ),
    enable_before: bool = typer.Option(
        True,
        "--enable-before/--no-enable-before",
        help="仅控制 Z：移动前是否使能 E3；XY 始终按 C 初始化流程。",
    ),
    disable_after: bool = typer.Option(
        True,
        "--disable-after/--keep-enabled",
        help="仅控制 Z：成功后是否失能 E3；XY 按 C 保持使能。",
    ),
) -> None:
    """按 Python 复刻逻辑执行 XY，并可随后独立执行 Z。"""

    run_with_cli_errors(
        lambda: run_move(
            _context(ctx.obj),
            x_mm=x_mm,
            y_mm=y_mm,
            z_degrees=z_degrees,
            speed=speed,
            enable_before=enable_before,
            disable_after=disable_after,
        )
    )


@app.command("enable")
def enable_command(
    ctx: typer.Context,
    axes: str = typer.Option(
        "z",
        "--axes",
        help="仅允许 z；XY 由 Python 复刻状态机统一完成上电使能。",
    ),
) -> None:
    """使能或预览使能指定轴。"""

    run_with_cli_errors(
        lambda: _run_simple(_context(ctx.obj), "enable", axes)
    )


@app.command("disable")
def disable_command(
    ctx: typer.Context,
    axes: str = typer.Option(
        "z",
        "--axes",
        help="仅允许 z；C 的正常 XY 生命周期不会在移动后失能。",
    ),
) -> None:
    """失能或预览失能指定轴。"""

    run_with_cli_errors(
        lambda: _run_simple(_context(ctx.obj), "disable", axes)
    )


@app.command("stop")
def stop_command(
    ctx: typer.Context,
    axes: str = typer.Option(
        "z",
        "--axes",
        help="仅允许 z；XY 停止必须通过 C 状态机的 emergency_stop。",
    ),
) -> None:
    """停止或预览停止指定轴。"""

    run_with_cli_errors(
        lambda: _run_simple(_context(ctx.obj), "stop", axes)
    )


if __name__ == "__main__":
    app()
