"""复刻当前 MKS 固件的 E3 上升、按键触发返回行为。

该脚本依赖直连 Servo42C 的真实串口和机械机构，
不属于默认单元测试。默认打开串口并执行运动；独立运行：
``poetry run python tests/external/motion/replay_current_behavior.py --help``
"""

from __future__ import annotations

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
    MotionExecution,
    MotionSerialConfig,
    MksAxisController,
    Servo42CSerial,
)
from tests.external.motion.helpers import (
    SerialWriteEventReporter,
    create_serial_config,
    dry_execution,
    execution_summary,
    run_with_cli_errors,
)


load_env_file(PROJECT_ROOT / ".env")

DEFAULT_ANGLE_DEGREES = 90.0
DEFAULT_ENABLE_SETTLE_SECONDS = 0.1

app = typer.Typer(
    add_completion=False,
    help="复刻 E3 向上 90°、按 Enter 后返回并失能的硬件行为。",
)


def run_replay(
    config: MotionSerialConfig,
    *,
    execute: bool = True,
    angle_degrees: float = DEFAULT_ANGLE_DEGREES,
    speed: int = DEFAULT_MOVE_SPEED,
    enable_settle_seconds: float = DEFAULT_ENABLE_SETTLE_SECONDS,
    prompt: Callable[[str], str] = input,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """执行或预览 E3 往返流程，并返回结构化诊断摘要。"""

    if not isinstance(config, MotionSerialConfig):
        raise TypeError("config 必须是 MotionSerialConfig")
    if not isinstance(execute, bool):
        raise TypeError("execute 必须是 bool")
    if isinstance(enable_settle_seconds, bool) or not isinstance(
        enable_settle_seconds,
        (int, float),
    ):
        raise TypeError("enable_settle_seconds 必须是实数")
    if enable_settle_seconds < 0:
        raise ValueError("enable_settle_seconds 不能小于 0")

    upward_commands = MksAxisController.plan_relative(
        z_degrees=angle_degrees,
        speed=speed,
    )
    return_commands = MksAxisController.plan_relative(
        z_degrees=-angle_degrees,
        speed=speed,
    )
    requested = {
        "axis": "z",
        "angle_degrees": angle_degrees,
        "speed": speed,
        "enable_before": True,
        "disable_after": True,
        "enable_settle_seconds": enable_settle_seconds,
        "return_trigger": "Enter" if execute else "dry-run 不等待输入",
    }

    if not execute:
        executions = (
            dry_execution(upward_commands),
            dry_execution(return_commands),
        )
        return execution_summary(
            operation="replay-current-behavior",
            execute=False,
            config=config,
            executions=executions,
            requested=requested,
        )

    executions_list: list[MotionExecution] = []
    with SerialWriteEventReporter() as write_reporter:
        with Servo42CSerial(
            config,
            write_observer=write_reporter.publish,
        ) as transport:
            controller = MksAxisController(transport)
            executions_list.append(controller.enable_axes("z"))
            sleep(float(enable_settle_seconds))
            executions_list.append(
                controller.move_relative(
                    z_degrees=angle_degrees,
                    speed=speed,
                )
            )
            prompt("Z 轴已到上方位置，按 Enter 返回 0°：")
            executions_list.append(
                controller.move_relative(
                    z_degrees=-angle_degrees,
                    speed=speed,
                )
            )
            executions_list.append(controller.disable_axes("z"))

    summary = execution_summary(
        operation="replay-current-behavior",
        execute=True,
        config=config,
        executions=tuple(executions_list),
        requested=requested,
    )
    return summary


@app.command()
def main(
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
        help="串口波特率，当前电机默认使用 115200。",
    ),
    read_timeout_seconds: float = typer.Option(
        DEFAULT_MOTION_READ_TIMEOUT_SECONDS,
        "--read-timeout",
        envvar="MOTION_SERIAL_READ_TIMEOUT_SECONDS",
        min=0.001,
        help="单次底层串口读取超时，单位秒。",
    ),
    write_timeout_seconds: float = typer.Option(
        DEFAULT_MOTION_WRITE_TIMEOUT_SECONDS,
        "--write-timeout",
        envvar="MOTION_SERIAL_WRITE_TIMEOUT_SECONDS",
        min=0.001,
        help="单次底层串口写入超时，单位秒。",
    ),
    reply_timeout_seconds: float = typer.Option(
        DEFAULT_MOTION_REPLY_TIMEOUT_SECONDS,
        "--reply-timeout",
        envvar="MOTION_SERIAL_REPLY_TIMEOUT_SECONDS",
        min=0.001,
        help="等待使能或开始回包的超时，单位秒。",
    ),
    angle_degrees: float = typer.Option(
        DEFAULT_ANGLE_DEGREES,
        "--angle",
        min=0.000001,
        max=90.0,
        help="Z 轴向上和返回的机械角度，当前安全上限为 90°。",
    ),
    speed: int = typer.Option(
        DEFAULT_MOVE_SPEED,
        "--speed",
        min=1,
        max=127,
        help="Servo42C 低七位速度档。",
    ),
    enable_settle_seconds: float = typer.Option(
        DEFAULT_ENABLE_SETTLE_SECONDS,
        "--enable-settle",
        min=0.0,
        help="使能成功后开始移动前的稳定等待时间，单位秒。",
    ),
    execute: bool = typer.Option(
        True,
        "--execute/--dry-run",
        help="默认真实使能和运动；传入 --dry-run 仅打印报文。",
    ),
) -> None:
    """收集参数并安全执行或预览当前 E3 往返行为。"""

    def callback() -> dict[str, object]:
        """构造强类型配置并调用可测试的复刻流程。"""

        config = create_serial_config(
            device_path,
            baudrate,
            read_timeout_seconds,
            write_timeout_seconds,
            reply_timeout_seconds,
        )
        return run_replay(
            config,
            execute=execute,
            angle_degrees=angle_degrees,
            speed=speed,
            enable_settle_seconds=enable_settle_seconds,
        )

    run_with_cli_errors(callback)


if __name__ == "__main__":
    app()
