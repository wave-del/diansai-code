"""通过 Python 复刻 XY 状态机连续执行多个 X 绝对目标。

该脚本依赖真实串口和机械机构，不属于默认单元测试。它只负责依次请求
绝对目标，所有使能、方向、脉冲、等待和位置提交均由 ``MksXYController``
执行。独立运行：
``poetry run python tests/external/motion/repeat_x_move.py --help``
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
    MksXYController,
    MotionSerialConfig,
    Servo42CSerial,
    XYResult,
    XYState,
    XY_X_MAX_0P01MM,
    mm_to_0p01mm,
)
from tests.external.motion.helpers import (
    SerialWriteEventReporter,
    create_serial_config,
    execution_summary,
    run_with_cli_errors,
)


load_env_file(PROJECT_ROOT / ".env")

DEFAULT_DISTANCE_MM = 15.0
DEFAULT_REPEAT_COUNT = 4

app = typer.Typer(
    add_completion=False,
    help="使用 Python 复刻 XY 状态机依次移动到多个 X 绝对目标。",
)


def _drive_until_ready(
    controller: MksXYController,
    *,
    sleep: Callable[[float], None],
) -> None:
    """循环推进非阻塞状态机，直到 READY 或明确故障。"""

    while controller.state != XYState.READY:
        controller.process()
        if controller.state == XYState.FAULT:
            raise RuntimeError(f"XY 状态机故障：{controller.fault.name}")
        if controller.state != XYState.READY:
            sleep(0.001)


def run_repeat_x(
    config: MotionSerialConfig,
    *,
    execute: bool = True,
    distance_mm: float = DEFAULT_DISTANCE_MM,
    repeat_count: int = DEFAULT_REPEAT_COUNT,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """初始化一次 XY 状态机并依次请求递增的 X 绝对目标。"""

    if not isinstance(config, MotionSerialConfig):
        raise TypeError("config 必须是 MotionSerialConfig")
    if not isinstance(execute, bool):
        raise TypeError("execute 必须是 bool")
    if isinstance(repeat_count, bool) or not isinstance(repeat_count, int):
        raise TypeError("repeat_count 必须是整数")
    if repeat_count <= 0:
        raise ValueError("repeat_count 必须大于 0")
    if not callable(sleep) or not callable(monotonic):
        raise TypeError("sleep 和 monotonic 必须可调用")

    step_units = mm_to_0p01mm("distance_mm", distance_mm)
    if step_units <= 0:
        raise ValueError("distance_mm 必须大于 0")
    targets = tuple(
        step_units * index
        for index in range(1, repeat_count + 1)
    )
    if targets[-1] > XY_X_MAX_0P01MM:
        raise ValueError("最后一个 X 绝对目标不能大于 230 mm")

    requested = {
        "axis": "x",
        "coordinate_mode": "absolute_0p01mm",
        "distance_mm_each": float(distance_mm),
        "repeat_count": repeat_count,
        "targets_0p01mm": list(targets),
        "targets_mm": [target / 100.0 for target in targets],
        "xy_require_motor_reply": False,
        "xy_leave_enabled_after_success": True,
    }
    if not execute:
        return execution_summary(
            operation="repeat-x-move",
            execute=False,
            config=config,
            executions=(),
            requested=requested,
        )

    with SerialWriteEventReporter() as write_reporter:
        with Servo42CSerial(
            config,
            write_observer=write_reporter.publish,
        ) as transport:
            controller = MksXYController(
                transport,
                monotonic=monotonic,
            )
            controller.initialize()
            _drive_until_ready(controller, sleep=sleep)
            for target in targets:
                result = controller.move_to(target, 0)
                if result != XYResult.OK:
                    raise RuntimeError(f"XY_MoveTo 返回 {result.name}")
                _drive_until_ready(controller, sleep=sleep)

    return execution_summary(
        operation="repeat-x-move",
        execute=True,
        config=config,
        executions=(),
        requested=requested,
    )


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
        help="串口波特率，当前默认使用 115200。",
    ),
    read_timeout_seconds: float = typer.Option(
        DEFAULT_MOTION_READ_TIMEOUT_SECONDS,
        "--read-timeout",
        envvar="MOTION_SERIAL_READ_TIMEOUT_SECONDS",
        min=0.001,
        help="底层单次读取超时，单位秒。",
    ),
    write_timeout_seconds: float = typer.Option(
        DEFAULT_MOTION_WRITE_TIMEOUT_SECONDS,
        "--write-timeout",
        envvar="MOTION_SERIAL_WRITE_TIMEOUT_SECONDS",
        min=0.001,
        help="底层单次写入超时，单位秒。",
    ),
    reply_timeout_seconds: float = typer.Option(
        DEFAULT_MOTION_REPLY_TIMEOUT_SECONDS,
        "--reply-timeout",
        envvar="MOTION_SERIAL_REPLY_TIMEOUT_SECONDS",
        min=0.001,
        help="兼容串口配置；当前 C 默认 XY 路径不等待回包。",
    ),
    distance_mm: float = typer.Option(
        DEFAULT_DISTANCE_MM,
        "--distance",
        min=0.01,
        max=230.0,
        help="相邻 X 绝对目标的增量，单位 mm，精度 0.01 mm。",
    ),
    repeat_count: int = typer.Option(
        DEFAULT_REPEAT_COUNT,
        "--count",
        min=1,
        help="递增绝对目标数量。",
    ),
    execute: bool = typer.Option(
        True,
        "--execute/--dry-run",
        help="默认真实运行 Python XY 状态机；dry-run 仅输出目标。",
    ),
) -> None:
    """收集参数并运行连续 X 绝对目标流程。"""

    def callback() -> dict[str, object]:
        """构造强类型配置并调用可测试流程。"""

        config = create_serial_config(
            device_path,
            baudrate,
            read_timeout_seconds,
            write_timeout_seconds,
            reply_timeout_seconds,
        )
        return run_repeat_x(
            config,
            execute=execute,
            distance_mm=distance_mm,
            repeat_count=repeat_count,
        )

    run_with_cli_errors(callback)


if __name__ == "__main__":
    app()
