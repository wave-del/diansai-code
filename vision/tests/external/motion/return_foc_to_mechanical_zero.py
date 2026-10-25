"""以 F32C 已保存的机械零点为基准，自动逆时针回零。

脚本先用 ``0x0E/0x02`` 读取持久机械角，而非使用本次上电多圈角。
默认仅查询并预览；只有 ``--execute`` 才会使能旋转轴并沿逆时针方向
回到该持久机械零点。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import typer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ec_26_e.config import load_env_file
from ec_26_e.motion import (
    DEFAULT_FOC_ADDRESS,
    DEFAULT_FOC_FEEDBACK_ATTEMPTS,
    DEFAULT_FOC_FEEDBACK_TIMEOUT_SECONDS,
    DEFAULT_FOC_MAX_RPM,
    DEFAULT_FOC_PORT,
    DEFAULT_FOC_RPM,
    FocRotationAxis,
    FocRotationConfig,
)
from ec_26_e.motion.f32c_controller import F32CMotor


MECHANICAL_ZERO_TOLERANCE_DEGREES = 1.0

load_env_file(PROJECT_ROOT / ".env")

app = typer.Typer(
    add_completion=False,
    help="读取 F32C 持久机械角，并可自动逆时针回到已保存机械零点。",
)


def read_mechanical_angle(config: FocRotationConfig) -> float:
    """仅读取持久机械角，不使能电机也不发送定位目标。"""

    motor = F32CMotor(
        port=config.port,
        address=config.address,
        max_rpm=config.max_rpm,
        feedback_timeout=config.feedback_timeout_seconds,
        feedback_attempts=config.feedback_attempts,
        clockwise_mechanical_sign=config.clockwise_mechanical_sign,
    )
    try:
        motor.connect(verify=False)
        return motor.get_angle()
    finally:
        motor.close(disable=False)


def run_return_to_mechanical_zero(
    config: FocRotationConfig,
    *,
    execute: bool = False,
) -> dict[str, object]:
    """预览或执行从当前持久机械角逆时针回到 0° 的流程。"""

    if not isinstance(config, FocRotationConfig):
        raise TypeError("config 必须是 FocRotationConfig")
    if not isinstance(execute, bool):
        raise TypeError("execute 必须是 bool")

    current_angle = read_mechanical_angle(config)
    already_at_zero = math.isclose(
        current_angle % 360.0,
        0.0,
        abs_tol=MECHANICAL_ZERO_TOLERANCE_DEGREES,
    ) or math.isclose(
        current_angle % 360.0,
        360.0,
        abs_tol=MECHANICAL_ZERO_TOLERANCE_DEGREES,
    )
    summary: dict[str, object] = {
        "operation": "return-foc-to-mechanical-zero",
        "dry_run": not execute,
        "port": config.port,
        "address": config.address,
        "current_persistent_mechanical_angle_deg": current_angle,
        "target_persistent_mechanical_angle_deg": 0.0,
        "direction": "counterclockwise",
        "already_at_zero": already_at_zero,
    }
    if not execute or already_at_zero:
        summary["executed"] = False
        return summary

    axis = FocRotationAxis(config)
    try:
        axis.open()
        # 该 API 会先核对实时持久机械角仍与 current_angle 一致，避免
        # 查询和下发之间轴被移动后按旧角度回零。
        internal_total_target = axis.return_counterclockwise_to_zero(
            current_angle,
            wait=True,
        )
        final_angle = axis.current_angle_deg()
    finally:
        axis.close()

    summary.update(
        {
            "executed": True,
            "internal_total_target_deg": internal_total_target,
            "final_persistent_mechanical_angle_deg": final_angle,
            "closed_with_disable": True,
        }
    )
    return summary


@app.command()
def main(
    execute: bool = typer.Option(
        False,
        "--execute/--dry-run",
        help="默认只读取并预览；--execute 才使能并逆时针回持久机械零点。",
    ),
    port: str = typer.Option(
        DEFAULT_FOC_PORT,
        "--port",
        envvar="FOC_SERIAL_PORT",
        help="F32C 串口设备。",
    ),
    address: int = typer.Option(
        DEFAULT_FOC_ADDRESS,
        "--address",
        envvar="FOC_ADDRESS",
        min=0,
        max=255,
        help="F32C 地址。",
    ),
    rpm: int = typer.Option(
        DEFAULT_FOC_RPM,
        "--rpm",
        envvar="FOC_RPM",
        min=1,
        max=1000,
        help="旋转速度，单位 RPM。",
    ),
    max_rpm: int = typer.Option(
        DEFAULT_FOC_MAX_RPM,
        "--max-rpm",
        envvar="FOC_MAX_RPM",
        min=1,
        max=1000,
        help="允许的最大 RPM。",
    ),
    feedback_timeout_seconds: float = typer.Option(
        DEFAULT_FOC_FEEDBACK_TIMEOUT_SECONDS,
        "--feedback-timeout",
        envvar="FOC_FEEDBACK_TIMEOUT_SECONDS",
        min=0.001,
        help="单次机械角反馈等待秒数。",
    ),
    feedback_attempts: int = typer.Option(
        DEFAULT_FOC_FEEDBACK_ATTEMPTS,
        "--feedback-attempts",
        envvar="FOC_FEEDBACK_ATTEMPTS",
        min=1,
        max=10,
        help="机械角反馈重试次数。",
    ),
) -> None:
    """运行机械零点预览或自动回零。"""

    config = FocRotationConfig(
        port=port,
        address=address,
        rpm=rpm,
        max_rpm=max_rpm,
        feedback_timeout_seconds=feedback_timeout_seconds,
        feedback_attempts=feedback_attempts,
    )
    try:
        summary = run_return_to_mechanical_zero(config, execute=execute)
    except KeyboardInterrupt:
        typer.echo("机械零点回归已中断；旋转轴已请求失能关闭。", err=True)
        raise typer.Exit(code=130) from None
    except Exception as error:
        typer.echo(f"机械零点回归失败：{error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
