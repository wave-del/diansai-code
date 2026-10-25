"""通过命令行测试 F32C 顺时针目标或逆时针原路回零。

该脚本依赖真实 UART3 与机械机构，不属于默认单元测试。除 ``--help`` 外，
必须显式给出 ``--angle``；配合 ``--zero`` 时从该目标逆时针回零。退出时
始终失能电机并关闭串口。

独立运行：
``poetry run python tests/external/motion/control_foc_rotation.py --help``。
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Callable

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


# 与项目其他外部诊断一致：命令行的 envvar 选项可读取仓库根目录 .env。
load_env_file(PROJECT_ROOT / ".env")

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="测试 F32C 从零位顺时针到目标，或从该目标逆时针回零。",
)


def run_rotation(
    config: FocRotationConfig,
    *,
    angle_deg: float | None = None,
    return_to_zero: bool = False,
    axis_factory: Callable[[FocRotationConfig], FocRotationAxis] = (
        FocRotationAxis
    ),
) -> dict[str, object]:
    """执行一次定向目标或原路回零，并返回可记录的反馈摘要。"""

    if not isinstance(config, FocRotationConfig):
        raise TypeError("config 必须是 FocRotationConfig")
    if angle_deg is None:
        raise ValueError("必须指定 --angle")
    if not 0.0 <= angle_deg <= 360.0:
        raise ValueError("--angle 必须满足 0 <= angle <= 360")
    if not callable(axis_factory):
        raise TypeError("axis_factory 必须可调用")

    with axis_factory(config) as axis:
        if return_to_zero:
            internal_target = axis.return_counterclockwise_to_zero(angle_deg)
            operation = "return_counterclockwise_to_zero"
            requested_angle = 0.0
        else:
            internal_target = axis.move_clockwise_from_zero(angle_deg)
            operation = "move_clockwise_from_zero"
            requested_angle = angle_deg
        feedback_angle = axis.current_angle_deg()

    return {
        "operation": operation,
        "port": config.port,
        "address": config.address,
        "rpm": config.rpm,
        "requested_mechanical_angle_deg": requested_angle,
        "internal_total_target_deg": internal_target,
        "feedback_mechanical_angle_deg": feedback_angle,
        "closed_with_disable": True,
    }


@app.command()
def main(
    angle_deg: float | None = typer.Option(
        None,
        "--angle",
        help="定向旋转角度，单位度，范围 [0, 360]。",
    ),
    return_to_zero: bool = typer.Option(
        False,
        "--zero",
        help="从 --angle 指定目标逆时针沿原角度回机械零点。",
    ),
    port: str = typer.Option(
        DEFAULT_FOC_PORT,
        "--port",
        envvar="FOC_SERIAL_PORT",
        help="F32C 串口设备；默认 UART3 /dev/ttyAS3。",
    ),
    address: int = typer.Option(
        DEFAULT_FOC_ADDRESS,
        "--address",
        envvar="FOC_ADDRESS",
        min=0,
        max=255,
        help="F32C 串口地址。",
    ),
    rpm: int = typer.Option(
        DEFAULT_FOC_RPM,
        "--rpm",
        envvar="FOC_RPM",
        min=1,
        max=1000,
        help="定位速度，单位 RPM；默认 15。",
    ),
    max_rpm: int = typer.Option(
        DEFAULT_FOC_MAX_RPM,
        "--max-rpm",
        envvar="FOC_MAX_RPM",
        min=1,
        max=1000,
        help="高层允许的速度上限，默认 30 RPM。",
    ),
    feedback_timeout_seconds: float = typer.Option(
        DEFAULT_FOC_FEEDBACK_TIMEOUT_SECONDS,
        "--feedback-timeout",
        envvar="FOC_FEEDBACK_TIMEOUT_SECONDS",
        min=0.001,
        help="单次 F32C 反馈等待秒数，默认 0.3。",
    ),
    feedback_attempts: int = typer.Option(
        DEFAULT_FOC_FEEDBACK_ATTEMPTS,
        "--feedback-attempts",
        envvar="FOC_FEEDBACK_ATTEMPTS",
        min=1,
        max=10,
        help="单次查询最多重试次数，默认 3。",
    ),
) -> None:
    """收集命令行配置后执行一次真实、安全的 F32C 旋转轴操作。"""

    try:
        summary = run_rotation(
            FocRotationConfig(
                port=port,
                address=address,
                rpm=rpm,
                max_rpm=max_rpm,
                feedback_timeout_seconds=feedback_timeout_seconds,
                feedback_attempts=feedback_attempts,
            ),
            angle_deg=angle_deg,
            return_to_zero=return_to_zero,
        )
    except KeyboardInterrupt:
        typer.echo("旋转轴控制已中断；已请求失能并关闭 UART。", err=True)
        raise typer.Exit(code=130) from None
    # F32C 参考实现的专用异常没有引入本项目的继承树；此处统一转换为
    # 稳定的诊断退出码，同时由旋转轴上下文完成失能和 UART 清理。
    except Exception as error:
        typer.echo(f"旋转轴控制失败：{error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
