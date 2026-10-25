"""安全地控制接在 Cubie A7Z 物理引脚 3（PJ23）的电磁铁。

该脚本依赖真实 GPIO，不属于默认单元测试。启动时始终先将 PJ23
配置为低电平输出；只有显式传入 ``--on-seconds`` 时才会上磁，且在
计时结束、异常或 Ctrl+C 后都会拉低。独立运行：
``poetry run python tests/external/motion/diancitie.py --help``。
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Callable

import typer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ec_26_e.actuation.contracts import MagnetConfig
from ec_26_e.actuation.magnet import GpiodMagnetOutput


# A7Z 的 PJ23 位于 gpiochip0；23 + 32 * 9（A 到 J）= 311。
GPIO_CHIP = Path("/dev/gpiochip0")
PJ23_LINE_OFFSET = 311

app = typer.Typer(
    add_completion=False,
    help="控制 A7Z 物理引脚 3（PJ23）上的高电平有效电磁铁。",
)


def run_magnet(
    on_seconds: float,
    *,
    hold_low: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """执行一次安全的电磁铁控制，并在返回前确保输出为低电平。"""

    if isinstance(on_seconds, bool) or not isinstance(on_seconds, (int, float)):
        raise TypeError("on_seconds 必须是非负秒数")
    if on_seconds < 0:
        raise ValueError("on_seconds 不能小于 0")
    if not isinstance(hold_low, bool):
        raise TypeError("hold_low 必须是 bool")
    if not callable(sleep):
        raise TypeError("sleep 必须可调用")
    if on_seconds > 0 and hold_low:
        raise ValueError("--on-seconds 与 --hold-low 不能同时使用")

    output = GpiodMagnetOutput(
        MagnetConfig(chip_path=GPIO_CHIP, line_offset=PJ23_LINE_OFFSET)
    )
    output.open()  # open() 会以低电平申请 GPIO，先消磁再做任何控制。
    try:
        if on_seconds > 0:
            output.set_enabled(True)
            sleep(float(on_seconds))
            output.set_enabled(False)
            return f"已上磁 {on_seconds:g} 秒，现已消磁"
        if hold_low:
            typer.echo("PJ23 已保持低电平；按 Ctrl+C 退出。")
            while True:
                sleep(1.0)
        return "PJ23 已设为低电平，电磁铁处于消磁状态"
    finally:
        # 无论正常结束、异常还是 Ctrl+C，释放前均再次写低电平。
        output.close()


@app.command()
def main(
    on_seconds: float = typer.Option(
        0.0,
        "--on-seconds",
        min=0.0,
        help="上磁的持续秒数；默认 0，只消磁而不拉高。",
    ),
    hold_low: bool = typer.Option(
        False,
        "--hold-low",
        help="持续占用 PJ23 并保持低电平，按 Ctrl+C 后仍会先消磁。",
    ),
) -> None:
    """从命令行执行电磁铁消磁或限时上磁。"""

    try:
        result = run_magnet(on_seconds, hold_low=hold_low)
    except KeyboardInterrupt:
        typer.echo("已中断，PJ23 已恢复低电平。")
        raise typer.Exit(code=130) from None
    except (RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"电磁铁控制失败：{error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(result)


if __name__ == "__main__":
    app()
