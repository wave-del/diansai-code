#!/usr/bin/env python3
"""可靠的 Z 轴抬笔/落笔实机测试。"""

from __future__ import annotations

import argparse
import math

try:
    from .cli_utils import select_port
    from .writer_controller import WriterController
except ImportError:
    from cli_utils import select_port
    from writer_controller import WriterController


def assert_z(controller: WriterController, expected: float, label: str) -> None:
    actual = controller.pen_position
    error = abs(actual - expected)
    print(f"  {label}: Z={actual:.3f}, 偏差={error:.3f}mm")
    if error > 0.05:
        raise AssertionError(
            f"{label}错误：期望 Z={expected:.3f}，实际 Z={actual:.3f}"
        )


def run(
    port: str,
    up_z: float,
    down_z: float,
    speed: float,
    cycles: int,
    automatic: bool,
) -> int:
    controller = WriterController()
    try:
        print(f"连接 {port} @ 115200")
        print(
            f"本次参数: 抬笔 Z={up_z:.3f}, 落笔 Z={down_z:.3f}, "
            f"速度={speed:.0f}mm/min"
        )
        if not automatic:
            input(
                "请把 XY 放在安全位置，并把笔放到约定的抬笔机械高度，"
                "按 Enter 将当前位置设为 Z0..."
            )
        controller.connect(port)
        controller.home()
        controller.configure_pen(up_z, down_z, speed)
        assert_z(controller, 0.0, "人工原点")

        for index in range(1, cycles + 1):
            print(f"\n循环 {index}/{cycles}")
            controller.pen_down()
            assert_z(controller, down_z, "落笔")
            controller.pen_up()
            assert_z(controller, up_z, "抬笔")

        print("\n执行 Z +0.2 / -0.2 相对微调")
        controller.pen_jog(0.2, feedrate=min(300.0, speed))
        assert_z(controller, up_z + 0.2, "向下微调")
        controller.pen_jog(-0.2, feedrate=min(300.0, speed))
        assert_z(controller, up_z, "微调返回")

        print("\n✅ Z 轴测试通过，笔已回到抬笔高度")
        return 0
    except Exception as exc:
        print(f"\n⛔ Z 轴测试失败: {exc}")
        if controller.is_connected():
            try:
                controller.pen_move_to(up_z, feedrate=min(speed, 1000.0))
            except Exception:
                pass
        return 1
    finally:
        if controller.is_connected():
            controller.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port")
    parser.add_argument("--up", type=float, default=0.0, help="抬笔绝对 Z")
    parser.add_argument("--down", type=float, default=2.0, help="落笔绝对 Z")
    parser.add_argument("--speed", type=float, default=3000.0)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()
    if not all(math.isfinite(value) for value in (args.up, args.down, args.speed)):
        parser.error("高度和速度必须是有限数字")
    if args.cycles < 1:
        parser.error("cycles 必须至少为 1")
    return run(
        select_port(args.port),
        args.up,
        args.down,
        args.speed,
        args.cycles,
        args.auto,
    )


if __name__ == "__main__":
    raise SystemExit(main())
