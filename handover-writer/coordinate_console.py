#!/usr/bin/env python3
"""交互式输入用户 X/Y/Z 坐标控制写字机。"""

from __future__ import annotations

import argparse
import math
import signal
import shlex

try:
    from .cli_utils import select_port
    from .writer_controller import WriterController
except ImportError:
    from cli_utils import select_port
    from writer_controller import WriterController


HELP = """
命令：
  x y z              移动到用户绝对坐标，三个坐标单位都是 mm
                     例：20 10 1.5
                     执行顺序：抬笔 → 移动 XY → 移动到目标 Z
  move x y z         与直接输入三个数字相同
  pos                显示当前用户 X/Y/Z 坐标和 GRBL 状态
  up                 移动到配置的抬笔高度
  down               移动到配置的落笔高度
  z Z                仅移动到绝对笔高 Z，单位 mm
  jogz DZ            Z 轴相对微调，单位 mm；正数向下，负数向上
  origin             人工重新归位后，把当前位置重新设为 X0/Y0/Z0
  help               显示本帮助
  quit / q           回到抬笔高度并退出

坐标方向：
  X+ 向左，Y+ 向上，Z+ 向下接近纸面。
"""


def finite_numbers(values: list[str], count: int) -> list[float]:
    if len(values) != count:
        raise ValueError(f"需要 {count} 个数字，实际收到 {len(values)} 个")
    numbers = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError("坐标必须是有限数字")
    return numbers


def show_position(controller: WriterController) -> None:
    x, y = controller.position
    z = controller.pen_position
    print(
        f"位置: X={x:.3f}mm  Y={y:.3f}mm  Z={z:.3f}mm  "
        f"状态={controller.machine_state}"
    )


def run(args: argparse.Namespace) -> int:
    controller = WriterController()
    try:
        controller.connect(port=select_port(args.port))
        print(
            f"笔参数: 抬笔 Z={args.up:.3f}mm，落笔 Z={args.down:.3f}mm，"
            f"XY速度={args.xy_feed:.0f}mm/min，Z速度={args.z_feed:.0f}mm/min"
        )
        input(
            "请把 XY 人工移到最右下角、把笔放到抬笔基准高度，"
            "然后按 Enter 建立 X0/Y0/Z0..."
        )
        controller.home()
        controller.configure_pen(
            up_z=args.up,
            down_z=args.down,
            speed=args.z_feed,
        )
        controller.pen_up()
        print("已归零；$1=255，运动结束后电机保持使能。")
        print(HELP)
        show_position(controller)

        while True:
            try:
                raw = input("\nxyz> ").strip()
                if not raw:
                    continue
                parts = shlex.split(raw)
                command = parts[0].lower()

                if command in ("q", "quit", "exit"):
                    controller.pen_up()
                    print("笔已抬起，退出。")
                    return 0
                if command in ("help", "h", "?"):
                    print(HELP)
                    continue
                if command in ("pos", "p"):
                    show_position(controller)
                    continue
                if command == "up":
                    controller.pen_up()
                    show_position(controller)
                    continue
                if command == "down":
                    controller.pen_down()
                    show_position(controller)
                    continue
                if command == "z":
                    (z,) = finite_numbers(parts[1:], 1)
                    controller.pen_move_to(z=z, feedrate=args.z_feed)
                    show_position(controller)
                    continue
                if command == "jogz":
                    (dz,) = finite_numbers(parts[1:], 1)
                    controller.pen_jog(dz=dz, feedrate=args.jog_z_feed)
                    show_position(controller)
                    continue
                if command == "origin":
                    input(
                        "请先人工重新放到最右下角和抬笔基准高度，"
                        "确认后按 Enter..."
                    )
                    controller.home()
                    controller.configure_pen(
                        up_z=args.up,
                        down_z=args.down,
                        speed=args.z_feed,
                    )
                    show_position(controller)
                    continue

                coordinate_parts = parts[1:] if command == "move" else parts
                x, y, z = finite_numbers(coordinate_parts, 3)
                controller.pen_up(feedrate=args.z_feed)
                controller.move_to(x=x, y=y, feedrate=args.xy_feed)
                controller.pen_move_to(z=z, feedrate=args.z_feed)
                show_position(controller)
            except (ValueError, RuntimeError, TimeoutError) as exc:
                print(f"输入或运动失败: {exc}")
    except (RuntimeError, TimeoutError) as exc:
        print(f"⛔ 控制台启动失败: {exc}")
        return 1
    finally:
        if controller.is_connected():
            controller.disconnect()


def main() -> int:
    # 某些 Qt/远程终端组合会让进程继承被忽略的 SIGINT 处理方式，
    # 表现为终端连续显示 ^C，但 input() 不抛出 KeyboardInterrupt。
    # 在创建 QCoreApplication 和打开串口前明确恢复 Python 默认处理。
    signal.signal(signal.SIGINT, signal.default_int_handler)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="串口路径，例如 /dev/ttyUSB0")
    parser.add_argument(
        "--xy-feed",
        type=float,
        default=2500.0,
        help="XY 进给速度，单位 mm/min，默认 2500",
    )
    parser.add_argument(
        "--z-feed",
        type=float,
        default=300.0,
        help="Z 绝对运动速度，单位 mm/min，默认 300",
    )
    parser.add_argument(
        "--jog-z-feed",
        type=float,
        default=200.0,
        help="Z 微调速度，单位 mm/min，默认 200",
    )
    parser.add_argument("--up", type=float, default=0.0, help="抬笔绝对 Z，单位 mm")
    parser.add_argument(
        "--down",
        type=float,
        default=2.0,
        help="落笔绝对 Z，单位 mm",
    )
    args = parser.parse_args()
    numeric = (args.xy_feed, args.z_feed, args.jog_z_feed, args.up, args.down)
    if not all(math.isfinite(value) for value in numeric):
        parser.error("速度和笔高必须是有限数字")
    if min(args.xy_feed, args.z_feed, args.jog_z_feed) <= 0:
        parser.error("速度必须大于 0")
    try:
        return run(args)
    except KeyboardInterrupt:
        # run() 的 finally 会先关闭串口；这里不发送任何运动命令，
        # 尤其避免在 Z 轴极性尚未标定时自动抬笔。
        print("\n收到 Ctrl+C，串口已关闭。")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
