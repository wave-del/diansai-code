#!/usr/bin/env python3
"""交互验证电磁铁圆心在蓝纸右上角坐标系中的 XY 位置。

本工具仅控制写字机 XY，不会连接无刷电机或电磁铁，也不会移动 Z。
它用于实物确认：输入的 ``X Y`` 经现场固定偏置与限幅后，是否让电磁铁圆心
到达预期位置。
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys

try:
    from .writer_controller import WriterController
except ImportError:
    from writer_controller import WriterController


PREFERRED_PORT = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("必须是正的有限数字")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", default=PREFERRED_PORT, help="写字机 USB 串口")
    parser.add_argument("--xy-feed", type=_positive_float, default=2500.0, help="XY 速度，mm/min")
    return parser


def _print_help() -> None:
    print("\n命令：")
    print("  X Y              输入目标坐标，单位 mm；X+ 向下，Y+ 向左")
    print("                   下发前固定修正：X-23.5、Y+16.45，再进行限幅")
    print("                   例：50 50 → 写字机 X=26.5、Y=66.45")
    print("  move X Y         与直接输入两个数字相同")
    print("  pos              查询写字机当前软件坐标")
    print("  origin           机构位于写字机自身原点时，重新建立机器 (0,0)")
    print("  home             回写字机自身原点，不应用蓝纸偏置")
    print("  help             显示帮助")
    print("  quit / q         保持电机使能并退出\n")


def _writer_xy(input_x: float, input_y: float) -> tuple[float, float]:
    """预览 WriterController 的统一映射；实际运动仍只调用一次 move_to。"""
    return WriterController.map_a4_xy(input_x, input_y)


def _parse_xy(command: str) -> tuple[float, float] | None:
    words = command.split()
    if words and words[0].lower() == "move":
        words = words[1:]
    if len(words) != 2:
        return None
    try:
        x, y = (float(value) for value in words)
    except ValueError:
        return None
    if not all(math.isfinite(value) for value in (x, y)):
        raise ValueError("坐标必须是有限数字")
    return x, y


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    args = _build_parser().parse_args(argv)
    if not os.path.exists(args.port):
        print(f"⛔ 写字机串口不存在：{args.port}", file=sys.stderr)
        return 2

    writer = WriterController()
    connected = False
    try:
        print("=" * 72)
        print("蓝纸 XY 坐标实物验证（只动 XY；不动 Z、不吸合电磁铁）")
        print("=" * 72)
        print("输入坐标定义：电磁铁圆心在蓝纸右上角时为 (0,0)。")
        print("输入 X+ 向下，Y+ 向左；WriterController 自动修正 X-23.5 mm、Y+16.45 mm。")
        print(
            "写字机最终限幅："
            f"X={WriterController.A4_WRITER_X_MIN_MM:.0f}～{WriterController.A4_WRITER_X_MAX_MM:.0f} mm，"
            f"Y={WriterController.A4_WRITER_Y_MIN_MM:.0f}～{WriterController.A4_WRITER_Y_MAX_MM:.0f} mm。"
        )
        print(f"XY 速度：{args.xy_feed:.0f} mm/min；运动结束后保持使能。")
        print()
        if input("请确认机构已在写字机自身原点，输入 ORIGIN 建立原点： ").strip().upper() != "ORIGIN":
            print("已取消；未连接硬件。")
            return 0

        writer.connect(args.port, baud=115200)
        connected = True
        writer.home()
        writer.set_motor_hold(True)
        print("已建立写字机自身原点；蓝纸偏置将由控制器自动应用；$1=255。")
        _print_help()

        while True:
            try:
                line = input("a4-xy> ").strip()
            except EOFError:
                print()
                break
            if not line:
                continue
            command = line.lower()
            if command in {"q", "quit", "exit"}:
                break
            if command in {"help", "?"}:
                _print_help()
                continue
            if command == "pos":
                x, y = writer.position
                print(f"软件坐标：X={x:.3f} mm，Y={y:.3f} mm；状态={writer.machine_state}")
                continue
            if command == "origin":
                if input("确认机构已在写字机自身原点，输入 ORIGIN： ").strip().upper() == "ORIGIN":
                    writer.home()
                    writer.set_motor_hold(True)
                    print("已重新建立 (0,0)。")
                else:
                    print("未修改原点。")
                continue
            if command == "home":
                print("▶ XY → 写字机自身原点；不应用蓝纸偏置。")
                writer.return_to_machine_origin(feedrate=args.xy_feed)
                print("位置：X=0.000 mm，Y=0.000 mm")
                continue
            try:
                target = _parse_xy(line)
                if target is None:
                    print("⛔ 无法识别。请输入 `X Y`，例如 `10 10`；或输入 help。")
                    continue
                input_x, input_y = target
                x, y = _writer_xy(input_x, input_y)
                clipped: list[str] = []
                if x != input_x + WriterController.A4_X_OFFSET_MM:
                    clipped.append("X")
                if y != input_y + WriterController.A4_Y_OFFSET_MM:
                    clipped.append("Y")
                limit_note = f"；{'/'.join(clipped)} 已限幅" if clipped else ""
                print(
                    f"▶ 输入 X={input_x:.3f} mm，Y={input_y:.3f} mm"
                    f" → 写字机 X={x:.3f} mm，Y={y:.3f} mm{limit_note}"
                )
                # move_to 内部已经调用同一份 map_a4_xy；这里绝不能把 x/y
                # （已映射机器坐标）再次传入，否则会重复偏置。
                writer.move_to(input_x, input_y, feedrate=args.xy_feed)
                actual_x, actual_y = writer.position
                print(f"固件报告：X={actual_x:.3f} mm，Y={actual_y:.3f} mm；请直接观察电磁铁圆心。")
            except (RuntimeError, TimeoutError, ValueError) as exc:
                print(f"⛔ 命令失败：{exc}")
    except KeyboardInterrupt:
        print("\n已通过 Ctrl+C 退出；不再发送运动命令。")
        return 130
    except Exception as exc:
        print(f"⛔ 控制台失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if connected:
            try:
                writer.set_motor_hold(True)
            except Exception as exc:
                print(f"⚠ 无法再次写入 $1=255：{exc}", file=sys.stderr)
            writer.disconnect()
            print("写字机串口已关闭；固件 $1=255，电机应继续保持使能。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
