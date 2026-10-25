#!/usr/bin/env python3
"""将 Z 从自然下垂零点抬起 8mm，并保持串口连接以检查静态保持力。

本脚本不移动 XY，不修改 $102，也不控制无刷电机或电磁铁。高速 20mm 曾撞
上方限位；8mm 是操作者明确要求的单点测试，每次运动前都会二次确认。
"""

from __future__ import annotations

import argparse
import glob
import os
import queue
import signal
import sys
import threading

try:
    from .writer_controller import WriterController
except ImportError:
    from writer_controller import WriterController


PREFERRED_PORT = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
LIFT_MM = 8.0
Z_FEED = 1500.0


class HoldTestInterrupt(BaseException):
    """把 Ctrl+C 转为测试内的安全暂停。"""


_INPUT_QUEUE: queue.Queue[str | None] = queue.Queue()
_INPUT_THREAD_STARTED = False


def _input_worker() -> None:
    while True:
        line = sys.stdin.readline()
        if line == "":
            _INPUT_QUEUE.put(None)
            return
        _INPUT_QUEUE.put(line.rstrip("\r\n"))


def _prompt(text: str) -> str:
    """后台读取终端，使主线程可正常响应 Ctrl+C。"""
    global _INPUT_THREAD_STARTED
    if not _INPUT_THREAD_STARTED:
        threading.Thread(
            target=_input_worker,
            name="z-hold-test-input",
            daemon=True,
        ).start()
        _INPUT_THREAD_STARTED = True
    print(text, end="", flush=True)
    while True:
        try:
            result = _INPUT_QUEUE.get(timeout=0.1)
        except queue.Empty:
            continue
        if result is None:
            raise HoldTestInterrupt("终端输入已关闭")
        return result


def _install_signal_handlers() -> None:
    def pause_test(signum, frame) -> None:
        raise HoldTestInterrupt(f"收到信号 {signum}")

    signal.signal(signal.SIGINT, pause_test)
    signal.signal(signal.SIGTERM, pause_test)
    signal.siginterrupt(signal.SIGINT, True)
    signal.siginterrupt(signal.SIGTERM, True)


def _feed_hold(controller: WriterController) -> None:
    """只发送实时暂停字符，不擅自运动 Z。"""
    if not controller.is_connected():
        return
    try:
        controller._cnc._write_bytes(b"!")
        controller._cnc._serial.flush()
        controller._cnc._serial.waitForBytesWritten(100)
    except BaseException:
        pass


def _resolve_port(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.path.exists(PREFERRED_PORT):
        return PREFERRED_PORT
    candidates = sorted(glob.glob("/dev/serial/by-id/*")) + sorted(
        glob.glob("/dev/ttyUSB*")
    )
    if not candidates:
        raise FileNotFoundError("未发现写字机串口；请接好设备或用 --port 指定")
    return candidates[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", help="写字机 USB 串口")
    parser.add_argument(
        "--z-feed",
        type=float,
        default=Z_FEED,
        help="Z 速度，单位 mm/min；本机实测 1500 有效",
    )
    return parser


def _set_z_zero_and_hold(controller: WriterController) -> None:
    """当前位置必须是自然下垂的下方机械限位；不触碰 XY 坐标。"""
    controller.send_gcode("$X")
    controller.send_gcode("$1=255")
    controller.send_gcode("G92 Z0")


def main(argv: list[str] | None = None) -> int:
    _install_signal_handlers()
    args = _parser().parse_args(argv)
    if args.z_feed <= 0:
        print("⛔ --z-feed 必须大于 0。")
        return 2

    controller = WriterController()
    connected = False
    z_is_raised = False
    try:
        port = _resolve_port(args.port)
        print("=" * 72)
        print("Z 轴抬起 8mm 静态保持测试")
        print("=" * 72)
        print(f"串口：{port} @ 115200")
        print(f"测试动作：自然下垂 Z0 → Z=-{LIFT_MM:g}mm（向上抬起）")
        print(f"速度：{args.z_feed:g}mm/min；不会移动 XY。")
        print("已知高速 20mm 会撞上方限位；本脚本只执行一次 8mm 动作。")
        print()
        if _prompt("确认 Z 已自然落到下方机械限位后，输入 READY: ").strip().upper() != "READY":
            print("已取消；未打开串口、未运动。")
            return 0

        controller.connect(port=port, baud=115200)
        connected = True
        _install_signal_handlers()  # Qt 可能会覆盖信号处理器。
        controller.set_motor_hold(True)
        if _prompt("再次确认 Z 在下方机械限位，输入 DOWN 建立 Z0: ").strip().upper() != "DOWN":
            print("已取消；未发送 Z 运动。")
            return 0
        _set_z_zero_and_hold(controller)

        if _prompt(
            f"即将以 {args.z_feed:g}mm/min 抬起 Z {LIFT_MM:g}mm；输入 LIFT 8 执行: "
        ).strip().upper() != "LIFT 8":
            print("已取消；未发送 Z 抬起命令。")
            return 0
        controller.pen_move_to(-LIFT_MM, feedrate=args.z_feed)
        controller.set_motor_hold(True)
        z_is_raised = True

        print("\n✅ Z 已抬起 8mm，串口保持打开，$1=255。")
        print("现在请观察 Z 是否在不施力时下坠；再轻触感受保持力。")
        print("命令：status / hold / down / quit")
        while True:
            try:
                command = _prompt("z-hold> ").strip().lower()
            except HoldTestInterrupt:
                _feed_hold(controller)
                print("\n已发送暂停；请执行 down 回到 Z0，再输入 quit。")
                continue
            if command in ("status", "pos"):
                print(
                    f"缓存 Z={controller.pen_position:.3f}mm，"
                    f"GRBL={controller.machine_state or '未知'}；"
                    "坐标是开环计算值，不是实际高度反馈。"
                )
            elif command == "hold":
                controller.set_motor_hold(True)
                print("已再次写入 $1=255。")
            elif command == "down":
                if _prompt("即将回到 Z0，输入 LOWER 执行: ").strip().upper() != "LOWER":
                    print("已取消，Z 保持在当前抬起位置。")
                    continue
                controller.pen_move_to(0.0, feedrate=args.z_feed)
                controller.set_motor_hold(True)
                z_is_raised = False
                print("✅ 已回到 Z0。现在可输入 quit。")
            elif command in ("quit", "q", "exit"):
                if z_is_raised:
                    print("⛔ Z 仍抬起；请先执行 down，避免关闭串口后状态改变。")
                    continue
                return 0
            elif command in ("help", "?"):
                print("status / hold / down / quit")
            elif command:
                print("未知命令；输入 help 查看。")
    except HoldTestInterrupt as exc:
        _feed_hold(controller)
        print(f"\n⛔ 测试中断：{exc}")
        if z_is_raised:
            print("Z 仍抬起；请现场确认安全后再重新建立 Z0。")
        return 130
    except Exception as exc:
        _feed_hold(controller)
        print(f"\n⛔ 测试失败：{exc}")
        return 1
    finally:
        if connected:
            try:
                if not z_is_raised:
                    controller.set_motor_hold(True)
                controller.disconnect()
            except BaseException:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
