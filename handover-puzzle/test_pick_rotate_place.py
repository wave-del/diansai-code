#!/usr/bin/env python3
"""真实硬件测试：在 (50,50) 吸取，搬运途中旋转 180°，在 (100,100) 放下。

上电前，操作者必须把机构人工放到最右下角，并使 Z 自然下垂到下方机械
限位。脚本把该位置定义为 X0/Y0/Z0；不会执行无传感器的 $H。
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import signal
import sys
import threading
import time
import traceback

try:
    from .electromagnet import ElectromagnetController
    from .f32c_controller import F32CMotor
    from .writer_controller import WriterController
except ImportError:
    from electromagnet import ElectromagnetController
    from f32c_controller import F32CMotor
    from writer_controller import WriterController


PREFERRED_WRITER_PORT = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的有限数字")
    return value


def _nonnegative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("必须是大于等于 0 的有限数字")
    return value


def _rpm(text: str) -> int:
    value = int(text)
    if not 1 <= value <= 30:
        raise argparse.ArgumentTypeError("RPM 必须是 1～30 的整数")
    return value


def _rotation_degrees(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or not 0 < abs(value) <= 360:
        raise argparse.ArgumentTypeError("旋转角度必须处于 (-360, 0) 或 (0, 360]")
    return value


def _resolve_writer_port(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.path.exists(PREFERRED_WRITER_PORT):
        return PREFERRED_WRITER_PORT
    candidates = sorted(glob.glob("/dev/serial/by-id/*")) + sorted(
        glob.glob("/dev/ttyUSB*")
    )
    if not candidates:
        raise FileNotFoundError("未发现写字机串口；请用 --writer-port 指定")
    return candidates[0]


def _install_signal_handlers() -> None:
    """Qt 串口初始化后也会再次安装，确保 Ctrl+C 可进入恢复流程。"""
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    signal.siginterrupt(signal.SIGINT, True)
    signal.siginterrupt(signal.SIGTERM, True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--writer-port", help="写字机 USB 串口")
    parser.add_argument("--motor-port", default="/dev/ttyAS3", help="F32C UART")
    parser.add_argument("--motor-address", type=int, default=2, help="F32C 地址")
    parser.add_argument("--pickup-x", type=_nonnegative_float, default=50.0, help="取件 X，mm")
    parser.add_argument("--pickup-y", type=_nonnegative_float, default=50.0, help="取件 Y，mm")
    parser.add_argument("--place-x", type=_nonnegative_float, default=100.0, help="放件 X，mm")
    parser.add_argument("--place-y", type=_nonnegative_float, default=100.0, help="放件 Y，mm")
    parser.add_argument(
        "--safe-z",
        type=float,
        default=-8.0,
        help="搬运时 Z 抬起高度，mm；负数代表向上",
    )
    parser.add_argument("--xy-feed", type=_positive_float, default=2500.0, help="XY 速度，mm/min")
    parser.add_argument("--z-feed", type=_positive_float, default=1500.0, help="Z 速度，mm/min")
    parser.add_argument("--rotation-deg", type=_rotation_degrees, default=180.0, help="相对当前机械方向旋转的角度，°")
    parser.add_argument("--rotation-rpm", type=_rpm, default=15, help="无刷位置速度，RPM")
    parser.add_argument("--pickup-dwell", type=_nonnegative_float, default=0.5, help="吸合稳定等待，秒")
    parser.add_argument("--release-dwell", type=_nonnegative_float, default=0.5, help="释放稳定等待，秒")
    parser.add_argument("--yes", action="store_true", help="跳过 RUN 确认，仅供自动化使用")
    parser.add_argument("--reuse-machine-origin", action="store_true", help="复用本次上电已建立的机器原点；不再发送 G92")
    parser.add_argument(
        "--disable-motor-on-exit",
        action="store_true",
        help="正常完成后失能无刷；默认保持驱动板当前使能和位置保持",
    )
    parser.add_argument("--debug", action="store_true", help="失败时输出 Python 调用栈")
    return parser


def _print_plan(args: argparse.Namespace, writer_port: str) -> None:
    print("=" * 72)
    print("拼图吸取 / 同步旋转 / 放件测试")
    print("=" * 72)
    print("起始原点：人工置于最右下角；Z 自然下垂的下方机械限位为 Z0")
    print(f"写字机：{writer_port} @ 115200")
    print(f"无刷：{args.motor_port}，地址={args.motor_address}，{args.rotation_rpm} RPM")
    print("电磁铁：PL6 / Pin 13，高电平吸合、低电平松开")
    print(f"取件：({args.pickup_x:.1f}, {args.pickup_y:.1f}) mm")
    print(f"放件：({args.place_x:.1f}, {args.place_y:.1f}) mm")
    print(f"搬运 Z：{args.safe_z:.1f} mm；XY/Z：{args.xy_feed:.0f}/{args.z_feed:.0f} mm/min")
    print(f"同步动作：XY 从取件点前往放件点时，无刷相对当前角度旋转 {args.rotation_deg:+.1f}°")
    print("结束位置：放件后抬起 Z，XY 回到 (0,0)；Z 保持抬起状态")
    print("注意：已知 Z 高速 20mm 会撞上限；本脚本只使用刚刚实测可保持的 8mm。")
    print("=" * 72)


def _safe_cleanup(
    writer: WriterController,
    writer_connected: bool,
    motor: F32CMotor,
    motor_connected: bool,
    magnet: ElectromagnetController,
    magnet_open: bool,
    *,
    z_is_safe: bool,
    disable_motor: bool,
) -> None:
    """故障时优先抬 Z、松开电磁铁；所有步骤尽力执行。"""
    if writer_connected and not z_is_safe:
        try:
            writer.pen_up()
        except BaseException as exc:
            print(f"⚠ 恢复时抬 Z 失败：{exc}", file=sys.stderr)
    if magnet_open:
        try:
            magnet.release()
        except BaseException as exc:
            print(f"⚠ 恢复时释放电磁铁失败：{exc}", file=sys.stderr)
    if motor_connected:
        try:
            motor.close(disable=disable_motor)
        except BaseException as exc:
            print(f"⚠ 关闭无刷电机失败：{exc}", file=sys.stderr)
    if writer_connected:
        try:
            writer.set_motor_hold(True)
            writer.disconnect()
        except BaseException as exc:
            print(f"⚠ 关闭写字机失败：{exc}", file=sys.stderr)
    if magnet_open:
        try:
            magnet.close(release=True)
        except BaseException as exc:
            print(f"⚠ 关闭电磁铁失败：{exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    _install_signal_handlers()
    args = _build_parser().parse_args(argv)
    if not -8.0 <= args.safe_z < 0:
        print("⛔ --safe-z 只允许 -8mm 到 0mm 之间；Z 负方向为抬起。")
        return 2
    if not 0 <= args.motor_address <= 255:
        print("⛔ --motor-address 必须处于 0～255。")
        return 2

    writer = WriterController()
    motor = F32CMotor(port=args.motor_port, address=args.motor_address)
    magnet = ElectromagnetController()
    writer_connected = False
    motor_connected = False
    magnet_open = False
    z_is_safe = False
    completed = False

    try:
        writer_port = _resolve_writer_port(args.writer_port)
        if not os.path.exists(writer_port):
            raise FileNotFoundError(f"写字机串口不存在：{writer_port}")
        if not os.path.exists(args.motor_port):
            raise FileNotFoundError(f"无刷 UART 不存在：{args.motor_port}")
        _print_plan(args, writer_port)
        if not args.yes:
            print("确认机构在右下角、Z 自然下垂到下限、铁片位于 (50,50)、路径无人手。")
            if input("输入 RUN 开始（其他输入取消）: ").strip().upper() != "RUN":
                print("已取消；未连接硬件，也未运动。")
                return 0

        magnet.open()
        magnet_open = True
        writer.connect(writer_port, baud=115200)
        writer_connected = True
        _install_signal_handlers()
        writer.set_motor_hold(True)
        motor.connect(verify=False)
        motor_connected = True
        motor.enable(rpm=args.rotation_rpm)
        print("▶ 无刷电机回驱动板机械零点...")
        motor.return_to_mechanical_zero(rpm=args.rotation_rpm, wait=True)

        if args.reuse_machine_origin:
            print("▶ 复用本次上电已建立的写字机原点（不发送 G92），并抬起 Z...")
        else:
            print("▶ 建立写字机自身原点 X0/Y0/Z0，并抬起 Z...")
            writer.home()
        writer.set_motor_hold(True)
        writer.configure_pen(up_z=args.safe_z, down_z=0.0, speed=args.z_feed)
        writer.pen_up()
        z_is_safe = True

        print(f"▶ 移动到取件点 ({args.pickup_x:.1f}, {args.pickup_y:.1f})...")
        writer.move_to(args.pickup_x, args.pickup_y, feedrate=args.xy_feed)
        print("▶ 落到 Z0 并吸合电磁铁...")
        writer.pen_move_to(0.0, feedrate=args.z_feed)
        z_is_safe = False
        magnet.engage()
        time.sleep(args.pickup_dwell)

        print(f"▶ 抬起 Z 到 {args.safe_z:.1f}mm，准备搬运...")
        writer.pen_up()
        z_is_safe = True

        rotation_started = threading.Event()
        rotation_errors: list[BaseException] = []
        rotation_target: list[float] = []

        def rotate_during_xy() -> None:
            try:
                current = motor.get_angle()
                target = (current + args.rotation_deg) % 360.0
                rotation_target.append(target)
                rotation_started.set()
                motor.move_to(target, rpm=args.rotation_rpm, wait=True)
            except BaseException as exc:
                rotation_errors.append(exc)
                rotation_started.set()

        worker = threading.Thread(
            target=rotate_during_xy,
            name="puzzle-rotate-during-xy",
            daemon=True,
        )
        worker.start()
        if not rotation_started.wait(timeout=2.0):
            raise TimeoutError("无刷电机未能在 2 秒内开始准备旋转")
        if rotation_errors:
            raise RuntimeError(f"无刷电机未能启动旋转：{rotation_errors[0]}")

        print(
            f"▶ 同步执行：XY → ({args.place_x:.1f}, {args.place_y:.1f})，"
            f"无刷 → {rotation_target[0]:.1f}°（相对 {args.rotation_deg:+.1f}°）..."
        )
        writer.move_to(args.place_x, args.place_y, feedrate=args.xy_feed)
        worker.join()
        if rotation_errors:
            raise RuntimeError(f"无刷旋转失败：{rotation_errors[0]}")

        print("▶ 落到 Z0，松开电磁铁...")
        writer.pen_move_to(0.0, feedrate=args.z_feed)
        z_is_safe = False
        magnet.release()
        time.sleep(args.release_dwell)

        print("▶ 放件后抬起 Z，XY 回蓝纸坐标 (0,0)（应用蓝纸偏置；Z 保持抬起）...")
        writer.pen_up()
        z_is_safe = True
        writer.move_to(0.0, 0.0, feedrate=args.xy_feed)

        writer.set_motor_hold(True)
        x, y = writer.position
        z = writer.pen_position
        print("\n✅ 测试完成")
        print(f"最终蓝纸业务坐标：X=0.000 mm，Y=0.000 mm；内部命令坐标：X={x:.3f} mm，Y={y:.3f} mm，Z={z:.3f} mm")
        print(f"电磁铁：已松开；无刷电机：已先回机械零点并完成相对 {args.rotation_deg:+.1f}° 旋转。")
        print(f"Z：保持抬起在 {args.safe_z:.3f} mm，不会回到 Z0。")
        completed = True
        return 0
    except KeyboardInterrupt:
        print("\n⛔ 收到 Ctrl+C，开始安全恢复（抬 Z、松开电磁铁、关闭设备）...", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\n⛔ 测试失败：{exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1
    finally:
        if writer_connected or motor_connected or magnet_open:
            _safe_cleanup(
                writer,
                writer_connected,
                motor,
                motor_connected,
                magnet,
                magnet_open,
                # 正常放件后要求停在 Z0；只有异常路径才需要为了恢复而抬 Z。
                z_is_safe=True if completed else z_is_safe,
                # 正常完成后保持无刷使能，避免工件角度受到手动扰动；异常时
                # 一律失能，避免把未知状态继续保持在机构上。
                disable_motor=(not completed) or args.disable_motor_on_exit,
            )
        if completed:
            motor_state = "无刷保持使能" if not args.disable_motor_on_exit else "无刷已失能"
            print(f"设备已关闭：电磁铁松开、{motor_state}；写字机已写入 $1=255。")


if __name__ == "__main__":
    raise SystemExit(main())
