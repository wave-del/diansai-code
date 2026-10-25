#!/usr/bin/env python3
"""真实硬件联动测试：50/50 mm 取件，100/100 mm 放件，最后回原点。"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import traceback

try:
    from .puzzle_controller import (
        PuzzleConfig,
        PuzzleController,
        TransferRequest,
    )
except ImportError:
    from puzzle_controller import (
        PuzzleConfig,
        PuzzleController,
        TransferRequest,
    )


PREFERRED_WRITER_PORT = (
    "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
)


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


def _rotation_angle(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or not 0 <= value < 360:
        raise argparse.ArgumentTypeError("角度必须处于 [0, 360)")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return value


def _motor_address(text: str) -> int:
    value = int(text)
    if not 0 <= value <= 255:
        raise argparse.ArgumentTypeError("地址必须处于 0～255")
    return value


def _resolve_writer_port(explicit_port: str | None) -> str:
    if explicit_port:
        return explicit_port
    if os.path.exists(PREFERRED_WRITER_PORT):
        return PREFERRED_WRITER_PORT
    candidates = sorted(glob.glob("/dev/serial/by-id/*")) + sorted(
        glob.glob("/dev/ttyUSB*")
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        "未发现写字机 USB 串口。请接好设备，或用 --writer-port 指定路径"
    )


def _status(stage: str, detail: str) -> None:
    print(f"  ▶ {stage}: {detail}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "拼图装置真实联动测试：写字机 + F32C 无刷电机 + PL6 电磁铁"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--writer-port",
        help="写字机串口；省略时优先按 USB by-id 自动寻找",
    )
    parser.add_argument(
        "--writer-baud",
        type=_positive_int,
        default=115200,
        help="写字机串口波特率，单位 bit/s",
    )
    parser.add_argument(
        "--motor-port",
        default="/dev/ttyAS3",
        help="F32C 无刷电机 UART",
    )
    parser.add_argument(
        "--motor-address",
        type=_motor_address,
        default=2,
        help="F32C 驱动板地址",
    )
    parser.add_argument(
        "--pickup-x",
        type=_nonnegative_float,
        default=50.0,
        help="取件点 X，单位 mm（50 mm = 5 cm）",
    )
    parser.add_argument(
        "--pickup-y",
        type=_nonnegative_float,
        default=50.0,
        help="取件点 Y，单位 mm（50 mm = 5 cm）",
    )
    parser.add_argument(
        "--place-x",
        type=_nonnegative_float,
        default=100.0,
        help="放件点 X，单位 mm（100 mm = 10 cm）",
    )
    parser.add_argument(
        "--place-y",
        type=_nonnegative_float,
        default=100.0,
        help="放件点 Y，单位 mm（100 mm = 10 cm）",
    )
    parser.add_argument(
        "--safe-z",
        type=float,
        default=-2.0,
        help=(
            "临时保守安全位置；Z未完成标定前默认只命令抬高2mm"
        ),
    )
    parser.add_argument(
        "--pickup-z",
        type=float,
        default=0.0,
        help="取件位置，单位 mm；默认是自然下垂的机械 Z0",
    )
    parser.add_argument(
        "--place-z",
        type=float,
        default=0.0,
        help="放件位置，单位 mm；默认是自然下垂的机械 Z0",
    )
    parser.add_argument(
        "--final-z",
        type=float,
        default=0.0,
        help="XY 回原点后的最终 Z，默认恢复自然下垂状态",
    )
    parser.add_argument(
        "--xy-feed",
        type=_positive_float,
        default=2500.0,
        help="XY 进给速度，单位 mm/min",
    )
    parser.add_argument(
        "--z-feed",
        type=_positive_float,
        default=1500.0,
        help="Z 轴进给速度，单位 mm/min",
    )
    parser.add_argument(
        "--pickup-dwell",
        type=_nonnegative_float,
        default=0.5,
        help="吸合后等待时间，单位秒",
    )
    parser.add_argument(
        "--release-dwell",
        type=_nonnegative_float,
        default=0.5,
        help="释放后等待时间，单位秒",
    )
    parser.add_argument(
        "--rotation-angle",
        type=_rotation_angle,
        help=(
            "抓起后转到的持久机械绝对角度，单位°；省略则保持当前角度"
        ),
    )
    parser.add_argument(
        "--rotation-rpm",
        type=int,
        default=15,
        help="F32C 位置运动速度，单位 RPM，允许 1～30",
    )
    parser.add_argument(
        "--return-rotation-zero",
        action="store_true",
        help="放件并抬高后，让无刷电机回到驱动板机械零点",
    )
    parser.add_argument(
        "--workspace-x-max",
        type=_positive_float,
        help="可选软件 X 上界，单位 mm",
    )
    parser.add_argument(
        "--workspace-y-max",
        type=_positive_float,
        help="可选软件 Y 上界，单位 mm",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="跳过 RUN 人工确认；只建议自动化环境使用",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="失败时打印完整 Python 调用栈",
    )
    return parser


def _print_plan(args: argparse.Namespace, writer_port: str) -> None:
    rotation = (
        "保持无刷电机当前角度"
        if args.rotation_angle is None
        else f"抓起后转到 {args.rotation_angle:.1f}°"
    )
    if args.return_rotation_zero:
        rotation += "，放件后回机械零点"

    print()
    print("=" * 68)
    print("拼图装置联动测试计划")
    print("=" * 68)
    print(f"写字机串口 : {writer_port} @ {args.writer_baud} bit/s")
    print(
        f"无刷电机   : {args.motor_port}，地址={args.motor_address}，"
        f"{args.rotation_rpm} RPM"
    )
    print("电磁铁     : /dev/gpiochip1 line 6（PL6 / Pin 13）")
    print(
        f"取件点     : X={args.pickup_x:.3f} mm，"
        f"Y={args.pickup_y:.3f} mm，Z={args.pickup_z:.3f} mm"
    )
    print(
        f"放件点     : X={args.place_x:.3f} mm，"
        f"Y={args.place_y:.3f} mm，Z={args.place_z:.3f} mm"
    )
    print(
        f"安全高度   : Z={args.safe_z:.3f} mm；"
        f"XY={args.xy_feed:.0f} mm/min，Z={args.z_feed:.0f} mm/min"
    )
    print(f"旋转策略   : {rotation}")
    print(
        "结束状态   : 电磁铁松开、XY 回到 (0, 0)、"
        f"Z 回到 {args.final_z:.3f} mm"
    )
    print()
    print("坐标方向：X+ 向左，Y+ 向上，Z+ 向下接近拼图件。")
    print(
        "注意：脚本不会执行限位寻零；它只把启动时的当前位置定义为"
        " X0/Y0/Z0。"
    )
    print("=" * 68)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        writer_port = _resolve_writer_port(args.writer_port)
        if not os.path.exists(writer_port):
            raise FileNotFoundError(f"写字机串口不存在：{writer_port}")
        if not os.path.exists(args.motor_port):
            raise FileNotFoundError(f"无刷电机 UART 不存在：{args.motor_port}")

        config = PuzzleConfig(
            safe_z=args.safe_z,
            pickup_z=args.pickup_z,
            place_z=args.place_z,
            rest_z=args.final_z,
            xy_feed=args.xy_feed,
            z_feed=args.z_feed,
            magnet_pickup_dwell=args.pickup_dwell,
            magnet_release_dwell=args.release_dwell,
            rotation_rpm=args.rotation_rpm,
            workspace_x_max=args.workspace_x_max,
            workspace_y_max=args.workspace_y_max,
        )
        request = TransferRequest(
            pickup_x=args.pickup_x,
            pickup_y=args.pickup_y,
            place_x=args.place_x,
            place_y=args.place_y,
            rotation_angle_deg=args.rotation_angle,
            return_rotation_to_zero=args.return_rotation_zero,
        )

        _print_plan(args, writer_port)
        if not args.yes:
            print(
                "请确认：机构已经自动/人工到达最右下角，Z 已自然下垂"
                "并停在下方机械限位，运动区域无人手和障碍物。"
            )
            answer = input("确认无误后输入 RUN 开始（其他输入取消）: ")
            if answer.strip().upper() != "RUN":
                print("已取消，未连接硬件，也未产生运动。")
                return 0

        controller = PuzzleController(
            config=config,
            status_callback=_status,
        )
        with controller:
            controller.connect(
                writer_port=writer_port,
                writer_baud=args.writer_baud,
                motor_port=args.motor_port,
                motor_address=args.motor_address,
            )
            controller.establish_origin()
            result = controller.transfer_piece(request)

        position_ok = (
            abs(result.final_x) <= 0.5
            and abs(result.final_y) <= 0.5
            and abs(result.final_z - config.rest_z) <= 0.5
        )
        if not position_ok or result.magnet_engaged:
            raise RuntimeError(
                "流程返回但最终状态复核未通过："
                f"X={result.final_x:.3f}, Y={result.final_y:.3f}, "
                f"Z={result.final_z:.3f}, "
                f"电磁铁吸合={result.magnet_engaged}"
            )

        print()
        print("✅ 联动测试完成")
        print(
            f"最终位置：X={result.final_x:.3f} mm，"
            f"Y={result.final_y:.3f} mm，Z={result.final_z:.3f} mm"
        )
        print(
            "电磁铁：已松开；Z：已恢复落下状态；"
            "无刷电机：已失能；写字机：保持使能。"
        )
        return 0
    except KeyboardInterrupt:
        print(
            "\n⛔ 收到 Ctrl+C；程序已尝试抬高、释放电磁铁并回原点，"
            "随后关闭设备。",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(f"\n⛔ 联动测试失败：{exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
