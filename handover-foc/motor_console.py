#!/usr/bin/env python3
"""F32C 无刷电机交互控制台。"""

from __future__ import annotations

import argparse
import math
import shlex
import signal

from f32c_controller import F32CError, F32CMotor


HELP = """
命令：
  status                 查询机械角度、上电多圈角度、使能和速度
  enable [RPM]           安全使能并保持当前位置；默认15 RPM
  move ANGLE [RPM]       移动到持久机械绝对角度，0 <= ANGLE < 360，单位°
                         始终选择最短路径，并等待真实到位
  zero [RPM]             沿最短路径回到驱动板内保存的机械零点
  turn DIRECTION [RPM]   完整旋转一圈；DIRECTION 只能是 1 或 -1
  disable                立即失能
  help                   显示本帮助
  quit / q               失能、关闭串口并退出

注意：
  connect不会使能。enable会先用1 RPM保持当前位置、清除残留目标。
  F32C失能时可能不返回反馈，因此status请在enable后使用。
  本控制台没有“设置机械零点”命令，避免误覆盖驱动板里的持久零点。
"""


def finite_float(text: str, name: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{name}必须是有限数字")
    return value


def show_status(motor: F32CMotor) -> None:
    status = motor.get_status()
    print(
        f"机械角度={status['mechanical_angle_deg']:.1f}°  "
        f"上电多圈角度={status['total_angle_deg']:.1f}°  "
        f"使能={'是' if status['enabled'] else '否'}  "
        f"速度={status['rpm']} RPM"
    )


def run(args: argparse.Namespace) -> int:
    motor = F32CMotor(
        port=args.port,
        address=args.address,
        feedback_timeout=args.feedback_timeout,
    )
    try:
        motor.connect(verify=False)
        print(
            f"已连接 {args.port} @ 115200 8N1，电机地址={args.address}。"
        )
        print("连接没有使能电机，也没有发送运动目标。")
        print(HELP)

        while True:
            try:
                raw = input("\nf32c> ").strip()
                if not raw:
                    continue
                parts = shlex.split(raw)
                command = parts[0].lower()

                if command in ("q", "quit", "exit"):
                    return 0
                if command in ("help", "h", "?"):
                    print(HELP)
                    continue
                if command in ("status", "s"):
                    if not motor.is_enabled:
                        print(
                            "串口已连接，电机当前未由本程序使能；"
                            "请先执行 enable。"
                        )
                        continue
                    show_status(motor)
                    continue
                if command == "enable":
                    if len(parts) > 2:
                        raise ValueError("用法: enable [RPM]")
                    rpm = int(parts[1]) if len(parts) == 2 else args.rpm
                    motor.enable(rpm=rpm)
                    print(
                        f"电机已安全使能并保持当前位置，"
                        f"速度={rpm} RPM。"
                    )
                    show_status(motor)
                    continue
                if command == "disable":
                    if len(parts) != 1:
                        raise ValueError("用法: disable")
                    motor.disable()
                    print("电机已失能。")
                    continue
                if command == "move":
                    if len(parts) not in (2, 3):
                        raise ValueError("用法: move ANGLE [RPM]")
                    angle = finite_float(parts[1], "目标角度")
                    rpm = int(parts[2]) if len(parts) == 3 else None
                    target = motor.move_to(angle, rpm=rpm, wait=True)
                    print(
                        f"已到机械角度 {angle:.1f}°；"
                        f"内部多圈目标={target:.1f}°"
                    )
                    continue
                if command in ("zero", "home"):
                    if len(parts) not in (1, 2):
                        raise ValueError("用法: zero [RPM]")
                    rpm = int(parts[1]) if len(parts) == 2 else None
                    target = motor.return_to_mechanical_zero(
                        rpm=rpm, wait=True
                    )
                    print(
                        "已回持久机械零点；"
                        f"内部多圈目标={target:.1f}°"
                    )
                    continue
                if command == "turn":
                    if len(parts) not in (2, 3):
                        raise ValueError("用法: turn DIRECTION [RPM]")
                    direction = int(parts[1])
                    rpm = int(parts[2]) if len(parts) == 3 else None
                    target = motor.rotate_one_turn(
                        direction=direction,
                        rpm=rpm,
                        wait=True,
                    )
                    print(
                        f"完整旋转一圈完成；内部多圈目标={target:.1f}°"
                    )
                    continue
                print("未知命令；输入 help 查看帮助。")
            except EOFError:
                print("\n标准输入已关闭。")
                return 0
            except (F32CError, ValueError) as exc:
                print(f"命令失败: {exc}")
                if not motor.is_enabled:
                    print("电机当前已失能；继续运动前请重新执行 enable。")
    finally:
        was_connected = motor.is_connected
        motor.close(disable=True)
        if was_connected:
            print("电机已失能，串口已关闭。")
        else:
            print("串口未打开，未向电机发送命令。")


def main() -> int:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyAS3")
    parser.add_argument("--address", type=int, default=2)
    parser.add_argument("--rpm", type=int, default=15)
    parser.add_argument("--feedback-timeout", type=float, default=0.5)
    args = parser.parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C。")
        return 130
    except (F32CError, ValueError) as exc:
        print(f"⛔ 控制台启动失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
