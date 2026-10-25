#!/usr/bin/env python3
"""F32C 真实电机交互测试：启动自动使能，直接输入 -360～360°绝对定位。"""

from __future__ import annotations

import argparse
import math
import shlex
import signal

from f32c_controller import F32CError, F32CMotor


HELP = """
直接输入：
  ANGLE              移动到持久机械绝对角度，单位°，范围 -360～360
                     例如：0、30、90、180、270、360、-50

其他命令：
  move ANGLE         与直接输入角度相同
  status / pos       查询机械角度、上电多圈角度和速度
  zero               沿最短路径回到驱动板中保存的机械零点
  turn 1             沿协议正方向完整旋转一圈
  turn -1            沿协议负方向完整旋转一圈
  speed RPM          修改速度，单位 RPM，允许 1～30
  help               显示本帮助
  quit / q           失能并退出

角度说明：
  输入值是驱动板持久机械零点坐标中的绝对角度。
  -50° 等同绝对 310°；0°、360°、-360°是同一机械方向，均按最短路径定位。
  如果需要真实转满一整圈，请使用 turn 1 或 turn -1。
"""


def parse_finite_float(text: str, name: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{name}必须是数字") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name}必须是有限数字")
    return value


def parse_angle(text: str) -> float:
    angle = parse_finite_float(text, "角度")
    if not -360.0 <= angle <= 360.0:
        raise ValueError("角度必须在 -360～360°")
    # 高层API使用半开区间；机械圆周中的360°与0°完全相同。
    return angle % 360.0


def show_status(motor: F32CMotor) -> None:
    status = motor.get_status()
    print(
        f"机械角度={status['mechanical_angle_deg']:.1f}°  "
        f"上电多圈角度={status['total_angle_deg']:.1f}°  "
        f"速度={status['rpm']} RPM  "
        f"使能={'是' if status['enabled'] else '否'}"
    )


def move_and_report(
    motor: F32CMotor,
    target: float,
    entered: float,
) -> None:
    internal_target = motor.move_to(target, wait=True)
    actual = motor.get_angle()
    display_target = (
        "360°（等同机械0°）"
        if math.isclose(entered, 360.0, abs_tol=1e-9)
        else f"{entered:.1f}°"
    )
    print(
        f"✅ 已到 {display_target}：反馈={actual:.1f}°，"
        f"内部多圈目标={internal_target:.1f}°"
    )


def run(args: argparse.Namespace) -> int:
    motor = F32CMotor(
        port=args.port,
        address=args.address,
        feedback_timeout=args.feedback_timeout,
        max_rpm=30,
    )

    try:
        # 某些F32C状态在失能时不返回反馈，因此这里只负责打开串口；
        # enable()会低速使能、清除残留目标，再设置请求速度。
        motor.connect(verify=False)
        motor.enable(rpm=args.rpm)

        print(
            f"✅ 已连接并自动使能：{args.port} @ 115200 8N1，"
            f"地址={args.address}，速度={args.rpm} RPM"
        )
        show_status(motor)
        print(HELP)

        while True:
            try:
                raw = input("\nangle> ").strip()
            except EOFError:
                print("\n标准输入已关闭。")
                return 0

            if not raw:
                continue

            try:
                parts = shlex.split(raw)
                command = parts[0].lower()

                if command in ("q", "quit", "exit"):
                    return 0
                if command in ("help", "h", "?"):
                    print(HELP)
                    continue
                if command in ("status", "pos", "s"):
                    if len(parts) != 1:
                        raise ValueError(f"用法: {command}")
                    show_status(motor)
                    continue
                if command in ("zero", "home"):
                    if len(parts) != 1:
                        raise ValueError("用法: zero")
                    internal_target = motor.return_to_mechanical_zero(
                        wait=True
                    )
                    actual = motor.get_angle()
                    print(
                        f"✅ 已回机械零点：反馈={actual:.1f}°，"
                        f"内部多圈目标={internal_target:.1f}°"
                    )
                    continue
                if command == "turn":
                    if len(parts) != 2:
                        raise ValueError("用法: turn 1 或 turn -1")
                    direction = int(parts[1])
                    internal_target = motor.rotate_one_turn(
                        direction=direction,
                        wait=True,
                    )
                    print(
                        "✅ 完整旋转一圈完成："
                        f"协议方向={direction:+d}，"
                        f"内部多圈目标={internal_target:.1f}°"
                    )
                    continue
                if command == "speed":
                    if len(parts) != 2:
                        raise ValueError("用法: speed RPM")
                    rpm = int(parts[1])
                    motor.set_speed(rpm)
                    print(f"✅ 速度已设置为 {rpm} RPM")
                    continue
                if command == "move":
                    if len(parts) != 2:
                        raise ValueError("用法: move ANGLE")
                    entered = parse_finite_float(parts[1], "角度")
                    target = parse_angle(parts[1])
                    move_and_report(motor, target, entered)
                    continue
                if len(parts) == 1:
                    entered = parse_finite_float(parts[0], "角度")
                    target = parse_angle(parts[0])
                    move_and_report(motor, target, entered)
                    continue

                raise ValueError("无法识别输入；输入 help 查看命令")
            except ValueError as exc:
                print(f"⛔ 命令失败: {exc}")
            except F32CError as exc:
                print(f"⛔ 通信或运动失败: {exc}")
                if motor.is_enabled:
                    motor.disable()
                print("电机已失能，程序退出。")
                return 1
    finally:
        was_connected = motor.is_connected
        motor.close(disable=True)
        if was_connected:
            print("电机已失能，UART3 已关闭。")
        else:
            print("UART3 未打开，未向电机发送命令。")


def main() -> int:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyAS3")
    parser.add_argument("--address", type=int, default=2)
    parser.add_argument(
        "--rpm",
        type=int,
        default=15,
        help="启动速度，单位 RPM，允许 1～30（默认 15）",
    )
    parser.add_argument(
        "--feedback-timeout",
        type=float,
        default=0.8,
        help="单次反馈超时，单位秒（默认 0.8）",
    )
    args = parser.parse_args()

    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C；正在失能并退出。")
        return 130
    except (F32CError, ValueError) as exc:
        print(f"⛔ 测试控制台启动失败: {exc}")
        print(
            "请检查F32C供电、共地、UART3 TX/RX交叉接线、"
            "115200 8N1和电机地址。"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
