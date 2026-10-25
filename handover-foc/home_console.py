#!/usr/bin/env python3
"""F32C 无刷电机机械回零一致性交互测试台。

启动即安全使能。每一步运动（移动、回零、整圈）在发送目标前都会打印
“正在...”进度，完成后打印真实反馈，便于对照实际电机运动。

每次回零（zero / home 命令或 rounds 自动轮）都会把结果累计到记录中，
随时执行 stats 查看全部回零的偏差统计和通过/不通过判定。

回零目标：驱动板中通过 0x0A 保存的持久机械零点（掉电保持）。
注意：直接发送多圈目标 0° 回到的是“本次上电位置”，不是持久机械零点；
本控制台的 zero 使用 return_to_mechanical_zero() 做坐标换算。

示例：
    python3 home_console.py --port /dev/ttyAS3 --address 2 --rpm 15
"""

from __future__ import annotations

import argparse
import math
import shlex
import signal
import statistics
import time

from f32c_controller import (
    F32CError,
    F32CMotor,
    shortest_delta_deg,
)


DEFAULT_ANGLES = "60,120,180,240,300,350,10,90,200,330"

HELP = """
命令：
  move ANGLE [RPM]    移动到持久机械绝对角度，0 <= ANGLE < 360，单位°
                      始终选择最短路径，并等待真实到位
  zero / home [RPM]   回驱动板内保存的持久机械零点，并记录本次回零结果
  turn DIRECTION [RPM] 完整旋转一圈；DIRECTION 只能是 1 或 -1
  rounds N            自动连续执行 N 轮：移动 -> 回零，每步详细打印
  angles A,B,C        设置自动轮使用的测试角度序列
  tolerance DEG       设置判定容差，单位°，0.1～10.0（默认 1.0）
  stats / report      显示全部回零记录、偏差统计和通过/不通过判定
  watch [SECONDS]     持续读取机械角度观察当前运动（默认 5 秒）
  speed RPM           修改后续运动速度，允许 1～30
  status / pos        查询机械角度、上电多圈角度、使能和速度
  help                显示本帮助
  quit / q            失能、关闭串口并退出

说明：
  启动即安全使能并清除驱动板残留目标。
  move/zero/turn 在发送目标前打印“正在...”，完成后打印反馈，
  便于知道电机什么时候在回零、什么时候在运动到其他角度。
  本控制台没有“设置机械零点”命令，避免误覆盖驱动板里的持久零点。
"""


def finite_float(text: str, name: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{name}必须是有限数字")
    return value


def parse_angle_list(text: str) -> list[float]:
    """解析逗号分隔的测试角度列表；360° 与机械 0° 相同。"""
    angles: list[float] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"测试角度必须是数字：{token!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"测试角度必须是有限数字：{token!r}")
        if not 0.0 <= value <= 360.0:
            raise ValueError(f"测试角度必须在 0～360°：{value:.1f}")
        angles.append(
            0.0 if math.isclose(value, 360.0, abs_tol=1e-9) else value
        )
    if not angles:
        raise ValueError("测试角度列表为空")
    return angles


def show_status(motor: F32CMotor) -> None:
    status = motor.get_status()
    print(
        f"机械角度={status['mechanical_angle_deg']:.1f}°  "
        f"上电多圈角度={status['total_angle_deg']:.1f}°  "
        f"使能={'是' if status['enabled'] else '否'}  "
        f"速度={status['rpm']} RPM"
    )


def do_move(motor: F32CMotor, target: float, rpm: int | None) -> None:
    """移动到绝对角度；打印阶段进度和到位反馈。"""
    print(f"  → 正在移动到 {target:g}° ...", flush=True)
    internal = motor.move_to(target, rpm=rpm, wait=True)
    actual = motor.get_angle()
    print(
        f"  ✅ 已到 {target:g}°：反馈={actual:.1f}°，"
        f"内部多圈目标={internal:.1f}°"
    )


def do_home(
    motor: F32CMotor,
    rpm: int | None,
    settle_seconds: float,
    came_from: float,
) -> dict[str, float]:
    """回持久机械零点；打印阶段进度，返回回零记录。"""
    print("  → 正在回机械零点 ...", flush=True)
    t0 = time.monotonic()
    internal = motor.return_to_mechanical_zero(rpm=rpm, wait=True)
    if settle_seconds > 0:
        time.sleep(settle_seconds)
    mechanical = motor.get_angle()
    total = motor.get_total_angle()
    elapsed = time.monotonic() - t0
    error = abs(shortest_delta_deg(mechanical, 0.0))
    mark = "✅" if error <= 1.0 else "❌"
    print(
        f"  {mark} 回零完成：反馈={mechanical:.1f}°，偏差={error:.2f}°，"
        f"内部多圈目标={internal:.1f}°，用时={elapsed:.1f}s"
    )
    return {
        "came_from": came_from,
        "mechanical": mechanical,
        "total": total,
        "error": error,
        "internal": internal,
        "elapsed": elapsed,
    }


def run_round(
    motor: F32CMotor,
    index: int,
    rounds_total: int,
    angle_deg: float,
    rpm: int,
    settle_seconds: float,
    records: list[dict[str, float]],
) -> None:
    """自动执行一轮：移动到测试角度 -> 回零，每步详细打印。"""
    print(f"\n[第{index}/{rounds_total}轮] 测试角度={angle_deg:g}°")
    do_move(motor, angle_deg, rpm)
    records.append(do_home(motor, rpm, settle_seconds, came_from=angle_deg))


def show_stats(records: list[dict[str, float]], tolerance: float) -> None:
    """显示全部回零记录、偏差统计和通过/不通过判定。"""
    if not records:
        print("还没有回零记录；执行 zero 或 rounds N。")
        return
    print("\n=== 回零一致性记录 ===")
    for index, record in enumerate(records, start=1):
        mark = "✅" if record["error"] <= tolerance else "❌"
        print(
            f"{mark} 第{index:>2}次  来向={record['came_from']:6.1f}°  "
            f"回零反馈={record['mechanical']:6.1f}°  "
            f"偏差={record['error']:5.2f}°  "
            f"用时={record['elapsed']:5.1f}s"
        )
    errors = [record["error"] for record in records]
    mean = statistics.mean(errors)
    stdev = statistics.stdev(errors) if len(errors) >= 2 else 0.0
    max_error = max(errors)
    print(
        f"\n偏差：最小={min(errors):.2f}° 最大={max_error:.2f}° "
        f"平均={mean:.2f}° 标准差={stdev:.2f}°"
    )
    print(
        f"偏差极差={max_error - min(errors):.2f}°"
        f"（越小说明每次回到同一个固定点）"
    )
    if max_error <= tolerance:
        print(
            f"✅ PASS：全部{len(records)}次回零均在容差{tolerance:g}°内"
        )
    else:
        print(
            f"❌ FAIL：存在超过容差{tolerance:g}°的回零"
            f"（最大偏差={max_error:.2f}°）"
        )


def watch(motor: F32CMotor, seconds: float) -> None:
    """持续读取机械角度，观察当前运动。"""
    deadline = time.monotonic() + seconds
    print(f"持续 {seconds:g}s 观察机械角度（Ctrl+C 提前停止）：")
    try:
        while time.monotonic() < deadline:
            angle = motor.get_angle()
            print(f"  机械角度={angle:6.1f}°", end="\r", flush=True)
            time.sleep(0.3)
    finally:
        print()


def run(args: argparse.Namespace) -> int:
    angles = parse_angle_list(args.angles)
    tolerance = args.tolerance
    if not 0.0 < tolerance <= 10.0:
        raise ValueError("容差必须在 0.1～10.0° 之间")
    if args.settle_seconds < 0:
        raise ValueError("回零后静止时间不能为负")

    motor = F32CMotor(
        port=args.port,
        address=args.address,
        feedback_timeout=args.feedback_timeout,
        max_rpm=30,
    )
    records: list[dict[str, float]] = []
    last_move: float = 0.0

    try:
        motor.connect(verify=False)
        motor.enable(rpm=args.rpm)
        print(
            f"✅ 已连接并自动使能：{args.port} @ 115200 8N1，"
            f"地址={args.address}，速度={args.rpm} RPM"
        )
        show_status(motor)
        print(f"自动轮角度序列：{', '.join(f'{a:g}°' for a in angles)}")
        print(HELP)

        while True:
            try:
                raw = input("\nhome> ").strip()
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
                if command == "move":
                    if len(parts) not in (2, 3):
                        raise ValueError("用法: move ANGLE [RPM]")
                    angle = finite_float(parts[1], "目标角度")
                    if not 0.0 <= angle <= 360.0:
                        raise ValueError("角度必须在 0～360°")
                    target = (
                        0.0
                        if math.isclose(angle, 360.0, abs_tol=1e-9)
                        else angle
                    )
                    rpm = int(parts[2]) if len(parts) == 3 else None
                    do_move(motor, target, rpm)
                    last_move = target
                    continue
                if command in ("zero", "home"):
                    if len(parts) not in (1, 2):
                        raise ValueError("用法: zero [RPM]")
                    rpm = int(parts[1]) if len(parts) == 2 else None
                    records.append(
                        do_home(motor, rpm, args.settle_seconds, last_move)
                    )
                    continue
                if command == "turn":
                    if len(parts) not in (2, 3):
                        raise ValueError("用法: turn DIRECTION [RPM]")
                    direction = int(parts[1])
                    if direction not in (-1, 1):
                        raise ValueError("direction 只能是 1 或 -1")
                    rpm = int(parts[2]) if len(parts) == 3 else None
                    print(
                        f"  → 正在沿协议{'正' if direction == 1 else '负'}"
                        f"方向完整旋转一圈 ...",
                        flush=True,
                    )
                    internal = motor.rotate_one_turn(
                        direction=direction, rpm=rpm, wait=True
                    )
                    print(
                        f"  ✅ 完整旋转一圈完成；内部多圈目标={internal:.1f}°"
                    )
                    continue
                if command == "rounds":
                    if len(parts) != 2:
                        raise ValueError("用法: rounds N")
                    count = int(parts[1])
                    if count < 1:
                        raise ValueError("轮数必须是正整数")
                    for index in range(1, count + 1):
                        angle = angles[(index - 1) % len(angles)]
                        run_round(
                            motor, index, count, angle, args.rpm,
                            args.settle_seconds, records,
                        )
                    show_stats(records, tolerance)
                    continue
                if command == "angles":
                    if len(parts) != 2:
                        raise ValueError("用法: angles A,B,C")
                    angles = parse_angle_list(parts[1])
                    print(
                        "测试角度序列已更新："
                        + ", ".join(f"{a:g}°" for a in angles)
                    )
                    continue
                if command == "tolerance":
                    if len(parts) != 2:
                        raise ValueError("用法: tolerance DEG")
                    tolerance = finite_float(parts[1], "容差")
                    if not 0.0 < tolerance <= 10.0:
                        raise ValueError("容差必须在 0.1～10.0° 之间")
                    print(f"判定容差已设置为 {tolerance:g}°")
                    continue
                if command in ("stats", "report"):
                    if len(parts) != 1:
                        raise ValueError(f"用法: {command}")
                    show_stats(records, tolerance)
                    continue
                if command == "watch":
                    if len(parts) not in (1, 2):
                        raise ValueError("用法: watch [SECONDS]")
                    seconds = (
                        finite_float(parts[1], "观察秒数")
                        if len(parts) == 2
                        else 5.0
                    )
                    if seconds <= 0:
                        raise ValueError("观察秒数必须大于0")
                    watch(motor, seconds)
                    continue
                if command == "speed":
                    if len(parts) != 2:
                        raise ValueError("用法: speed RPM")
                    rpm = int(parts[1])
                    motor.set_speed(rpm)
                    print(f"✅ 速度已设置为 {rpm} RPM")
                    continue
                raise ValueError("无法识别输入；输入 help 查看命令")
            except KeyboardInterrupt:
                print("\n收到 Ctrl+C；正在失能并退出。")
                return 130
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
        help="启动和自动轮速度，单位 RPM，允许 1～30（默认 15）",
    )
    parser.add_argument(
        "--angles",
        type=str,
        default=DEFAULT_ANGLES,
        help="自动轮测试角度列表，逗号分隔，0～360",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="判定通过的最大偏差，单位°，0.1～10.0（默认 1.0）",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=0.5,
        dest="settle_seconds",
        help="回零到位后静止时间，单位秒，0 表示不等待（默认 0.5）",
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
        print(f"⛔ 控制台启动失败: {exc}")
        print(
            "请检查F32C供电、共地、UART3 TX/RX交叉接线、"
            "115200 8N1和电机地址。"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
