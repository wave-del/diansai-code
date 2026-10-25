#!/usr/bin/env python3
"""F32C 无刷电机机械回零一致性测试。

驱动板中已通过 0x0A 保存持久机械零点（掉电保持）。本脚本重复执行：

    使能 -> 移动到测试角度 -> return_to_mechanical_zero() -> 读取机械角度

验证每次回零是否都回到同一个固定点（即驱动板中保存的机械零点）。

注意：
- 直接发送多圈目标 0° 回到的是“本次上电位置”，不是持久机械零点；
  本测试使用 f32c_controller 的 return_to_mechanical_zero()，它会读取
  当前持久机械角度和上电多圈角度，换算最短路径后回持久机械零点。
- 判定标准：每轮回零后的机械角度与 0° 的偏差均不超过 --tolerance
  （默认 1.0°，与 f32c_controller 的到位判定一致）。
- 默认测试角度覆盖 0° 两侧（350°、10°），同时检验最短路径换算在
  两个方向上的收敛一致性。

示例：
    python3 test_home_consistency.py --port /dev/ttyAS3 --address 2 \
        --rounds 10 --rpm 15
"""

from __future__ import annotations

import argparse
import math
import signal
import statistics
import time

from f32c_controller import (
    F32CError,
    F32CMotor,
    shortest_delta_deg,
)


DEFAULT_ANGLES = "60,120,180,240,300,350,10,90,200,330"


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


def run_home_round(
    motor: F32CMotor,
    angle_deg: float,
    rpm: int,
    settle_seconds: float,
) -> dict[str, float]:
    """移动到位后回机械零点，返回本轮的反馈记录。"""
    t0 = time.monotonic()
    move_internal = motor.move_to(angle_deg, rpm=rpm, wait=True)
    zero_internal = motor.return_to_mechanical_zero(rpm=rpm, wait=True)
    if settle_seconds > 0:
        time.sleep(settle_seconds)
    mechanical = motor.get_angle()
    total = motor.get_total_angle()
    elapsed = time.monotonic() - t0
    error = abs(shortest_delta_deg(mechanical, 0.0))
    return {
        "target": angle_deg,
        "mechanical": mechanical,
        "total": total,
        "error": error,
        "move_internal": move_internal,
        "zero_internal": zero_internal,
        "elapsed": elapsed,
    }


def estimate_seconds(
    angles: list[float],
    start_angle: float,
    rpm: int,
    rounds: int,
    settle_seconds: float,
) -> float:
    """按最短路径位移和 1 RPM = 6°/s 估算总耗时，单位秒。"""
    if rpm <= 0:
        return float("nan")
    move_deg = 0.0
    current = start_angle
    for index in range(rounds):
        target = angles[index % len(angles)]
        move_deg += abs(shortest_delta_deg(current, target))
        move_deg += abs(shortest_delta_deg(target, 0.0))
        current = 0.0
    # 每轮另留约4秒用于使能保持、反馈确认和命令间隔。
    return move_deg / (rpm * 6.0) + rounds * (settle_seconds + 4.0)


def summarize_and_report(records: list[dict[str, float]], tolerance: float) -> int:
    """打印汇总并返回退出码：0=通过，1=不通过。"""
    errors = [record["error"] for record in records]

    print("\n=== 回零一致性汇总 ===")
    for index, record in enumerate(records, start=1):
        mark = "✅" if record["error"] <= tolerance else "❌"
        print(
            f"{mark} 第{index:>2}轮  目标={record['target']:6.1f}°  "
            f"回零反馈={record['mechanical']:6.1f}°  "
            f"偏差={record['error']:5.2f}°  "
            f"用时={record['elapsed']:5.1f}s"
        )

    mean = statistics.mean(errors)
    stdev = statistics.stdev(errors) if len(errors) >= 2 else 0.0
    max_error = max(errors)
    spread = max_error - min(errors)
    print(
        f"\n回零反馈与0°的偏差：最小={min(errors):.2f}°，"
        f"最大={max_error:.2f}°，平均={mean:.2f}°，"
        f"标准差={stdev:.2f}°"
    )
    print(
        f"回零偏差极差（最大-最小）={spread:.2f}°"
        f"（越小说明每次回到同一个固定点）"
    )

    if max_error <= tolerance:
        print(
            f"\n✅ PASS：全部{len(records)}轮均回到同一个固定点"
            f"（偏差≤{tolerance:g}°）"
        )
        return 0
    print(
        f"\n⛔ FAIL：有轮次偏差超过{tolerance:g}°"
        f"（最大偏差={max_error:.2f}°）"
    )
    return 1


def run(args: argparse.Namespace) -> int:
    angles = parse_angle_list(args.angles)
    if args.rounds < 1:
        raise ValueError("轮数必须是正整数")
    if not 0.0 < args.tolerance <= 10.0:
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

    try:
        # 失能状态可能不返回反馈，因此不验证；enable()会低速使能并
        # 清除残留目标后再设置请求速度。
        motor.connect(verify=False)
        motor.enable(rpm=args.rpm)
        start_mechanical = motor.get_angle()

        print(
            f"✅ 已连接并自动使能：{args.port} @ 115200 8N1，"
            f"地址={args.address}，速度={args.rpm} RPM"
        )
        print(f"使能后机械角度={start_mechanical:.1f}°（估算起点）")
        print(
            f"共{args.rounds}轮，测试角度依次循环："
            + ", ".join(f"{angle:g}°" for angle in angles)
        )
        estimated = estimate_seconds(
            angles, start_mechanical, args.rpm, args.rounds,
            args.settle_seconds,
        )
        if math.isfinite(estimated):
            print(f"预计总耗时约 {estimated / 60:.1f} 分钟")

        for index in range(args.rounds):
            target = angles[index % len(angles)]
            record = run_home_round(
                motor, target, args.rpm, args.settle_seconds
            )
            records.append(record)
            mark = "✅" if record["error"] <= args.tolerance else "❌"
            print(
                f"{mark} 第{index + 1:>2}/{args.rounds}轮  "
                f"目标={target:6.1f}° → 回零反馈={record['mechanical']:6.1f}°  "
                f"偏差={record['error']:5.2f}°  "
                f"用时={record['elapsed']:5.1f}s"
            )
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C；正在失能并退出。")
        return 130
    except F32CError as exc:
        print(f"⛔ 通信或运动失败（已停止测试）: {exc}")
        return 1
    finally:
        was_connected = motor.is_connected
        motor.close(disable=True)
        if was_connected:
            print("电机已失能，UART3 已关闭。")
        else:
            print("UART3 未打开，未向电机发送命令。")

    if not records:
        print("没有任何有效记录。")
        return 1
    return summarize_and_report(records, args.tolerance)


def main() -> int:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyAS3")
    parser.add_argument("--address", type=int, default=2)
    parser.add_argument(
        "--rpm",
        type=int,
        default=15,
        help="移动速度，单位 RPM，允许 1～30（默认 15）",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="回零测试轮数（默认 10）",
    )
    parser.add_argument(
        "--angles",
        type=str,
        default=DEFAULT_ANGLES,
        help="每轮移动到的测试角度列表，逗号分隔，0～360（默认循环执行）",
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
    except (F32CError, ValueError) as exc:
        print(f"⛔ 测试启动失败: {exc}")
        print(
            "请检查F32C供电、共地、UART3 TX/RX交叉接线、"
            "115200 8N1和电机地址。"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
