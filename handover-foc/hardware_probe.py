#!/usr/bin/env python3
"""F32C 硬件通信探测：安全使能、保持当前位置、读取反馈后失能。"""

from __future__ import annotations

import argparse
import math
import time

from f32c_controller import F32CError, F32CMotor, shortest_delta_deg


def run(args: argparse.Namespace) -> int:
    motor = F32CMotor(
        port=args.port,
        address=args.address,
        feedback_timeout=args.timeout,
    )
    try:
        # 当前真实F32C在失能状态不返回角度。enable()会先预设1 RPM，
        # 使能后立即发送当前位置保持目标，不会主动执行角度位移。
        motor.connect(verify=False)
        motor.enable(rpm=1)

        samples: list[tuple[float, float]] = []
        for _ in range(args.samples):
            samples.append(
                (motor.get_total_angle(), motor.get_angle())
            )
            time.sleep(0.05)

        first_total, first_mechanical = samples[0]
        last_total, last_mechanical = samples[-1]
        total_drift = last_total - first_total
        mechanical_drift = shortest_delta_deg(
            first_mechanical,
            last_mechanical,
        )

        print(f"✅ F32C 通信正常: {args.port} @ 115200 8N1")
        print(f"   电机地址: {args.address}")
        print(f"   上电多圈坐标: {last_total:.1f}°")
        print(f"   持久机械坐标: {last_mechanical:.1f}°")
        print(
            f"   保持漂移: 多圈={total_drift:+.1f}°，"
            f"机械={mechanical_drift:+.1f}°"
        )

        if (
            abs(total_drift) > args.max_drift
            or abs(mechanical_drift) > args.max_drift
        ):
            print(
                f"⛔ 通信正常，但保持漂移超过 "
                f"{args.max_drift:.1f}°；请检查负载和参数。"
            )
            return 2

        print("   未发送业务角度目标；退出时将电机失能。")
        return 0
    except (F32CError, ValueError) as exc:
        print(f"⛔ F32C 硬件探测失败: {exc}")
        print(
            "   请检查供电、共地、UART3 TX/RX交叉接线、"
            "115200 8N1和电机地址，并确认没有其他程序占用串口。"
        )
        return 1
    finally:
        was_connected = motor.is_connected
        motor.close(disable=True)
        if was_connected:
            print("   电机已失能，UART3 已关闭。")
        else:
            print("   串口未打开，未向电机发送命令。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyAS3")
    parser.add_argument("--address", type=int, default=2)
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.8,
        help="单次反馈等待时间，单位秒（默认 0.8）",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4,
        help="保持状态采样次数，至少2次（默认 4）",
    )
    parser.add_argument(
        "--max-drift",
        type=float,
        default=1.0,
        help="允许的最大保持漂移，单位度（默认 1.0）",
    )
    args = parser.parse_args()

    if args.samples < 2:
        parser.error("--samples 必须至少为 2")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout 必须大于 0")
    if not math.isfinite(args.max_drift) or args.max_drift < 0:
        parser.error("--max-drift 必须是大于等于 0 的有限数字")
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C；正在失能并退出。")
        return 130
    except (F32CError, ValueError) as exc:
        print(f"⛔ 硬件探测参数错误: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
