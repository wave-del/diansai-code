#!/usr/bin/env python3
"""交互标定：视觉顺时针和 F32C 机械正方向的对应关系。

本脚本只会连接 UART3 上的 F32C；不会连接写字机、GPIO 或电磁铁。它在
操作者确认后仅转动一个小角度，再以同样角度回到起点。请用笔、胶带或夹具
上的明显标记观察俯视方向，切勿把未吸住的拼图件放在转盘上。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

from f32c_controller import F32CMotor
from task1_adapter import (
    F32CRotationCalibration,
    save_rotation_calibration,
)


DEFAULT_OUTPUT = Path(__file__).with_name("task1.tangram.rotation.json")


def _probe_degrees(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or not 1.0 <= value <= 90.0:
        raise argparse.ArgumentTypeError("探测角度必须在 1～90 度")
    return value


def _rpm(text: str) -> int:
    value = int(text)
    if not 1 <= value <= 30:
        raise argparse.ArgumentTypeError("RPM 必须在 1～30")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--motor-port", default="/dev/ttyAS3", help="F32C UART")
    parser.add_argument("--motor-address", type=int, default=2, help="F32C 地址")
    parser.add_argument("--rpm", type=_rpm, default=10, help="标定转速")
    parser.add_argument("--probe-deg", type=_probe_degrees, default=30.0, help="正方向探测角度")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="保存标定 JSON 的路径")
    return parser


def _answer_to_sign(answer: str) -> int | None:
    normalised = answer.strip().upper()
    if normalised in {"CW", "C", "顺", "顺时针"}:
        return 1
    if normalised in {"CCW", "A", "逆", "逆时针"}:
        return -1
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 0 <= args.motor_address <= 255:
        print("⛔ F32C 地址必须在 0～255", file=sys.stderr)
        return 2
    motor = F32CMotor(port=args.motor_port, address=args.motor_address)
    connected = False
    probe_sent = False
    try:
        print("本脚本不会移动写字机或电磁铁。")
        print("请清空转盘周围空间，并在转盘/夹具上贴一个清晰方向标记。")
        if input("确认安全后输入 READY： ").strip().upper() != "READY":
            print("已取消；未连接电机。")
            return 0
        motor.connect(verify=False)
        connected = True
        motor.enable(rpm=args.rpm)
        initial_angle = motor.get_angle()
        print(f"当前 F32C 机械角度：{initial_angle:.1f}°（仅供诊断）")
        if input(
            f"即将按 F32C 机械正方向转动 +{args.probe_deg:g}°；输入 MOVE： "
        ).strip().upper() != "MOVE":
            print("已取消；电机未执行探测转动。")
            return 0
        motor.rotate_relative(args.probe_deg, rpm=args.rpm, wait=True)
        probe_sent = True
        while True:
            answer = input(
                "从俯视画面看，标记刚才移动的方向是 CW（顺时针）还是 "
                "CCW（逆时针）？输入 CW/CCW： "
            )
            sign = _answer_to_sign(answer)
            if sign is not None:
                break
            print("⛔ 请输入 CW 或 CCW。")
        if input(
            f"将按相反方向 -{args.probe_deg:g}° 回到标定起点；输入 RETURN： "
        ).strip().upper() != "RETURN":
            print("⚠ 未确认回转；请手动执行相反方向回到起点后再使用装置。")
            return 1
        motor.rotate_relative(-args.probe_deg, rpm=args.rpm, wait=True)
        probe_sent = False
        calibration = F32CRotationCalibration(
            vision_clockwise_to_motor_sign=sign,
            probe_degrees=args.probe_deg,
        )
        save_rotation_calibration(args.output, calibration)
        direction = "同向（+1）" if sign == 1 else "反向（-1）"
        print(f"✅ 标定完成：视觉顺时针 → F32C 机械方向 {direction}")
        print(f"已保存：{args.output}")
        print("后续 run_task1_tangram.py --execute 会自动读取此文件。")
        return 0
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"⛔ 标定失败：{error}", file=sys.stderr)
        return 1
    finally:
        if connected:
            if probe_sent:
                print(
                    "⚠ 探测角仍未回退；为避免不明确的反向运动，"
                    "程序不会自动补发命令。",
                    file=sys.stderr,
                )
            try:
                motor.close(disable=False)
            except Exception as error:
                print(f"⚠ 关闭 F32C 失败：{error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
