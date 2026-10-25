#!/usr/bin/env python3
"""PL6 / Pin 13 电磁铁交互测试：高电平吸合，低电平松开。"""

from __future__ import annotations

import argparse
import math
import shlex
import signal

from electromagnet import ElectromagnetController, ElectromagnetError


HELP = """
命令：
  on                 输出高电平，电磁铁保持吸合
  off                输出低电平，电磁铁松开
  pulse SECONDS      吸合指定秒数后自动松开，例如 pulse 0.5
  status             读取当前GPIO状态
  help               显示本帮助
  quit / q           先松开，再释放GPIO并退出

程序启动、正常退出、Ctrl+C或EOF时都会尝试输出低电平。
"""


def finite_seconds(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("时长必须是大于0的有限秒数")
    return value


def run(args: argparse.Namespace) -> int:
    magnet = ElectromagnetController(
        chip=args.chip,
        line_offset=args.line,
        active_high=True,
        expected_line_name=args.expected_name,
    )
    try:
        magnet.open()
        print(
            f"✅ 已独占 {args.chip} line {args.line}"
            f"（{args.expected_name} / PL6 / Pin 13）"
        )
        print("当前输出低电平，电磁铁应处于松开状态。")
        print(HELP)

        while True:
            try:
                raw = input("\nmagnet> ").strip()
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
                if command in ("on", "engage"):
                    if len(parts) != 1:
                        raise ValueError("用法: on")
                    magnet.engage()
                    print("✅ 高电平：电磁铁已吸合。")
                    continue
                if command in ("off", "release"):
                    if len(parts) != 1:
                        raise ValueError("用法: off")
                    magnet.release()
                    print("✅ 低电平：电磁铁已松开。")
                    continue
                if command in ("status", "s"):
                    if len(parts) != 1:
                        raise ValueError("用法: status")
                    print(
                        "当前状态："
                        + (
                            "高电平 / 吸合"
                            if magnet.is_engaged
                            else "低电平 / 松开"
                        )
                    )
                    continue
                if command == "pulse":
                    if len(parts) != 2:
                        raise ValueError("用法: pulse SECONDS")
                    duration = finite_seconds(parts[1])
                    print(f"高电平吸合 {duration:g} 秒……")
                    magnet.pulse(duration)
                    print("✅ 脉冲完成，已恢复低电平松开。")
                    continue
                raise ValueError("未知命令；输入 help 查看帮助")
            except ValueError as exc:
                print(f"⛔ 输入错误: {exc}")
            except ElectromagnetError as exc:
                print(f"⛔ GPIO控制失败: {exc}")
                return 1
    finally:
        magnet.close(release=True)
        print("已尝试恢复低电平，GPIO已释放。")


def main() -> int:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip", default="/dev/gpiochip1")
    parser.add_argument("--line", type=int, default=6)
    parser.add_argument("--expected-name", default="PIN_13")
    args = parser.parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C；正在恢复低电平并退出。")
        return 130
    except (ElectromagnetError, ValueError) as exc:
        print(f"⛔ 电磁铁测试启动失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
