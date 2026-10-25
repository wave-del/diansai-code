#!/usr/bin/env python3
"""
写字机可靠队列自动化回归测试。

测试生产用 WriterController 的完整链路：
- ESP32/DTR 启动握手；
- 单条命令的 ok + 真实 Idle 等待；
- 以 ok 做流控的连续路径；
- XY 混合路径及回原点；
- GRBL error 是否会可靠抛给调用方。

默认只移动 XY，不移动笔轴，不执行 $H，不修改 EEPROM。
"""

from __future__ import annotations

import argparse
import math
import sys
import time

try:
    from .writer_controller import WriterController
    from .grbl_controller import list_serial_ports
except ImportError:
    from writer_controller import WriterController
    from grbl_controller import list_serial_ports


TOLERANCE_MM = 0.15


def select_port() -> str:
    ports = list_serial_ports()
    if not ports:
        raise RuntimeError("未检测到串口")
    print("可用串口:")
    for index, port in enumerate(ports):
        print(f"  [{index}] {port['name']} — {port['desc']}")
    while True:
        try:
            choice = int(input("选择串口号: "))
            if 0 <= choice < len(ports):
                return ports[choice]["name"]
        except ValueError:
            pass


def distance(actual: tuple[float, float], expected: tuple[float, float]) -> float:
    return math.dist(actual, expected)


def assert_position(
    controller: WriterController,
    expected: tuple[float, float],
    label: str,
) -> tuple[float, float]:
    actual = controller.position
    error = distance(actual, expected)
    print(
        f"  {label}: X={actual[0]:.3f}, Y={actual[1]:.3f}, "
        f"偏差={error:.3f}mm"
    )
    if error > TOLERANCE_MM:
        raise AssertionError(
            f"{label}位置错误：期望 {expected}，实际 {actual}，偏差 {error:.3f}mm"
        )
    if controller.machine_state != "Idle":
        raise AssertionError(
            f"{label}返回时状态不是 Idle: {controller.machine_state}"
        )
    return actual


def timed(label: str, action) -> float:
    start = time.monotonic()
    action()
    elapsed = time.monotonic() - start
    print(f"  {label}耗时: {elapsed:.3f}s")
    return elapsed


def run(port: str, automatic: bool) -> int:
    controller = WriterController()
    homed = False
    passed: list[str] = []

    print("=" * 64)
    print("写字机可靠队列自动化回归")
    print("=" * 64)
    print(f"串口: {port} @ 115200")
    print("最大测试范围: 用户坐标 X=30mm、Y=10mm；结束后回测试原点")
    if not automatic:
        input("请确认机构当前位置四周至少有 30mm 余量，然后按 Enter 开始...")

    try:
        timed("连接与 ESP32 启动握手", lambda: controller.connect(port))
        print(f"  固件状态: {controller.machine_state}")

        controller.home()
        homed = True
        assert_position(controller, (0.0, 0.0), "临时测试原点")

        print("\n[1/5] 单条用户坐标移动：X=5mm @ 600mm/min")
        elapsed = timed(
            "单条 5mm",
            lambda: controller.move_to(5.0, 0.0, feedrate=600),
        )
        assert_position(controller, (5.0, 0.0), "单条终点")
        if elapsed < 0.25:
            raise AssertionError(f"单条运动 {elapsed:.3f}s，疑似再次读取旧 Idle")
        controller.move_to(0.0, 0.0, feedrate=1200)
        assert_position(controller, (0.0, 0.0), "单条测试回原点")
        passed.append("单条 ok/Idle")

        print("\n[2/5] 20 个连续绝对坐标点：收到 ok 即发下一点")
        path = [(float(x), 0.0) for x in range(1, 21)]
        elapsed = timed(
            "20 段连续路径",
            lambda: controller.move_path(path, feedrate=2000),
        )
        assert_position(controller, (20.0, 0.0), "连续路径终点")
        if elapsed < 0.25:
            raise AssertionError(f"连续路径 {elapsed:.3f}s，完成等待不可信")
        controller.move_to(0.0, 0.0, feedrate=1200)
        assert_position(controller, (0.0, 0.0), "连续路径回原点")
        passed.append("20 段应答流控")

        print("\n[3/5] XY 混合矩形路径")
        rectangle = [
            (0.0, 10.0),
            (20.0, 10.0),
            (20.0, 0.0),
            (0.0, 0.0),
        ]
        timed(
            "20×10mm 矩形",
            lambda: controller.move_path(rectangle, feedrate=1500),
        )
        assert_position(controller, (0.0, 0.0), "矩形终点")
        passed.append("XY 混合路径")

        print("\n[4/5] 30 段密集折线路径")
        zigzag = [
            (float(index), 5.0 if index % 2 else 0.0)
            for index in range(1, 31)
        ]
        timed(
            "30 段折线",
            lambda: controller.move_path(zigzag, feedrate=2000),
        )
        assert_position(controller, (30.0, 0.0), "折线终点")
        controller.move_to(0.0, 0.0, feedrate=1200)
        assert_position(controller, (0.0, 0.0), "折线回原点")
        passed.append("30 段队列")

        print("\n[5/5] 错误传播：安全发送一个不支持的 G-code")
        try:
            controller.send_gcode("G999")
        except RuntimeError as exc:
            print(f"  正确捕获: {exc}")
        else:
            raise AssertionError("G999 被拒绝后，上层没有收到异常")
        controller.move_to(2.0, 0.0, feedrate=600)
        assert_position(controller, (2.0, 0.0), "错误后的正常命令")
        controller.move_to(0.0, 0.0, feedrate=1200)
        assert_position(controller, (0.0, 0.0), "最终原点")
        passed.append("error 传播与恢复")

        print("\n" + "=" * 64)
        print(f"✅ 全部通过 ({len(passed)}/{len(passed)})")
        for item in passed:
            print(f"  ✓ {item}")
        print("机器已回到本次测试原点；测试没有移动 Z 轴。")
        return 0
    except Exception as exc:
        print(f"\n⛔ 自动化回归失败: {exc}", file=sys.stderr)
        if homed and controller.is_connected():
            try:
                print("尝试回到本次测试原点...")
                controller.move_to(0.0, 0.0, feedrate=1000)
                print(f"恢复位置: {controller.position}")
            except Exception as recovery_exc:
                print(f"⚠ 自动回原点失败: {recovery_exc}", file=sys.stderr)
        return 1
    finally:
        if controller.is_connected():
            controller.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="串口，如 /dev/ttyUSB0；省略时交互选择")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="跳过开始前的人工确认（用于已确认安全余量的自动回归）",
    )
    args = parser.parse_args()
    return run(args.port or select_port(), args.auto)


if __name__ == "__main__":
    raise SystemExit(main())
