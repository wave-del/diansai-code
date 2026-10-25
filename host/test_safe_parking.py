#!/usr/bin/env python3
"""记录并重复验证拼图装置的安全停靠点。

默认读取同目录 ``safe_parking.json``。本工具只连接写字机：不连接 F32C、
不打开电磁铁。每轮先移动至可见的离开点，再回到安全停靠点，供操作者观察
重复定位与相机对 A4 纸/拼图的遮挡情况。

使用前，须人工使机构位于写字机自身的可重复机械限位角且 Z 已自然下垂。脚本
会在收到明确确认后才连接串口，以该机械原点执行 ``G92``；蓝纸坐标则始终由
已标定的 ``X-23.5/Y+16.45`` 偏置推导，绝不要求人工对准蓝纸边界。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
PUZZLE_DIR = PROJECT_ROOT / "handover-puzzle"
if str(PUZZLE_DIR) not in sys.path:
    sys.path.insert(0, str(PUZZLE_DIR))

from writer_controller import WriterController


DEFAULT_PORT = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
DEFAULT_CONFIG = HERE / "safe_parking.json"


def _positive(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("必须为正的有限数")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return value


def _xy(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} 必须是两个数字组成的数组")
    x, y = (float(item) for item in value)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{label} 必须是有限数")
    return x, y


def _load_config(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("安全停靠点配置版本不受支持")
    _xy(raw.get("blue_paper_xy_mm"), "blue_paper_xy_mm")
    _xy(raw.get("repeat_departure_xy_mm"), "repeat_departure_xy_mm")
    for name in ("safe_lift_mm", "xy_feed_mm_min"):
        value = float(raw.get(name, math.nan))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} 必须为正的有限数")
    if not 0.0 <= float(raw["safe_lift_mm"]) <= 15.0:
        raise ValueError("safe_lift_mm 必须位于 0～15 mm")
    return raw


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--port", default=DEFAULT_PORT, help="写字机 USB 串口")
    parser.add_argument("--repeat", type=_positive_int, default=5, help="往返次数")
    parser.add_argument("--xy-feed", type=_positive, help="覆盖记录中的 XY 速度，mm/min")
    parser.add_argument("--dry-run", action="store_true", help="只显示坐标映射，不连接硬件")
    return parser


def _mapped_xy(x: float, y: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """返回未限幅机器坐标和 WriterController 实际映射坐标。"""

    raw = (
        x + WriterController.A4_X_OFFSET_MM,
        y + WriterController.A4_Y_OFFSET_MM,
    )
    return raw, WriterController.map_a4_xy(x, y)


def _print_point(label: str, point: tuple[float, float]) -> None:
    raw, mapped = _mapped_xy(*point)
    clipped = [
        axis for axis, before, after in zip(("X", "Y"), raw, mapped)
        if not math.isclose(before, after, abs_tol=1e-9)
    ]
    suffix = f"；{'/'.join(clipped)} 受既有机器限幅" if clipped else ""
    print(
        f"{label}：蓝纸 X={point[0]:.3f}, Y={point[1]:.3f} mm"
        f" → 机器 X={mapped[0]:.3f}, Y={mapped[1]:.3f} mm{suffix}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = _load_config(args.config)
        safe_xy = _xy(config["blue_paper_xy_mm"], "blue_paper_xy_mm")
        departure_xy = _xy(
            config["repeat_departure_xy_mm"], "repeat_departure_xy_mm"
        )
        lift = float(config["safe_lift_mm"])
        feed = args.xy_feed or float(config["xy_feed_mm_min"])
    except Exception as error:
        print(f"⛔ 无法读取安全停靠点配置：{error}", file=sys.stderr)
        return 2

    print("=" * 72)
    print(f"安全停靠点：{config.get('name', 'unnamed')}")
    print(str(config.get("description", "")))
    _print_point("离开点", departure_xy)
    _print_point("安全停靠点", safe_xy)
    print(f"安全 Z 抬高：{lift:g} mm；XY 速度：{feed:g} mm/min；往返：{args.repeat} 次")
    print("=" * 72)

    # Y=-1 的记录点必须没有被限幅；否则它并未真正退到纸张外侧。
    raw_safe, mapped_safe = _mapped_xy(*safe_xy)
    if not math.isclose(raw_safe[1], mapped_safe[1], abs_tol=1e-9):
        print("⛔ 安全停靠点的 Y 坐标被机器限幅；拒绝进行伪定位测试。", file=sys.stderr)
        return 2
    if args.dry_run:
        print("预览完成：未连接写字机。")
        return 0
    if not Path(args.port).exists():
        print(f"⛔ 写字机串口不存在：{args.port}", file=sys.stderr)
        return 2
    prompt = (
        "请确认：机构已手动位于写字机自身机械限位角、Z 已自然下垂、"
        "运动路径无障碍。输入 PARK 建立机械原点并开始： "
    )
    if input(prompt).strip().upper() != "PARK":
        print("已取消；未连接硬件。")
        return 0

    writer = WriterController()
    connected = False
    try:
        writer.connect(args.port, baud=115200)
        connected = True
        writer.home()
        writer.configure_pen(up_z=-lift, down_z=-2.0, speed=1500.0)
        writer.pen_up()
        writer.set_motor_hold(True)
        for index in range(1, args.repeat + 1):
            print(f"\n第 {index}/{args.repeat} 轮：安全 Z 下移动到离开点...")
            writer.move_to(*departure_xy, feedrate=feed)
            print("回安全停靠点；请观察吸头与 A4 检测画面。")
            writer.move_to(*safe_xy, feedrate=feed)
            x, y = writer.position
            print(f"固件报告位置：X={x:.3f} mm，Y={y:.3f} mm；状态={writer.machine_state}")
            if index != args.repeat:
                answer = input("回车继续下一轮；输入 q 结束并停在安全点： ").strip().lower()
                if answer in {"q", "quit", "exit"}:
                    break
        print("测试结束：XY 停在安全停靠点，Z 保持抬高。")
        return 0
    except KeyboardInterrupt:
        print("\n已中断；不再发送新运动命令。", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"⛔ 安全停靠点测试失败：{error}", file=sys.stderr)
        return 1
    finally:
        if connected:
            try:
                writer.set_motor_hold(True)
            except Exception as error:
                print(f"⚠ 无法确认步进保持：{error}", file=sys.stderr)
            writer.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
