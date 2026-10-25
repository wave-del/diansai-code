#!/usr/bin/env python3
"""可靠的用户绝对坐标往返测试。"""

from __future__ import annotations

import argparse
import math

try:
    from .cli_utils import select_port
    from .writer_controller import WriterController
except ImportError:
    from cli_utils import select_port
    from writer_controller import WriterController


def run(port: str, distance_mm: float, automatic: bool) -> int:
    controller = WriterController()
    origin_ready = False
    points = [
        (distance_mm, 0.0),
        (0.0, 0.0),
        (0.0, distance_mm),
        (0.0, 0.0),
        (distance_mm, distance_mm),
        (0.0, 0.0),
    ]
    try:
        if not automatic:
            input(
                "请把机构人工移动到约定的最右下角，并确认笔已抬起，"
                "按 Enter 建立本次原点..."
            )
        controller.connect(port)
        controller.home()
        origin_ready = True
        for index, point in enumerate(points, 1):
            controller.move_to(*point, feedrate=1500)
            actual = controller.position
            error = math.dist(actual, point)
            print(
                f"[{index}/{len(points)}] 目标={point} 实际="
                f"({actual[0]:.3f},{actual[1]:.3f}) 偏差={error:.3f}mm"
            )
            if error > 0.15:
                raise AssertionError(f"位置偏差超过 0.15mm: {error:.3f}mm")
        print("✅ 绝对坐标往返测试通过，机器已回原点")
        return 0
    except Exception as exc:
        print(f"⛔ 绝对坐标测试失败: {exc}")
        if origin_ready and controller.is_connected():
            try:
                controller.move_to(0.0, 0.0, feedrate=1000)
            except Exception:
                pass
        return 1
    finally:
        if controller.is_connected():
            controller.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port")
    parser.add_argument("--distance", type=float, default=20.0)
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()
    if not 0 < args.distance <= 70:
        parser.error("distance 必须在 (0, 70] mm 内")
    return run(select_port(args.port), args.distance, args.auto)


if __name__ == "__main__":
    raise SystemExit(main())
