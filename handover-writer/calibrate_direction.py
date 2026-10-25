#!/usr/bin/env python3
"""人工确认交付接口的用户坐标方向。"""

from __future__ import annotations

import argparse

try:
    from .cli_utils import select_port
    from .writer_controller import WriterController
except ImportError:
    from cli_utils import select_port
    from writer_controller import WriterController


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port")
    parser.add_argument("--step", type=float, default=5.0)
    args = parser.parse_args()
    if not 0 < args.step <= 20:
        parser.error("step 必须在 (0, 20] mm 内")

    controller = WriterController()
    origin_ready = False
    try:
        input(
            "请把机构放到行程中间、确认笔已抬起，按 Enter 将当前位置设为原点..."
        )
        controller.connect(select_port(args.port))
        controller.home()
        origin_ready = True
        tests = [
            ("用户 X+", args.step, 0.0, "预期往左"),
            ("用户 X-", -args.step, 0.0, "预期往右"),
            ("用户 Y+", 0.0, args.step, "预期往上"),
            ("用户 Y-", 0.0, -args.step, "预期往下"),
        ]
        for label, dx, dy, expected in tests:
            input(f"\n按 Enter 执行 {label} {args.step:.1f}mm（{expected}）...")
            controller.jog(dx, dy, feedrate=600)
            print(f"  到达用户坐标 {controller.position}")
            controller.move_to(0.0, 0.0, feedrate=600)
            answer = input("  实际方向是否符合预期？[Y/n]: ").strip().lower()
            if answer == "n":
                raise AssertionError(f"{label} 方向不符合预期")
        print("\n✅ 四个用户坐标方向全部确认，机器已回测试原点")
        return 0
    except Exception as exc:
        print(f"\n⛔ 方向确认失败: {exc}")
        if origin_ready and controller.is_connected():
            try:
                controller.move_to(0.0, 0.0, feedrate=600)
            except Exception:
                pass
        return 1
    finally:
        if controller.is_connected():
            controller.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
