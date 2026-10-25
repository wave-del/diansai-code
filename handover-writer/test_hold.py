#!/usr/bin/env python3
"""人工验证 GRBL $1 电机空闲保持设置。"""

from __future__ import annotations

import argparse

try:
    from .cli_utils import select_port
    from .writer_controller import WriterController
except ImportError:
    from cli_utils import select_port
    from writer_controller import WriterController


def ask(prompt: str) -> bool:
    return input(prompt).strip().lower() == "y"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port")
    args = parser.parse_args()

    print("注意：$1 会写入 GRBL EEPROM；本测试结束时恢复为 $1=25。")
    input("请把机构放在安全位置，按 Enter 开始...")
    controller = WriterController()
    origin_ready = False
    try:
        controller.connect(select_port(args.port))
        controller.home()
        origin_ready = True

        controller.set_motor_hold(True)
        controller.move_to(2.0, 0.0, feedrate=600)
        held = not ask("$1=255：现在机构能被手推吗？[y/N]: ")
        print("  " + ("✓ 保持使能正常" if held else "⚠ 仍可推动"))

        controller.set_motor_hold(False)
        controller.move_to(0.0, 0.0, feedrate=600)
        released = ask("$1=25：等待片刻后机构能被手推吗？[y/N]: ")
        print("  " + ("✓ 空闲失能正常" if released else "⚠ 仍不可推动"))

        print("✅ 测试结束，已恢复 $1=25，机器在测试原点")
        return 0 if held and released else 1
    except Exception as exc:
        print(f"⛔ 电机保持测试失败: {exc}")
        return 1
    finally:
        if controller.is_connected():
            try:
                controller.set_motor_hold(False)
                if origin_ready:
                    controller.move_to(0.0, 0.0, feedrate=600)
            except Exception:
                pass
            controller.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
