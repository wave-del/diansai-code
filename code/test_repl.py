#!/usr/bin/env python3
"""
写字机命令行交互控制器
======================

在终端中输入短命令控制写字机运动。

命令:
  w         上 (Y 负方向)
  s         下 (Y 正方向)
  a         左 (X 负方向)
  d         右 (X 正方向)
  u         抬笔
  j         落笔
  +         步长×2
  -         步长/2
  p         查位置
  h         帮助
  q         退出

用法:
  python code/test_repl.py
"""

import sys
import select
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent
REF_DIR  = REPO_DIR / 'reference'
sys.path.insert(0, str(REF_DIR))

from PySide6.QtCore import QCoreApplication
from grbl_controller import GrblController, list_serial_ports


# ═══════════════════════════════════════════════════════════════

def select_port() -> str:
    ports = list_serial_ports()
    if not ports:
        print("❌ 未检测到串口")
        sys.exit(1)

    print("\n可用串口:")
    for i, p in enumerate(ports):
        print(f"  [{i}]  {p['name']} — {p['desc']}")

    while True:
        try:
            idx = int(input("\n选择串口号: "))
            if 0 <= idx < len(ports):
                return ports[idx]['name']
        except ValueError:
            pass


# ═══════════════════════════════════════════════════════════════

def main():
    app = QCoreApplication(sys.argv)
    port = select_port()

    cnc = GrblController()
    print(f"\n🔌 连接 {port} @ 115200 ...")
    cnc.open_port(port)

    # 轮询等待连接（connected 信号在你这版 PySide6 上有冲突）
    while not cnc.is_connected:
        app.processEvents()

    print("✅ 已连接")
    print("=" * 45)
    print("  命令:  w/a/s/d = 方向  u = 抬笔  j = 落笔")
    print("        +/- = 步长调大/调小  p = 查位置")
    print("        h = 帮助  q = 退出")
    print("=" * 45)

    step_mm = 10.0
    feed = 1500

    try:
        while True:
            # 泵 Qt 事件，让串口正常收发
            app.processEvents()

            # 非阻塞检查 stdin
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not r:
                continue

            cmd = sys.stdin.readline().strip().lower()
            if not cmd:
                continue

            # ---- 退出 ----
            if cmd == 'q':
                print("退出")
                break

            # ---- 帮助 ----
            if cmd == 'h':
                print(
                    "\n"
                    "  w      上 (Y 负方向)\n"
                    "  s      下 (Y 正方向)\n"
                    "  a      左 (X 负方向)\n"
                    "  d      右 (X 正方向)\n"
                    "  u      抬笔 (Z0)\n"
                    "  j      落笔 (Z2)\n"
                    "  +      步长×2\n"
                    "  -      步长/2\n"
                    "  p      查询位置\n"
                    "  q      退出"
                )
                continue

            # ---- 步长 ----
            if cmd == '+':
                step_mm = min(step_mm * 2, 100.0)
                print(f"  步长 → {step_mm:.1f} mm")
                continue
            if cmd == '-':
                step_mm = max(step_mm / 2, 0.5)
                print(f"  步长 → {step_mm:.1f} mm")
                continue

            # ---- 查位置 ----
            if cmd == 'p':
                cnc.query_status()
                app.processEvents()
                if cnc.position_valid:
                    x, y = cnc.position
                    z = cnc.pen_position
                    print(f"  X={x:.3f}  Y={y:.3f}  Z={z:.3f}  [{cnc.state}]")
                else:
                    print(f"  等待坐标…  [{cnc.state}]")
                continue

            # ---- 方向点动 ----
            moves = {'w': (0, -step_mm), 's': (0, step_mm),
                     'a': (-step_mm, 0), 'd': (step_mm, 0)}
            labels = {'w': '上', 's': '下', 'a': '左', 'd': '右'}
            if cmd in moves:
                dx, dy = moves[cmd]
                print(f"  {labels[cmd]}  {step_mm:.1f} mm")
                cnc.jog(dx, dy, feed)
                # 等 GRBL 回到 Idle 再接受下一条
                while cnc.state != "Idle":
                    app.processEvents()
                continue

            # ---- 笔控制 ----
            if cmd == 'u':
                print("  抬笔  Z0")
                cnc.pen_up()
                while cnc.state != "Idle":
                    app.processEvents()
                continue

            if cmd == 'j':
                print("  落笔  Z2")
                cnc.pen_down()
                while cnc.state != "Idle":
                    app.processEvents()
                continue

            # 不认识的命令
            print(f"  未知命令 '{cmd}'，输入 h 看帮助")

    except (EOFError, KeyboardInterrupt):
        print("\n中断")

    finally:
        cnc.close_port()
        app.processEvents()


if __name__ == '__main__':
    main()
