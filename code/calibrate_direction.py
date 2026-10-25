#!/usr/bin/env python3
"""
方向标定脚本
============

目的：确认 X+ / X- / Y+ / Y- 的实际运动方向，并调整参数。

流程：
  ① 把机构放到行程中间 → 连接
  ② 依次测试 X+/X-/Y+/Y- 各走 2mm
  ③ 每次动完你告诉我"实际往哪边走了"
  ④ 如果方向不对，自动调整 reverse_x / reverse_y / swap_xy
  ⑤ 全部确认后打印最终参数

用法:
  python code/calibrate_direction.py
"""

import sys
import select
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent
REF_DIR  = REPO_DIR / 'reference'
sys.path.insert(0, str(REF_DIR))

from PySide6.QtCore import QCoreApplication
from grbl_controller import GrblController, list_serial_ports


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


def wait_cmd(app, prompt="> ") -> str:
    """非阻塞等待用户输入一行命令。"""
    sys.stdout.write(prompt)
    sys.stdout.flush()
    while True:
        app.processEvents()
        r, _, _ = select.select([sys.stdin], [], [], 0.1)
        if r:
            return sys.stdin.readline().strip().lower()


def main():
    app = QCoreApplication(sys.argv)
    port = select_port()

    cnc = GrblController()
    print(f"\n🔌 连接 {port} @ 115200 ...")
    cnc.open_port(port)

    while not cnc.is_connected:
        app.processEvents()
    print("✅ 已连接")

    # ── 第一步：归零 ─────────────────────────────────────
    print("\n" + "=" * 50)
    print("请把机构放到行程**中间附近**，留出各方向余量")
    print("=" * 50)
    input("放好后按 Enter 继续...")

    print("\n是否执行 G92 X0 Y0 Z0 归零？（建议归零，方便后续观察坐标）")
    ans = input("[Y/n]: ").strip().lower()
    if ans != 'n':
        cnc.set_manual_origin()
        # 等位置有效
        for _ in range(50):
            app.processEvents()
            if cnc.position_valid:
                break
        print("✅ 已归零，当前位置 = (0, 0)")
    else:
        print("跳过归零")

    input("\n确认笔已抬起，按 Enter 开始方向测试...")

    # ── 当前坐标轴配置 ───────────────────────────────────
    step = 2.0  # 小步 2mm，安全
    feed = 500  # 慢速

    # 当前配置
    swap_xy = True      # 已知为 True
    rev_x = False
    rev_y = False

    # ── 逐个方向测试 ─────────────────────────────────────
    # 测试序列：先测 4 个方向，每个方向让用户反馈实际运动
    test_dirs = [
        ("X+ 方向",    step,  0),
        ("X- 方向",   -step,  0),
        ("Y+ 方向",    0,  step),
        ("Y- 方向",    0, -step),
    ]

    direction_map = {}  # 记录用户看到的实际方向

    print(f"\n开始方向测试（步长 {step}mm，速度 {feed}mm/min）")
    print("每次动完后，请输入你观察到的**实际**运动方向：")
    print("  例如输入 '左' 或 'l'  表示往左走了")
    print("  例如输入 '右' 或 'r'  表示往右走了")
    print("  例如输入 '上' 或 'u'  表示往上走了")
    print("  例如输入 '下' 或 'd'  表示往下走了")
    print("  如果完全没动或有异常声响，输入 'x'")
    print()

    for label, dx, dy in test_dirs:
        input(f"按 Enter 执行 {label}...")

        # 设置正确的 transform
        cnc.swap_xy = swap_xy
        cnc.reverse_x = rev_x
        cnc.reverse_y = rev_y

        print(f"  → jog({dx:.1f}, {dy:.1f})  ", end="", flush=True)
        cnc.jog(dx, dy, feed)
        while cnc.state != "Idle":
            app.processEvents()
        print("done")

        # 读取坐标（如果有归零的话）
        if cnc.position_valid:
            x, y = cnc.position
            print(f"  汇报坐标: X={x:.3f}  Y={y:.3f}")

        # 问用户实际看到了什么
        obs = ""
        while not obs:
            obs = input("  实际往哪个方向动了？(左/右/上/下/x异常): ").strip().lower()

        direction_map[label] = obs
        print(f"  记录: {label} → 实际 {obs}")
        print()

    # ── 显示结果并给出建议 ───────────────────────────────
    print("\n" + "=" * 50)
    print("方向测试结果：")
    print("=" * 50)
    for label, obs in direction_map.items():
        print(f"  {label}: 实际 {obs}")

    # 分析方向是否正确
    # 注: swap_xy=True 时，用户的 X 映射到电机 Y，用户的 Y 映射到电机 X
    # 所以：
    #   X+ 应该实际的"右"或对应方向
    #   Y+ 应该实际的"下"或对应方向
    # 取决于机器实际安装方向，需要用户来确认是否"符合预期"

    print("\n这些方向是否符合你的预期？")
    print("  对参考工程来说，默认配置下：")
    print("    X+ = 右  (或电机 Y 正方向)")
    print("    Y+ = 下  (或电机 X 正方向)")
    print("    但你的机器机械安装可能不同，以实际为准")
    print()
    correct = input("所有方向都符合预期吗？[y/N]: ").strip().lower()

    if correct != 'y':
        print("\n请告诉我哪个方向不对，我来调整配置。")
        print("选项: swap(对调XY)  rev_x(X反向)  rev_y(Y反向)")
        while True:
            cmd = input("\n要调什么？(swap/rev_x/rev_y/done): ").strip().lower()
            if cmd == 'done':
                break
            elif cmd == 'swap':
                swap_xy = not swap_xy
                print(f"  swap_xy 已切换为 {swap_xy}")
                # 重新测试全部方向
                for label, dx, dy in test_dirs:
                    input(f"按 Enter 重新测试 {label}...")
                    cnc.swap_xy = swap_xy
                    cnc.reverse_x = rev_x
                    cnc.reverse_y = rev_y
                    print(f"  → jog({dx:.1f}, {dy:.1f})  ", end="", flush=True)
                    cnc.jog(dx, dy, feed)
                    while cnc.state != "Idle":
                        app.processEvents()
                    print("done")
                    if cnc.position_valid:
                        x, y = cnc.position
                        print(f"    坐标: X={x:.3f}  Y={y:.3f}")
                    input("    确认方向？按 Enter 继续...")
            elif cmd == 'rev_x':
                rev_x = not rev_x
                print(f"  reverse_x 已切换为 {rev_x}")
            elif cmd == 'rev_y':
                rev_y = not rev_y
                print(f"  reverse_y 已切换为 {rev_y}")
    else:
        print("\n✅ 方向标定完成！")

    # ── 打印最终配置 ─────────────────────────────────────
    print("\n" + "=" * 50)
    print("最终参数配置：")
    print("=" * 50)
    print(f"  swap_xy    = {swap_xy}")
    print(f"  reverse_x  = {rev_x}")
    print(f"  reverse_y  = {rev_y}")
    print()
    print("可以在代码中这样设置：")
    print(f"    cnc.swap_xy   = {swap_xy}")
    print(f"    cnc.reverse_x = {rev_x}")
    print(f"    cnc.reverse_y = {rev_y}")

    cnc.close_port()


if __name__ == '__main__':
    main()
