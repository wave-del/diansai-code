#!/usr/bin/env python3
"""
绝对坐标往返精度测试
====================

验证右下角为原点后，绝对坐标 move_to() 是否能准确往返。

流程：
  ① 连接串口
  ② 手动把机构放到右下角 → G92 X0 Y0 Z0
  ③ 抬笔状态，依次走到工作区各点
  ④ 记录每次回到原点的偏差

用法:
  python code/test_absolute.py
"""

import sys
import select
import math
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


def wait_enter(app) -> None:
    """等用户按 Enter（可以是空的，直接敲 Enter 即可）。"""
    while True:
        app.processEvents()
        r, _, _ = select.select([sys.stdin], [], [], 0.2)
        if r:
            sys.stdin.readline()  # 消费整行，内容无所谓
            return


def main():
    app = QCoreApplication(sys.argv)
    port = select_port()

    cnc = GrblController()
    print(f"\n🔌 连接 {port} @ 115200 ...")
    cnc.open_port(port)

    while not cnc.is_connected:
        app.processEvents()
    print("✅ 已连接")

    # ── 归零 ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  请把机构放到**右下角**物理限位处")
    print("  确保笔已抬起")
    print("=" * 55)
    input("  放好后按 Enter 继续...")

    cnc.set_manual_origin()
    for _ in range(80):
        app.processEvents()
        if cnc.position_valid:
            break

    print("\n✅ G92 X0 Y0 Z0 已执行")
    print("  方向说明: X+ = 往左(进工作区)  Y+ = 往上(进工作区)")
    print()

    # ── 测试点 ───────────────────────────────────────────
    test_points = [
        ("往左 20mm",     20,    0),
        ("回到原点",       0,    0),
        ("往上 20mm",      0,   20),
        ("回到原点",       0,    0),
        ("往左 50mm",     50,    0),
        ("回到原点",       0,    0),
        ("往上 50mm",      0,   50),
        ("回到原点",       0,    0),
        ("往左30+往上30", 30,   30),
        ("回到原点",       0,    0),
        ("往左80+往上50", 80,   50),
        ("回到原点",       0,    0),
        ("往左 100mm",   100,    0),
        ("回到原点",       0,    0),
        ("往上 100mm",     0,  100),
        ("回到原点",       0,    0),
    ]

    results = []

    print("每一步都会等你按 Enter 再执行。")
    print(f"  {'步':>3s}  {'动作':<18s}  {'结果':>6s}  {'偏差':>8s}")
    print("  " + "-" * 45)

    for i, (label, tx, ty) in enumerate(test_points):
        is_return = label == "回到原点"

        # 等你按 Enter
        sys.stdout.write(f"\n  [{i+1:2d}] 按 Enter 执行「{label}」... ")
        sys.stdout.flush()
        wait_enter(app)

        # 执行
        if is_return:
            cnc.return_to_manual_origin()
        else:
            cnc.move_to(tx, ty, 2000)

        # 主动查状态，等 GRBL 回到 Idle（必须不停发 ? 才能及时捕获）
        cnc.query_status()
        while True:
            app.processEvents()
            if cnc.state == "Idle":
                break
            # 还没收到状态变化，继续查
            cnc.query_status()
        # 等坐标更新
        cnc.query_status()
        for _ in range(30):
            app.processEvents()

        rx, ry = cnc.position

        if is_return:
            dev = math.sqrt(rx**2 + ry**2)
            if dev < 0.5:
                mark = "✅"
            elif dev < 2.0:
                mark = "⚠️"
            else:
                mark = "❌"
            print(f"  {mark:>6s}  偏差 {dev:.3f} mm")
            results.append(dev)
        else:
            x_s = f"{rx:.2f}"
            y_s = f"{ry:.2f}"
            print(f"  {'到达':>6s}  X={x_s}  Y={y_s}")

    # ── 汇总 ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  精度测试汇总")
    print("=" * 55)

    if results:
        print(f"  共 {len(results)} 次回到原点")
        print(f"  最大偏差: {max(results):.3f} mm")
        print(f"  最小偏差: {min(results):.3f} mm")
        print(f"  平均偏差: {sum(results)/len(results):.3f} mm")

        good = sum(1 for d in results if d < 0.5)
        warn = sum(1 for d in results if 0.5 <= d < 2.0)
        bad  = sum(1 for d in results if d >= 2.0)

        print(f"  优秀(<0.5mm): {good} 次")
        print(f"  警告(0.5~2mm): {warn} 次")
        print(f"  失败(>2mm): {bad} 次")

        if bad > 0:
            print("\n  ⚠️  有较大偏差，建议检查：")
            print("    · 右下角定位是否每次都一致")
            print("    · 皮带张力是否合适")
            print("    · 步进电机是否丢步")
        elif warn > 0:
            print("\n  ⚠️  有轻微偏差，实际使用前确认是否可接受")
        else:
            print("\n  ✅ 精度良好，可以继续开发 Tangram 逻辑")

    cnc.close_port()
    print("\n测试结束")


if __name__ == '__main__':
    main()
