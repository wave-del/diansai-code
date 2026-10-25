#!/usr/bin/env python3
"""
写字机最小闭环测试 —— 各方向运动验证（安全版）

设计思路：
  - 不依赖原点位置，所有运动使用相对点动 (jog)。
  - 从 1mm 小步长起步，确认方向正确后逐步扩大。
  - 每个阶段都等你确认，避免撞到机械限位。

用法:
  python code/test_smoke.py
"""

import sys
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent
REF_DIR  = REPO_DIR / 'reference'
sys.path.insert(0, str(REF_DIR))

from PySide6.QtCore import QCoreApplication, QTimer, QObject
from grbl_controller import GrblController, list_serial_ports


# ═══════════════════════════════════════════════════════════════
# 工具：串口选择
# ═══════════════════════════════════════════════════════════════

def select_port() -> str:
    ports = list_serial_ports()
    if not ports:
        print("❌ 未检测到串口，请检查 USB 连接")
        sys.exit(1)

    print("\n🔌 可用串口:")
    for i, p in enumerate(ports):
        print(f"  [{i}]  {p['name']} — {p['desc']}")
        print(f"       {p['hwid']}")

    while True:
        try:
            idx = int(input("\n请选择串口号: "))
            if 0 <= idx < len(ports):
                return ports[idx]['name']
        except ValueError:
            pass
        print("  无效选择，请输入数字")


# ═══════════════════════════════════════════════════════════════
# 安全烟雾测试
# ═══════════════════════════════════════════════════════════════

class SafeSmokeTest(QObject):
    """
    安全版运动验证。

    策略：
      - 所有运动用相对 jog，不走绝对坐标，不依赖原点位置。
      - 先做小步 1mm 试探，你确认方向正确后，再做 5mm / 10mm。
      - 每轮动作结束后等 Idle 再走下一步。
    """

    def __init__(self, port: str):
        super().__init__()
        self.cnc = GrblController(self)
        self._port = port

        # 信号
        self.cnc.connected.connect(self._on_connected)
        self.cnc.disconnected.connect(lambda: print("← 串口已断开"))
        self.cnc.machine_state.connect(lambda s: setattr(self, '_state', s))
        self.cnc.position_changed.connect(lambda x, y: (setattr(self, '_x', x), setattr(self, '_y', y)))
        self.cnc.pen_position_changed.connect(lambda z: setattr(self, '_z', z))
        self.cnc.position_valid_changed.connect(lambda v: setattr(self, '_pos_valid', v))
        self.cnc.error_occurred.connect(lambda e: print(f"\n  ⛔ {e}"))

        self._state = ""
        self._x = self._y = self._z = 0.0
        self._pos_valid = False

        # 当前动作的 feedrate
        self._feed = 1500
        self._z_feed = 300
        self._step_mm = 1  # 当前步长

    # ── 入口 ──────────────────────────────────────────────────

    def start(self):
        print(f"\n🔌 正在连接 {self._port} @ 115200 ...")
        self.cnc.open_port(self._port)

    def _on_connected(self):
        print("✅ 串口连接成功")
        self._ask_homing()

    def _ask_homing(self):
        """连接后第一步：问用户要不要归零，但不强制。"""
        print()
        print("─" * 50)
        print("  这台机器没有限位开关，所有运动将使用相对点动 (jog)，")
        print("  不依赖绝对坐标，所以不强制要求人工归零。")
        print("  但建议你手动把机构放到行程中间附近的位置，")
        print("  留出各方向的活动余量。")
        print("─" * 50)

        answer = input("\n是否执行人工归零 (G92 X0 Y0 Z0)？[y/N]: ").strip().lower()
        if answer == 'y':
            print("→ 发送 G92 X0 Y0 Z0 ...")
            self.cnc.set_manual_origin()
            self._wait_pos_valid()
        else:
            print("→ 跳过归零，直接开始测试（注意观察不要撞限位）")
            self._prompt_start()

    def _wait_pos_valid(self):
        if getattr(self, '_pos_valid', False):
            print("✅ 位置已有效")
            self._prompt_start()
        else:
            print("⏳ 等待 GRBL 返回有效坐标...")
            self.cnc.query_status()
            QTimer.singleShot(500, self._wait_pos_valid)

    def _prompt_start(self):
        """确认笔已抬起，准备开始。"""
        print()
        print("─" * 50)
        print("  请确认以下事项：")
        print("  ① 笔已抬起，不会刮到纸面")
        print("  ② 机构在行程中间附近，各方向都有余量")
        print("  ③ 周围没有障碍物")
        print("─" * 50)

        input("准备好后按 Enter 开始测试...")
        self._enter_test_phase()

    # ── 测试阶段 ──────────────────────────────────────────────

    def _enter_test_phase(self):
        """
        测试流程：
          阶段 1 — Z 轴微调 (±0.1mm)
          阶段 2 — X/Y 小步试探 (每次动完等你确认)
          阶段 3 — 如需扩大步长，手动输入
        """
        self._phase_run("Z 轴微调", self._test_z)

    def _phase_run(self, phase_name: str, coro):
        """进入一个测试阶段。"""
        print(f"\n{'='*50}")
        print(f"  阶段: {phase_name}")
        print(f"{'='*50}")
        coro()

    # ── Z 轴测试 ──────────────────────────────────────────────

    def _test_z(self):
        """先验证 Z 轴（最安全，不影响 XY）。"""
        print("\n[Z 轴] 笔轴微调 ±0.1mm，确认运动方向")
        print("  正数 = 落笔方向，负数 = 抬笔方向")
        print("  如方向相反，可在 calibration.json 中调整")
        print()

        input("按 Enter 执行 Z +0.1（落笔方向微调）...")
        self._jog_z_and_wait(+0.1, lambda: self._test_z_2())

    def _test_z_2(self):
        input("\n按 Enter 执行 Z -0.1（抬笔方向微调）...")
        self._jog_z_and_wait(-0.1, lambda: self._test_z_3())

    def _test_z_3(self):
        input("\n按 Enter 执行 Z +1.0（较大步长）...")
        self._jog_z_and_wait(+1.0, lambda: self._test_z_4())

    def _test_z_4(self):
        input("\n按 Enter 执行 Z -1.0（回到接近初始高度）...")
        self._jog_z_and_wait(-1.0, lambda: self._ask_z_result())

    def _ask_z_result(self):
        print("\n[Z 轴] 是否正常运动？")
        answer = input("  输入 y 确认通过，或任意键标记失败 (y/N): ").strip().lower()
        if answer != 'y':
            print("⚠️  Z 轴测试标记为失败，继续测试 XY")
        else:
            print("✅ Z 轴测试通过")

        self._ask_xy_ready()

    def _ask_xy_ready(self):
        """XY 轴测试前确认安全。"""
        print()
        print("─" * 50)
        print("  接下来测试 X/Y 轴。")
        print("  请确保笔已抬起，机构在行程中部。")
        print("  初始步长 = 1mm，每个动作后等你确认。")
        print("─" * 50)

        answer = input("\n开始 XY 测试？[Y/n]: ").strip().lower()
        if answer == 'n':
            print("跳过 XY 测试")
            self._finish()
            return

        self._test_xy()

    # ── XY 测试 ────────────────────────────────────────────────

    def _test_xy(self):
        """逐一测试四个方向，每次动完等你确认后再动下一步。"""
        self._step_mm = 1   # 当前步长
        self._dir_tests = [
            ("X+ 正向",  self._step_mm,  0),
            ("X- 负向", -self._step_mm,  0),
            ("Y+ 正向",  0,  self._step_mm),
            ("Y- 负向",  0, -self._step_mm),
        ]
        self._dir_idx = 0
        self._test_next_direction()

    def _test_next_direction(self):
        if self._dir_idx >= len(self._dir_tests):
            self._ask_scale_up()
            return

        name, dx, dy = self._dir_tests[self._dir_idx]
        print(f"\n[{self._dir_idx + 1}/4] {name}  ({dx:+d}, {dy:+d}) mm")

        # 先执行动作
        self._do_jog_and_wait(dx, dy, self._feed, self._ask_direction_confirm)

    def _ask_direction_confirm(self):
        """每次动完后问用户：方向对不对？"""
        name, _, _ = self._dir_tests[self._dir_idx]

        print(f"  当前坐标: X={self._x:.3f}  Y={self._y:.3f}")
        answer = input(f"  [{name}] 运动方向是否正确？[y/n/s(跳過)]: ").strip().lower()

        if answer == 's':
            print("  已跳过")
        elif answer != 'y':
            print("  ⚠️  标记为异常（方向错误或异常声响）")
            # 反向回到原位
            _, dx, dy = self._dir_tests[self._dir_idx]
            self._do_jog_and_wait(-dx, -dy, self._feed, None)
        else:
            print("  ✅ 方向正确")

        self._dir_idx += 1
        QTimer.singleShot(300, self._test_next_direction)

    def _ask_scale_up(self):
        """四个方向都测完后，问用户要不要加大步长再测一轮。"""
        print("\n" + "─" * 50)
        print("  四个方向 1mm 试探完成。")
        print("─" * 50)

        print("\n是否加大步长到 5mm 再测一轮？")
        print("  这可以验证更大范围的运动是否顺畅。")
        answer = input("  [y/N]: ").strip().lower()

        if answer == 'y':
            self._step_mm = 5
            self._dir_tests = [
                ("X+ 5mm",  5,  0),
                ("X- 5mm", -5,  0),
                ("Y+ 5mm",  0,  5),
                ("Y- 5mm",  0, -5),
            ]
            self._dir_idx = 0
            self._test_next_direction()
        else:
            self._ask_scale_up_10mm()

    def _ask_scale_up_10mm(self):
        print("\n是否加大步长到 10mm 再测一轮？")
        answer = input("  [y/N]: ").strip().lower()
        if answer == 'y':
            self._step_mm = 10
            self._dir_tests = [
                ("X+ 10mm",  10,  0),
                ("X- 10mm", -10,  0),
                ("Y+ 10mm",  0,  10),
                ("Y- 10mm",  0, -10),
            ]
            self._dir_idx = 0
            self._test_next_direction()
        else:
            self._ask_square()

    def _ask_square(self):
        """可选：画一个正方形往返，验证回原点精度（相对运动）。"""
        print("\n" + "─" * 50)
        print("  最后一项：正方形往返测试（相对点动）。")
        print("  会走一个 10×10mm 正方形，最终应回到起点附近。")
        print("─" * 50)

        answer = input("\n执行正方形往返？[y/N]: ").strip().lower()

        if answer == 'y':
            self._run_square()
        else:
            self._finish()

    def _run_square(self):
        """用相对点动走正方形，最终回到起点，验证有无明显丢步。"""
        print("\n[正方形] 10×10mm → 回到起点")

        waypoints = [
            ("→ X+10",  10,  0),
            ("→ Y+10",   0, 10),
            ("→ X-10", -10,  0),
            ("→ Y-10",   0, -10),
        ]
        self._sq_idx = 0
        self._sq_wp = waypoints
        self._sq_start_x = self._x
        self._sq_start_y = self._y
        self._run_square_next()

    def _run_square_next(self):
        if self._sq_idx >= len(self._sq_wp):
            self._check_square_result()
            return

        name, dx, dy = self._sq_wp[self._sq_idx]
        print(f"  {name}  ", end="", flush=True)
        self._do_jog_and_wait(dx, dy, self._feed, lambda: self._sq_step_done())

    def _sq_step_done(self):
        print("✅")
        self._sq_idx += 1
        QTimer.singleShot(200, self._run_square_next)

    def _check_square_result(self):
        dx = self._x - self._sq_start_x
        dy = self._y - self._sq_start_y
        dist = (dx**2 + dy**2) ** 0.5

        print(f"\n[结果] 起点 (X={self._sq_start_x:.3f}, Y={self._sq_start_y:.3f})")
        print(f"       终点 (X={self._x:.3f}, Y={self._y:.3f})")
        print(f"       偏差: {dist:.3f} mm")

        if dist < 1.0:
            print("✅ 回原点精度良好（偏差 < 1mm）")
        elif dist < 3.0:
            print("⚠️  偏差在 1-3mm 之间，可能有轻微丢步或间隙")
        else:
            print("❌ 偏差 > 3mm，可能有明显丢步")
            print("  建议检查：传动皮带松紧、电机电流、是否卡顿")

        self._finish()

    # ── 核心：点动 + 等待 Idle ────────────────────────────────

    def _do_jog_and_wait(self, dx: float, dy: float, feed: float, callback):
        """发送相对点动，然后等待 GRBL 回到 Idle 再回调。"""
        self._next_callback = callback
        self._waiting = True
        self.cnc.jog(dx, dy, feed)
        QTimer.singleShot(100, self._poll_idle)

    def _jog_z_and_wait(self, dz: float, callback):
        self._next_callback = callback
        self._waiting = True
        self.cnc.jog_pen(dz, self._z_feed)
        QTimer.singleShot(100, self._poll_idle)

    def _poll_idle(self):
        if not self._waiting or not hasattr(self, '_next_callback'):
            return
        if self._state == "Idle":
            self._waiting = False
            cb = self._next_callback
            self._next_callback = None
            if cb:
                QTimer.singleShot(200, cb)
        else:
            QTimer.singleShot(150, self._poll_idle)

    # ── 结束 ──────────────────────────────────────────────────

    def _finish(self):
        print()
        print("═" * 50)
        print("  测试流程结束。")
        print("  如有异常，请检查机械结构后再重试。")
        print("═" * 50)

        self.cnc.close_port()
        QCoreApplication.quit()


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

def main():
    app = QCoreApplication(sys.argv)
    port = select_port()
    runner = SafeSmokeTest(port)
    QTimer.singleShot(100, runner.start)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
