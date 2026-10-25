#!/usr/bin/env python3
"""
handover-buzzer 蜂鸣器测试脚本
===============================
驱动低电平触发的有源蜂鸣器:
  - 物理引脚 32 = PM5 (gpiochip1 线 37, R_PIO)
  - 输出低电平 = 响, 输出高电平 = 静音
  - 板级外接 10kΩ 上拉电阻 → 引脚未被占用时保持高电平(静音)

测试内容:
  1. 释放状态下引脚电平应为高(静音), 验证上拉电阻接线
  2. 长响 1 秒
  3. 短响 3 声
  4. 验证完成提示 5 声短响

用法:
    .venv/bin/python gpio_buzzer_test.py
"""
import time

from periphery import GPIO

from buzzer import Buzzer

CHIP = "/dev/gpiochip1"
LINE = 37


def check_idle_silent():
    """引脚未占用时，硬件上拉应使其保持高电平（静音）。"""
    g = GPIO(CHIP, LINE, "in")
    high = g.read()
    g.close()
    ok = high is True
    print(f"[检查] 释放状态下引脚电平: {'高(静音) ✓' if ok else '低(蜂鸣器在响!) ✗'}")
    return ok


def main():
    print("蜂鸣器测试开始 (引脚32/PM5, gpiochip1 线37, 低电平触发)\n")

    if not check_idle_silent():
        print("异常: 未占用时蜂鸣器应为静音，请检查 10kΩ 上拉电阻接线！")
        return

    with Buzzer() as buzzer:
        print("[1/3] 长响 1 秒 ...")
        buzzer.beep(duration=1.0)
        time.sleep(0.5)

        print("[2/3] 短响 3 声 (0.2s/声, 间隔 0.2s) ...")
        buzzer.beep(duration=0.2, times=3, interval=0.2)
        time.sleep(0.5)

        print("[3/3] 验证完成提示: 5 声短响 ...")
        buzzer.beep(duration=0.1, times=5, interval=0.15)

    print("\n测试完成 ✓")
    if check_idle_silent():
        print("退出后蜂鸣器已恢复静音 ✓")
    else:
        print("提示: 退出后引脚被释放，应由上拉电阻恢复为静音，请检查接线！")


if __name__ == "__main__":
    main()
