#!/usr/bin/env python3
"""
handover-button 按钮 GPIO 测试脚本
====================================
使用 python-periphery 读取三个按键（按下时接地，脚本运行时使能内部上拉，
悬空稳定读高电平）。

按键接线（40 pin 排针物理引脚 -> 信号 -> /dev/gpiochip0 线号）:
  物理引脚 36 -> PB4 -> gpiochip0 线 36
  物理引脚 38 -> PB8 -> gpiochip0 线 40
  物理引脚 40 -> PB7 -> gpiochip0 线 39

线号依据: 官方示例 "PB7 = /dev/gpiochip0 线 39"（GPIOB 引脚连续，
PB4=36, PB8=40）。若某引脚读数异常，用 `gpioinfo 0` 或官方手册
核对后调整 BUTTONS 表中的 line 值即可。

用法:
    python3 gpio_button_test.py
"""
from periphery import GPIO
import time

CHIP = "/dev/gpiochip0"
POLL_INTERVAL = 0.05     # 轮询间隔 50ms
DEBOUNCE_MS = 0.02       # 软件消抖窗口 20ms
DEBOUNCE_STEP = 0.002    # 消抖窗口内采样间隔 2ms

# 三个按钮: key 用于区分按钮, phys_pin 为排针物理引脚, signal 为信号名, line 为 gpiochip 线号
BUTTONS = [
    {"key": 1, "label": "按钮1", "phys_pin": 36, "signal": "PB4", "line": 36},
    {"key": 2, "label": "按钮2", "phys_pin": 38, "signal": "PB8", "line": 40},
    {"key": 3, "label": "按钮3", "phys_pin": 40, "signal": "PB7", "line": 39},
]


def is_stable(gpio, expected):
    """在 DEBOUNCE_MS 窗口内连续采样，始终等于 expected 才确认电平稳定。

    机械按键抖动可持续几十 ms，且可能正好落在单次延时重读的采样点上，
    这里改为 20ms 窗口内每 2ms 采一次样，全部一致才算有效跳变。
    """
    end = time.monotonic() + DEBOUNCE_MS
    while True:
        if gpio.read() != expected:
            return False
        remaining = end - time.monotonic()
        if remaining <= 0:
            return True
        time.sleep(min(DEBOUNCE_STEP, remaining))


def open_buttons():
    """将所有按键配置为输入模式，返回 GPIO 句柄列表。"""
    handles = []
    try:
        for btn in BUTTONS:
            # bias="pull_up": 运行时使能内部上拉，浮空引脚稳定读高，避免误触发
            gpio = GPIO(CHIP, btn["line"], "in", bias="pull_up")
            handles.append(gpio)
            level = "高电平(未按下)" if gpio.read() else "低电平(按下)"
            print(f"[{btn['key']}] {btn['label']}: 引脚{btn['phys_pin']}({btn['signal']}) "
                  f"-> {CHIP} 线{btn['line']}  当前: {level}")
    except Exception as e:
        for h in handles:
            h.close()
        raise
    return handles


def main():
    handles = open_buttons()
    # 上一状态: True=高(未按下), False=低(按下)
    last = [h.read() for h in handles]

    print(f"\n开始轮询按键（软件消抖 {DEBOUNCE_MS*1000:.0f}ms，按下/松开时打印，Ctrl+C 退出）\n")
    try:
        while True:
            for i, (gpio, btn, prev) in enumerate(zip(handles, BUTTONS, last)):
                cur = gpio.read()
                # 电平变化后进入 20ms 消抖窗口，窗口内全部采样一致才接受
                if cur != prev and is_stable(gpio, cur):
                    last[i] = cur
                    if cur is False:
                        print(f"[{btn['key']}] {btn['label']} 按下  "
                              f"(引脚{btn['phys_pin']}/{btn['signal']}) -> LOW")
                    else:
                        print(f"[{btn['key']}] {btn['label']} 松开  "
                              f"(引脚{btn['phys_pin']}/{btn['signal']}) -> HIGH")
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\n程序退出")
    finally:
        for h in handles:
            h.close()


if __name__ == "__main__":
    main()
