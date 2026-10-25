#!/usr/bin/env python3
"""
handover-buzzer 上层封装接口
=============================
低电平触发有源蜂鸣器的控制封装，供上层模块（handover-puzzle、
handover-writer 等）直接 import 使用。

硬件:
  - 物理引脚 32 = PM5 (gpiochip1 线 37, R_PIO 控制器)
  - 低电平触发: 输出低电平 = 响, 输出高电平 = 静音
  - 板级已外接 10kΩ 上拉电阻到 3.3V，解决上电瞬间引脚不受控导致
    "一直滴滴叫" 的问题；未占用时由电阻保持高电平 → 静音

用法（上层模块）:
    import sys
    sys.path.insert(0, "<仓库根>/handover-buzzer")   # 按实际克隆位置替换
    from buzzer import Buzzer

    with Buzzer() as buzzer:      # 自动恢复静音并释放 GPIO
        buzzer.beep()             # 短响一声
        buzzer.beep(0.5, 3)       # 每次 0.5s, 间隔 0.2s, 共 3 声
        buzzer.on()               # 持续响
        buzzer.off()              # 静音
"""
import time

from periphery import GPIO

CHIP = "/dev/gpiochip1"   # R_PIO 控制器
LINE = 37                 # PM5（物理引脚 32）


class Buzzer:
    """低电平触发蜂鸣器。输出高电平为静音（默认），输出低电平为响。"""

    def __init__(self, chip=CHIP, line=LINE):
        # direction "high": 输出方向且初始值为高电平（静音），避免先输出
        # 低电平造成"哒"一声的启动杂音。
        self._gpio = GPIO(chip, line, "high")

    def on(self):
        """持续响（输出低电平）。"""
        self._gpio.write(False)

    def off(self):
        """静音（输出高电平）。"""
        self._gpio.write(True)

    def beep(self, duration=0.1, times=1, interval=0.2):
        """按节奏响若干次（阻塞式，调用期间占用引脚）。

        Args:
            duration: 每次响的时长（秒）。
            times:    响的次数。
            interval: 两次之间的静音间隔（秒）。
        """
        for i in range(times):
            self.on()
            time.sleep(duration)
            self.off()
            if i < times - 1:
                time.sleep(interval)

    def close(self):
        """恢复静音并释放 GPIO。"""
        try:
            self._gpio.write(True)
        except Exception:
            pass
        self._gpio.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
