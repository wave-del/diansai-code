#!/usr/bin/env python3
"""电磁铁Python API最小接入示例。"""

from electromagnet import ElectromagnetController


def main() -> None:
    magnet = ElectromagnetController()
    try:
        magnet.open()       # PL6初始化为低电平，电磁铁松开
        magnet.engage()     # 高电平，保持吸合
        magnet.release()    # 低电平，松开
        magnet.pulse(0.5)   # 吸合0.5秒，随后自动松开
    finally:
        magnet.close()      # 再次尝试低电平，然后释放GPIO


if __name__ == "__main__":
    main()
