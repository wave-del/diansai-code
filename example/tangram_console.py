#!/usr/bin/env python3
"""TangramAPI 真实硬件交互测试台；所有运动命令使用蓝纸右上角坐标。"""
from __future__ import annotations
import argparse, signal
from tangram_api import TangramAPI

HELP = """\norigin                  机构在写字机自身原点时建立原点（不运动）
home [FEED]             回写字机自身原点，不应用蓝纸偏置
xyz X Y LIFT [FEED]     安全移动；Z 正数为抬起 mm，XY速度默认2500mm/min
z LIFT                  仅设置 Z 抬起高度 0..15mm（固定1500mm/min）
foc ANGLE [RPM]         FOC 绝对 -360..360°；-50 等同 310，自动就近旋转
rotate DELTA [RPM]      FOC 相对 -180..180°，保留正负方向
zero [RPM]              FOC 回驱动板机械零点
magnet on|off           高电平吸合 / 低电平松开
quit / q                松开电磁铁并退出；XY、FOC 保持使能\n"""

def main() -> int:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--writer-port", default="/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0")
    parser.add_argument("--motor-port", default="/dev/ttyAS3")
    args = parser.parse_args()
    api = TangramAPI(args.writer_port, args.motor_port)
    try:
        api.connect(); print("已连接：未移动。请确认机构在写字机自身原点后输入 origin。" + HELP)
        while True:
            words = input("tangram> ").split()
            if not words: continue
            command = words[0].lower()
            if command in {"q", "quit", "exit"}: break
            if command in {"help", "?"}: print(HELP); continue
            if command == "origin" and len(words) == 1:
                if input("确认机构在写字机自身原点，输入 ORIGIN： ").upper() == "ORIGIN": api.establish_origin(); print("原点已建立")
                continue
            if command == "home" and len(words) in (1, 2):
                api.return_machine_origin(float(words[1]) if len(words)==2 else 2500)
                continue
            if command == "xyz" and len(words) in (4, 5):
                api.move_xyz(float(words[1]), float(words[2]), float(words[3]), xy_feed_mm_min=float(words[4]) if len(words)==5 else 2500); continue
            if command == "z" and len(words) == 2: api.set_lift(float(words[1])); continue
            if command == "foc" and len(words) in (2, 3): api.foc_angle(float(words[1]), int(words[2]) if len(words)==3 else 15); continue
            if command == "rotate" and len(words) in (2, 3): api.foc_rotate(float(words[1]), int(words[2]) if len(words)==3 else 15); continue
            if command == "zero" and len(words) in (1, 2): api.foc_zero(int(words[1]) if len(words)==2 else 15); continue
            if command == "magnet" and len(words)==2: (api.magnet_on() if words[1].lower()=="on" else api.magnet_off() if words[1].lower()=="off" else (_ for _ in ()).throw(ValueError("仅 on/off"))); continue
            print("⛔ 命令格式错误。" + HELP)
    except KeyboardInterrupt: print("\n已中断，不再发送运动命令。")
    except Exception as exc: print(f"⛔ {exc}")
    finally: api.close()
    return 0
if __name__ == "__main__": raise SystemExit(main())
