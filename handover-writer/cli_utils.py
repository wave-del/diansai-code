"""交付脚本共用的命令行工具。"""

from __future__ import annotations

try:
    from .grbl_controller import list_serial_ports
except ImportError:
    from grbl_controller import list_serial_ports


def select_port(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    ports = list_serial_ports()
    if not ports:
        raise RuntimeError("未检测到串口，请检查 USB 连接")
    print("可用串口:")
    for index, port in enumerate(ports):
        print(f"  [{index}] {port['name']} — {port['desc']}")
    while True:
        try:
            choice = int(input("选择串口号: "))
            if 0 <= choice < len(ports):
                return ports[choice]["name"]
        except ValueError:
            pass

